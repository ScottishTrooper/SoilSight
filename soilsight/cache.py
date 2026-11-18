"""Batch cache builder: download -> verify -> make views -> pack shards -> upload -> delete.

Shard layout in the Hugging Face dataset repo (WebDataset-style tars):
  views/batch_XXX.tar    {key}.overview.jpg, {key}.cropNN.jpg, {key}.mask.png, {key}.json
  halfres/batch_XXX.tar  {key}.half.jpg
  meta/batch_XXX.jsonl   one JSON line per photo (+ one per calibration photo with its scale)
  qc/batch_XXX.jpg       contact sheet: every photo with its soil outline and crop boxes

A batch is "done" when its meta/batch_XXX.jsonl exists in the repo (uploaded in the
same commit as the shards), so a rerun skips it.
"""
from __future__ import annotations

import concurrent.futures as cf
import contextlib
import hashlib
import io
import json
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .scale import sample_scale
from .views import WORK_SCALE, make_views, soil_mask


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(src_base: str, rel: str, dest: Path, sha: str, retries: int = 4) -> Path:
    """Download (or copy, if src_base is a local folder) and verify sha256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256_file(dest) == sha:
        return dest
    last = None
    for attempt in range(retries):
        try:
            if src_base.startswith("http"):
                url = src_base.rstrip("/") + "/" + rel
                req = urllib.request.Request(url, headers={"User-Agent": "soilsight-cache/0.1"})
                with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
            else:
                shutil.copyfile(Path(src_base) / rel, dest)
            got = sha256_file(dest)
            if got == sha:
                return dest
            last = f"checksum mismatch ({got[:12]} != {sha[:12]})"
        except Exception as e:  # network hiccup: back off and retry
            last = repr(e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to fetch {rel}: {last}")


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    ti.mtime = 0
    tar.addfile(ti, io.BytesIO(data))


def _qc_tile(img: np.ndarray, meta: dict, label: str, width: int = 320) -> np.ndarray:
    sm = cv2.resize(img, None, fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)
    mask, _ = soil_mask(img)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sm, cnts, -1, (0, 255, 255), 3)
    for x0, y0, x1, y1 in meta["crop_boxes_xyxy"]:
        cv2.rectangle(sm, (int(x0 * WORK_SCALE), int(y0 * WORK_SCALE)),
                      (int(x1 * WORK_SCALE), int(y1 * WORK_SCALE)), (0, 0, 255), 3)
    f = width / sm.shape[1]
    t = cv2.resize(sm, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    flag = " !" if meta["qc_flags"] else ""
    cv2.rectangle(t, (0, 0), (width, 18), (0, 0, 0), -1)
    cv2.putText(t, label + flag, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 0, 255) if flag else (255, 255, 255), 1, cv2.LINE_AA)
    return t


def _contact_sheet(tiles: list[np.ndarray], cols: int = 8) -> bytes:
    if not tiles:
        return b""
    h, w = tiles[0].shape[:2]
    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.full((rows * h, cols * w, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * h:r * h + t.shape[0], c * w:c * w + t.shape[1]] = t
    return cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def build_batch(rows: list[dict], src_base: str, work: Path, n_crops: int = 12,
                with_halfres: bool = True, threads: int = 6, log=print) -> dict:
    """Process one batch. rows: manifest rows (dicts) for this batch. Returns local file paths."""
    b = int(rows[0]["batch"])
    raw = work / "raw"
    out = work / "out"
    for d in ("views", "halfres", "meta", "qc"):
        (out / d).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with cf.ThreadPoolExecutor(threads) as ex:
        futs = {ex.submit(fetch, src_base, r["file"], raw / r["file"], r["sha256"]): r for r in rows}
        for f in cf.as_completed(futs):
            f.result()  # raise on failure
    gb = sum((raw / r["file"]).stat().st_size for r in rows) / 1e9
    log(f"batch {b:03d}: downloaded {len(rows)} files, {gb:.1f} GB in {time.time() - t0:.0f}s")

    paths = {k: out / k / f"batch_{b:03d}.{ext}" for k, ext in
             [("views", "tar"), ("halfres", "tar"), ("meta", "jsonl"), ("qc", "jpg")]}
    tiles, n_flag = [], 0
    t1 = time.time()
    with tarfile.open(paths["views"], "w") as tv, \
         (tarfile.open(paths["halfres"], "w") if with_halfres else contextlib.nullcontext()) as th, \
         open(paths["meta"], "w") as fm:
        for r in rows:
            p = raw / r["file"]
            if r["kind"] == "calibration":
                g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                rec = {"kind": "calibration", "sample_id": r["sample_id"], "file": r["file"], **sample_scale(g)}
                fm.write(json.dumps(rec) + "\n")
                p.unlink()
                continue
            img = cv2.imread(str(p))
            if img is None:
                raise RuntimeError(f"could not decode {r['file']}")
            key = Path(r["file"]).stem
            v = make_views(img, key, n_crops=n_crops, with_halfres=with_halfres)
            _add(tv, f"{key}.overview.jpg", v["overview"])
            for i, c in enumerate(v["crops"]):
                _add(tv, f"{key}.crop{i:02d}.jpg", c)
            _add(tv, f"{key}.mask.png", v["mask_png"])
            meta = {"kind": "soil", "key": key, "sample_id": r["sample_id"], "file": r["file"],
                    "state": r["dry_or_moist"], "tray": r["tray"], "split": r["split"],
                    "fold": int(r["fold"]), **v["meta"]}
            _add(tv, f"{key}.json", json.dumps(meta).encode())
            if with_halfres:
                _add(th, f"{key}.half.jpg", v["halfres"])
            fm.write(json.dumps(meta) + "\n")
            n_flag += bool(meta["qc_flags"])
            tiles.append(_qc_tile(img, meta, key.replace("_", " ")))
            p.unlink()  # free disk as we go
    paths["qc"].write_bytes(_contact_sheet(tiles))
    if not with_halfres:
        paths.pop("halfres")
    log(f"batch {b:03d}: processed in {time.time() - t1:.0f}s, {n_flag} photo(s) flagged for QC")
    shutil.rmtree(raw, ignore_errors=True)
    return paths
