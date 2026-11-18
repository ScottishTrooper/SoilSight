"""Frozen-backbone and handcrafted features for every cached photo.

Per photo we store, for each backbone:
  <bb>.crop    mean embedding of the 12 full-res crops        (fine texture)
  <bb>.ovw     embedding of the soil overview                 (coarse structure)
and for ViT backbones also
  <bb>.ptok    mean patch token of the overview                (feeds the heatmap head)
plus handcrafted colour/texture statistics (crop mean and overview).

The same frozen features power the prediction heads, the similar-sample index and the
heatmap, so the GPU is only needed once, for this extraction pass.
"""
from __future__ import annotations

import concurrent.futures as cf
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

BACKBONES = {
    "effnet_b0": ("timm", "efficientnet_b0.ra_in1k"),
    "dinov3_convnext_t": ("hf", "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"),
    "dinov3_vits16": ("hf", "facebook/dinov3-vits16-pretrain-lvd1689m"),
}
OVW_LONG = 1024  # overview is resized to this long side (multiple of 16) before the backbone


# ---------------------------------------------------------------- handcrafted
def handcrafted(rgb: np.ndarray) -> np.ndarray:
    """21 colour/texture numbers: Lab mean/std, multi-scale LoG energy, gradients, grey histogram, shadow fraction."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    var = g.var() + 1e-6
    log_e = []
    for s in (1, 2, 4, 8):
        b = cv2.GaussianBlur(g, (0, 0), s)
        log_e.append(float((cv2.Laplacian(b, cv2.CV_32F) ** 2).mean() * s ** 4 / var))
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx * gx + gy * gy)
    hist = np.histogram(g, bins=8, range=(0, 1))[0] / g.size
    dark = float((g < 0.6 * np.median(g)).mean())
    return np.concatenate([lab.mean(0), lab.std(0), log_e, [mag.mean(), mag.std()], hist, [dark]]).astype(np.float32)


# ---------------------------------------------------------------- backbones
class Backbone:
    """Uniform wrapper: embed(list of RGB uint8 arrays) -> (pooled [N,D], mean patch token [N,Dp] or None)."""

    def __new__(cls, name: str, *a, **k):
        if name == "stub":
            return NumpyStub()
        return super().__new__(cls)

    def __init__(self, name: str, device: str = "cuda", fp16: bool = True):
        import torch
        self.torch, self.name, self.device = torch, name, device
        self.fp16 = fp16 and device.startswith("cuda")
        kind, ident = BACKBONES.get(name, ("stub", None))
        self.kind = kind
        if kind == "timm":
            import timm
            self.model = timm.create_model(ident, pretrained=True, num_classes=0)
        elif kind == "hf":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(ident)
            self.n_reg = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
            self.is_vit = "vit" in ident
        else:
            raise ValueError(f"unknown backbone {name}")
        self.model.eval().to(device)

    def _tensor(self, imgs):
        x = np.stack([(im.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD for im in imgs])
        return self.torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device)

    def embed(self, imgs: list[np.ndarray]):
        torch = self.torch
        x = self._tensor(imgs)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            if self.kind == "timm":
                return self.model(x).float().cpu().numpy(), None
            if self.kind == "hf":
                out = self.model(pixel_values=x)
                pooled = out.pooler_output.float().cpu().numpy()
                if self.is_vit:
                    pt = out.last_hidden_state[:, 1 + self.n_reg:].float().mean(1).cpu().numpy()
                    return pooled, pt
                return pooled, None

    def patch_tokens(self, rgb: np.ndarray):
        """Full patch-token grid for one image (used by the heatmap at inference). Returns (h, w, D)."""
        torch = self.torch
        x = self._tensor([rgb])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            if self.kind == "hf" and self.is_vit:
                t = self.model(pixel_values=x).last_hidden_state[0, 1 + self.n_reg:].float().cpu().numpy()
                h, w = rgb.shape[0] // 16, rgb.shape[1] // 16
                return t.reshape(h, w, -1)
            raise ValueError("patch tokens need a ViT backbone")


class NumpyStub:
    """Torch-free stand-in for offline tests: fixed random projection of 16x16 patch colour statistics."""
    kind, is_vit, name = "stub", True, "stub"

    def __init__(self):
        self.W = np.random.default_rng(0).normal(0, 1, (6, 24)).astype(np.float32)

    def _tokens(self, rgb):
        h, w = rgb.shape[0] // 16, rgb.shape[1] // 16
        p = rgb[:h * 16, :w * 16].astype(np.float32).reshape(h, 16, w, 16, 3) / 255.0
        stats = np.concatenate([p.mean((1, 3)), p.std((1, 3))], -1)
        return np.tanh(stats @ self.W)

    def embed(self, imgs):
        toks = [self._tokens(im) for im in imgs]
        pooled = np.stack([t.reshape(-1, t.shape[-1]).max(0) for t in toks])
        return pooled, np.stack([t.reshape(-1, t.shape[-1]).mean(0) for t in toks])

    def patch_tokens(self, rgb):
        return self._tokens(rgb)


def resize_overview(rgb: np.ndarray, long_side: int = OVW_LONG) -> np.ndarray:
    h, w = rgb.shape[:2]
    f = long_side / max(h, w)
    nh, nw = max(16, int(round(h * f / 16)) * 16), max(16, int(round(w * f / 16)) * 16)
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- shard reading
def iter_photos(tar_paths: list[Path]):
    """Yield (meta, overview_rgb, [crop_rgb]) per photo from view shards, decoding in a thread pool."""
    dec = lambda b: cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    with cf.ThreadPoolExecutor(4) as ex:
        for tp in tar_paths:
            groups: dict[str, dict] = defaultdict(dict)
            with tarfile.open(tp) as t:
                for m in t:
                    key, _, ext = m.name.partition(".")
                    groups[key][ext] = t.extractfile(m).read()
            for key, g in groups.items():
                meta = json.loads(g["json"])
                crops_b = [g[k] for k in sorted(g) if k.startswith("crop")]
                ovw, *crops = list(ex.map(dec, [g["overview.jpg"], *crops_b]))
                yield meta, ovw, crops


def extract(tar_paths: list[Path], backbones: list[str], device: str = "cuda", log=print,
            limit: int | None = None) -> dict:
    """Run every backbone + handcrafted features over all photos. Returns dict of arrays + 'index' rows."""
    models = {b: Backbone(b, device) for b in backbones}
    feats: dict[str, list] = defaultdict(list)
    index = []
    for n, (meta, ovw, crops) in enumerate(iter_photos(tar_paths)):
        if limit and n >= limit:
            break
        ov = resize_overview(ovw)
        for b, m in models.items():
            pc, ptc = m.embed(crops)
            po, pto = m.embed([ov])
            feats[f"{b}.crop"].append(pc.mean(0))
            feats[f"{b}.ovw"].append(po[0])
            if pto is not None and getattr(m, "is_vit", False):
                feats[f"{b}.ptok"].append(pto[0])
        feats["hand.crop"].append(np.mean([handcrafted(c) for c in crops], 0))
        feats["hand.ovw"].append(handcrafted(ov))
        index.append({k: meta[k] for k in ("key", "sample_id", "state", "tray", "split", "fold")})
        if (n + 1) % 200 == 0:
            log(f"  {n + 1} photos")
    out = {k: np.stack(v).astype(np.float32) for k, v in feats.items()}
    out["index"] = index
    log(f"extracted {len(index)} photos; arrays: " + ", ".join(f"{k}{v.shape}" for k, v in out.items() if k != "index"))
    return out


def save_features(f: dict, path: Path) -> None:
    arrays = {k: v for k, v in f.items() if k != "index"}
    np.savez_compressed(path, index_json=np.array(json.dumps(f["index"])), **arrays)


def load_features(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files if k != "index_json"}
    out["index"] = json.loads(str(z["index_json"]))
    return out
