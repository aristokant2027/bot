"""Tiny dependency-free preview renderer (point-splat z-buffer, orthographic)."""
import os
import warnings
import numpy as np
from PIL import Image

COL = {"S": (198, 201, 206), "W": (236, 233, 224), "K": (46, 46, 52), "H": (230, 150, 40)}
BG = (244, 244, 248)
warnings.filterwarnings("ignore")


def _view(az, el):
    az, el = np.radians(az), np.radians(el)
    fwd = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), -np.sin(el)])
    right = np.cross(fwd, (0, 0, 1)); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return right, up, fwd


def render(items, path, az=0, el=8, scale=0.6, ss=2, pad=30):
    """items: list of (verts Nx3, faces Mx3, color key). Camera looks along +Y at az=0."""
    right, up, fwd = _view(az, el)
    s = scale * ss
    allv = np.vstack([v for v, f, c in items])
    u, w = allv @ right, allv @ up
    u0, w1 = u.min() - pad, w.max() + pad
    W = int((u.max() + pad - u0) * s) + 1; H = int((w1 - (w.min() - pad)) * s) + 1
    zb = np.full(H * W, np.inf, np.float32)
    img = np.tile(np.array(BG, np.float32), (H * W, 1))
    light = np.array([-0.45, -0.75, 0.6]); light /= np.linalg.norm(light)
    for v, f, c in items:
        tri = v[f]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        vis = n @ fwd < 0.05
        tri, n = tri[vis], n[vis]
        shade = 0.38 + 0.62 * np.clip(n @ light, 0, 1) + 0.12 * np.clip(-(n @ fwd), 0, 1)
        col = np.array(COL[c], np.float32)[None] * np.minimum(shade, 1.1)[:, None]
        P = np.stack([(tri @ right - u0) * s, (w1 - tri @ up) * s, tri @ fwd], -1)
        e = np.maximum.reduce([np.linalg.norm(P[:, i, :2] - P[:, j, :2], axis=1)
                               for i, j in ((0, 1), (1, 2), (2, 0))])
        k = np.minimum(np.ceil(e * 1.5).astype(int) + 1, 4000)
        for kk in np.unique(k):
            sel = k == kk
            a, b = np.meshgrid(np.arange(kk + 1), np.arange(kk + 1), indexing="ij")
            m = a + b <= kk
            a, b = a[m] / kk, b[m] / kk
            Q = (P[sel][:, None, 0] * (1 - a - b)[None, :, None]
                 + P[sel][:, None, 1] * a[None, :, None] + P[sel][:, None, 2] * b[None, :, None])
            px = np.clip(np.round(Q[..., 0]).astype(int), 0, W - 1)
            py = np.clip(np.round(Q[..., 1]).astype(int), 0, H - 1)
            idx = (py * W + px).ravel(); z = Q[..., 2].ravel().astype(np.float32)
            cc = np.repeat(col[sel], Q.shape[1], axis=0)
            order = np.argsort(-z)          # far first, near written last
            idx, z, cc = idx[order], z[order], cc[order]
            better = z < zb[idx] + 1e-3
            idx, z, cc = idx[better], z[better], cc[better]
            zb[idx] = z; img[idx] = cc
    zimg = zb.reshape(H, W); im = img.reshape(H, W, 3)
    # depth-discontinuity outlines
    zf = np.where(np.isinf(zimg), 1e5, zimg)
    gx = np.abs(np.diff(zf, axis=1, prepend=zf[:, :1])); gy = np.abs(np.diff(zf, axis=0, prepend=zf[:1]))
    edge = (np.maximum(gx, gy) > 6)
    im[edge] *= 0.45
    out = Image.fromarray(np.clip(im, 0, 255).astype(np.uint8)).resize((W // ss, H // ss), Image.LANCZOS)
    out.save(path)
    return out


def _items(world, explode=0.0, skip=()):
    items = []
    c0 = np.array([0, 0, 700.0])
    for name, (man, col) in world.items():
        if any(name.startswith(s) for s in skip):
            continue
        mesh = man.to_mesh()
        v = np.asarray(mesh.vert_properties)[:, :3].astype(np.float64)
        f = np.asarray(mesh.tri_verts)
        if explode:
            c = (v.min(0) + v.max(0)) / 2
            v = v + (c - c0) * explode * np.array([1.0, 0.6, 0.55])
        items.append((v, f, col))
    return items


def render_all(world, docs):
    full = _items(world, skip=("tenon_",))
    views = [("front", 0, 6), ("three_quarter", -32, 12), ("back", 180, 6), ("side", -90, 6)]
    ims = [render(full, os.path.join(docs, f"preview_{n}.png") if n == "three_quarter" else os.path.join(docs, ".tmp.png"), az, el) for n, az, el in views]
    if os.path.exists(os.path.join(docs, ".tmp.png")):
        os.remove(os.path.join(docs, ".tmp.png"))
    h = max(i.height for i in ims)
    sheet = Image.new("RGB", (sum(i.width for i in ims), h), BG)
    x = 0
    for i in ims:
        sheet.paste(i, (x, h - i.height)); x += i.width
    sheet.save(os.path.join(docs, "preview_sheet.png"))
    render(_items(world, explode=0.55), os.path.join(docs, "preview_exploded.png"), -25, 10, scale=0.42)
    head = [(v, f, c) for (v, f, c), n in zip(_items(world), world) if v[:, 2].min() > 1100 and v[:, 0].min() > -110 and not n.startswith("tenon")]
    render(head, os.path.join(docs, "preview_bust.png"), -28, 8, scale=1.6)
