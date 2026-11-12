"""HUMAN+ 配色的 Unitree G1 静态摆件：把 MakerWorld 的 G1Static.stl 放大到真机身高，
分色、拆件、开榫孔、加底座和背后支撑柱，导出全部可打印 STL 与报告。

usage: python3 generator/g1_build.py [--out .] [--no-render]
源模型：source/g1/G1Static.zip.part-*（自动拼接解压到 work/）。
"""
import argparse, csv, glob, io, json, os, struct, sys, time, warnings, zipfile
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import manifold3d as m3
from manifold3d import Manifold, Mesh, CrossSection

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ------------------------------------------------------------------ 参数
HEAD_TOP = 1320.0          # 头顶离地高度 (mm)，宇树 G1 官方站立身高
SIMPLIFY = 0.03            # 网格简化容差 (mm)
CLR = 0.20                 # 榫孔单边间隙
WALL = 4.0                 # 榫孔四周最小壁厚
SIZES = (24, 16, 12, 8)    # 可选榫条截面
X0 = 9.0                   # 身体左右对称面 (模型略偏)
BEDS = [("A1 / X2D", (256, 256, 256)), ("A2L", (330, 320, 325))]
COLOR_DIR = {"W": "1_white", "K": "2_black", "P": "3_purple", "H": "4_connectors"}
# HUMAN+ 配色 (取自效果图色卡)：白壳 / 炭黑关节 / 紫色点缀
PALETTE = {"W": "#ECE9E0", "K": "#2E2E34", "P": "#5C34D6"}
Z_WAISTCOL = 770.0         # 腰柱 (黑) 与骨盆 (白) 的分界
# 胸口 logo 贴纸位：压一块平底浅槽 (盖掉原模型的 UNITREE 浮雕字)
LOGO = dict(cx=4.0, cz=1050.0, w=112.0, h=22.0, min_depth=0.4, r=3.0)

# 躯干/腿的水平切面 (z, mm)
Z_ANKLE, Z_SHIN, Z_KNEE, Z_THIGH = 115, 300, 400, 630
Z_WAIST, Z_NECK0, Z_NECK1 = 820, 1135, 1152
X_HIP_L, X_HIP_R = 73.0, -59.0          # 髋壳与黑色骨盆的分界
X_ARM_L, X_ARM_R = 121.0, -103.0        # 肩与躯干的分界 (最细处)
# 左臂 (机器人左手 = +X，前伸)
ZL_ELBOW, ZL_SLEEVE = 897.0, 1003.0
CL = np.array([210.4, -47.9, 730.2]); DL = np.array([0.088, -0.819, -0.568])
TL = dict(ring0=-147, ring1=-115, wrist0=-3, wrist1=17)
# 右臂 (举起)
C2 = np.array([-165.4, 32.7, 1112.1]); D2 = np.array([-0.257, -0.842, 0.475])
T2 = dict(sleeve0=-35, sleeve1=64)
CR = np.array([-209.0, -64.7, 1396.3]); DR = np.array([-0.042, -0.046, 0.998])
TR = dict(ring0=-157, ring1=-125, wrist0=-13, wrist1=7)
for _d in (DL, D2, DR):
    _d /= np.linalg.norm(_d)

BASE_W, BASE_D, BASE_T, BASE_R = 440.0, 420.0, 30.0, 30.0
POST = 30.0                # 背后支撑柱截面
POST_Z = (0.0, 240.0, 480.0, 725.0)
POST_TIE_Z = 700.0         # 支撑柱与骨盆的横榫高度


# ------------------------------------------------------------ 基础工具
def load_source():
    path = os.path.join(ROOT, "work", "G1Static.stl")
    if not os.path.exists(path):
        parts = sorted(glob.glob(os.path.join(ROOT, "source", "g1", "G1Static.zip.part-*")))
        assert parts, "缺少 source/g1/G1Static.zip.part-*"
        blob = b"".join(open(p, "rb").read() for p in parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            open(path, "wb").write(z.read("G1Static.stl"))
    raw = open(path, "rb").read()
    n = struct.unpack("<I", raw[80:84])[0]
    rec = np.frombuffer(raw, offset=84, count=n,
                        dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    pts = rec["v"].reshape(-1, 3)
    v, inv = np.unique(pts, axis=0, return_inverse=True)
    f = inv.reshape(-1, 3)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    v = v.astype(np.float64) * 1000.0                       # 源文件单位是米
    v = np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])       # Y 朝上 -> Z 朝上，正面朝 -Y
    v -= [(v[:, 0].min() + v[:, 0].max()) / 2, (v[:, 1].min() + v[:, 1].max()) / 2, v[:, 2].min()]
    head = v[np.abs(v[:, 0]) < 70][:, 2].max()              # 举起的手在 |x|>150，不算身高
    scale = HEAD_TOP / head
    v *= scale
    man = Manifold(Mesh(vert_properties=v.astype(np.float32), tri_verts=f.astype(np.uint32)))
    assert man.status() == m3.Error.NoError, man.status()
    return man.simplify(SIMPLIFY), scale


def frame(n):
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    c = np.eye(3)[np.argmin(np.abs(n))]
    e1 = c - n * (c @ n); e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1), n


def far(man, p, n):
    """保留平面 (p, n) 法向一侧。"""
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    return man.trim_by_plane(tuple(n), float(n @ np.asarray(p, float)))


def near(man, p, n):
    return far(man, p, -np.asarray(n, float))


def slab(man, n, a, b, origin=(0, 0, 0)):
    o = np.asarray(origin, float); n = np.asarray(n, float)
    return near(far(man, o + n * a, n), o + n * b, n)


def biggest(man):
    return max(man.decompose(), key=lambda c: c.volume())


def to_local(man, origin, axis):
    e1, e2, n = frame(axis)
    R = np.vstack([e1, e2, n])
    return man.transform(np.column_stack([R, -R @ np.asarray(origin, float)])), (e1, e2, n)


def place(man, e1, axis, origin):
    e1_, e2_, ax = frame(axis)
    e1 = np.asarray(e1, float); e1 = e1 - ax * (e1 @ ax); e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    return man.transform(np.column_stack([e1, e2, ax, np.asarray(origin, float)]))


def orient(man, down):
    """把 down 方向转到朝床，落到 z=0，绕 z 转到占床面最小，居中。"""
    e1, e2, e3 = frame(-np.asarray(down, float))
    r = man.transform(np.column_stack([np.vstack([e1, e2, e3]), np.zeros(3)]))
    best = None
    for ang in range(0, 180, 5):                # 0° 放得进 A1 就不转
        q = r.rotate((0, 0, ang)); b = q.bounding_box()
        key = max(b[3] - b[0], b[4] - b[1])
        if best is None or key < best[0] - 1e-6:
            best = (key, q)
        if ang == 0 and key <= 250:
            break
    q = best[1]; b = q.bounding_box()
    return q.translate((-(b[0] + b[3]) / 2, -(b[1] + b[4]) / 2, -b[2]))


def write_stl(path, man):
    mesh = man.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3]; f = np.asarray(mesh.tri_verts)
    tri = v[f]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = nrm; rec["v"] = tri
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"Unitree G1 static display 1:1".ljust(80, b" "))
        fh.write(struct.pack("<I", len(f))); fh.write(rec.tobytes())
    return len(f)


# ------------------------------------------------------------ 榫条
class Tenon:
    def __init__(self, name, size, p0, axis, length, e1, parts):
        self.name, self.size, self.p0, self.axis = name, size, np.asarray(p0, float), np.asarray(axis, float)
        self.length, self.e1, self.parts = length, e1, parts

    def cutter(self, grow=CLR, extra=0.0):
        s = self.size + 2 * grow
        c = Manifold.cube((s, s, self.length + 2 * extra), True)
        return place(c, self.e1, self.axis, self.p0 + self.axis * self.length / 2)

    def body(self):
        return place(make_tenon(self.size, self.length - 1.0).rotate((-90, 0, 0)),
                     self.e1, self.axis, self.p0 + self.axis * self.length / 2)


def make_tenon(size, length):
    e, c = min(1.2, size * 0.1), min(2.0, size * 0.15)
    a = Manifold.cube((size, size - 2 * e, length - 2 * c), True)
    b = Manifold.cube((size - 2 * e, size, length - 2 * c), True)
    d = Manifold.cube((size - 2 * c, size - 2 * c, length), True)
    return Manifold.batch_hull([a, b, d]).rotate((90, 0, 0))     # 平躺


def clearance_map(solid, origin, axis, t0, t1, win=70, step=5.0):
    """沿轴每隔 step 取截面，返回各截面“到边界距离”的逐点最小值 (1 mm 栅格)。"""
    loc, fr = to_local(solid, origin, axis)
    n = 2 * win + 1
    mn = np.full((n, n), 1e9)
    for t in np.arange(t0 + 1.0, t1 - 0.5, step):
        cs = loc.slice(float(t))
        img = Image.new("L", (n, n), 0); d = ImageDraw.Draw(img)
        polys = sorted((np.asarray(p) for p in cs.to_polygons()),
                       key=lambda p: -abs(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])))
        for p in polys:
            a = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
            d.polygon([(float(x + win), float(y + win)) for x, y in p], fill=255 if a > 0 else 0)
        mn = np.minimum(mn, ndimage.distance_transform_edt(np.asarray(img) > 0))
    return mn, fr


def auto_tenon(name, solids, parts, origin, axis, t0, t1, sizes=SIZES, wall=WALL):
    """在给定轴向和范围内，自动找壁厚最大的位置并选能放下的最大榫条。"""
    solid = Manifold.batch_boolean([solids[p] for p in parts], m3.OpType.Add)
    mn, (e1, e2, n) = clearance_map(solid, origin, axis, t0, t1)
    iy, ix = np.unravel_index(mn.argmax(), mn.shape)
    r = mn[iy, ix]; win = (mn.shape[0] - 1) // 2
    centre = np.asarray(origin, float) + e1 * (ix - win) + e2 * (iy - win)
    size = next((s for s in sizes if s / np.sqrt(2) + CLR + wall <= r), None)
    print(f"  tenon {name:14s} clearance r={r:5.1f} -> " + (f"T{size}" if size else "!! 放不下"), flush=True)
    if size is None:
        return None
    return Tenon(name, size, centre + n * t0, n, t1 - t0, e1, parts)


# ------------------------------------------------------------ 底座
def bowtie_cs(grow=0.0):
    pts = [(-25, -13), (-4, -7), (4, -7), (25, -13), (25, 13), (4, 7), (-4, 7), (-25, 13)]
    cs = CrossSection([pts])
    return cs.offset(grow, m3.JoinType.Miter) if grow else cs


def base_bowties(cx, cy):
    return [(cx, cy - BASE_D / 4, 0), (cx, cy + BASE_D / 4, 0),
            (cx - BASE_W / 4, cy, 90), (cx + BASE_W / 4, cy, 90)]


def make_base(cx, cy):
    c = CrossSection.circle(BASE_R, 96)
    hx, hy = BASE_W / 2 - BASE_R, BASE_D / 2 - BASE_R
    def rr(inset):
        cc = CrossSection.circle(BASE_R - inset, 96)
        return CrossSection.batch_hull([cc.translate((sx * hx, sy * hy)) for sx in (-1, 1) for sy in (-1, 1)])
    lower = Manifold.extrude(rr(0), BASE_T - 6).translate((0, 0, -BASE_T))
    upper = Manifold.extrude(rr(6), 1).translate((0, 0, -1))
    base = Manifold.batch_hull([lower, upper]).translate((cx, cy, 0))
    for x, y, ang in base_bowties(cx, cy):
        pk = Manifold.extrude(bowtie_cs(0.2), 10.6).translate((0, 0, -BASE_T - 0.1))
        base -= pk.rotate((0, 0, ang)).translate((x, y, 0))
    return base


# ------------------------------------------------------------ 分件
def segment(whole):
    S = {}                                   # name -> (manifold, color, down)
    top = lambda c: c.bounding_box()[5]
    armL = max(far(whole, (X_ARM_L, 0, 0), (1, 0, 0)).decompose(), key=top)
    armR = max(far(whole, (X_ARM_R, 0, 0), (-1, 0, 0)).decompose(), key=top)
    body = whole - armL - armR

    # 腿：z<Z_THIGH 的两个最大实体是腿，其余 (裆部下缘) 归骨盆
    low = sorted(near(body, (0, 0, Z_THIGH), (0, 0, 1)).decompose(), key=lambda c: -c.volume())
    crotch = low[2:]
    for c in low[:2]:
        b = c.bounding_box(); s = "L" if (b[0] + b[3]) / 2 > X0 else "R"
        z = (0, 0, 1)
        S["foot_" + s] = (slab(c, z, -1, Z_ANKLE), "K", (0, 0, -1))        # 脚掌+脚踝一体 (最细处不拆)
        S["shin_" + s] = (slab(c, z, Z_ANKLE, Z_SHIN), "W", (0, 0, 1))
        S["knee_" + s] = (slab(c, z, Z_SHIN, Z_KNEE), "K", (0, 0, -1))
        S["thigh_" + s] = (slab(c, z, Z_KNEE, Z_THIGH + 1), "W", (0, 0, 1))
    # 骨盆
    mid = slab(body, (0, 0, 1), Z_THIGH, Z_WAIST)
    hipL = biggest(far(mid, (X_HIP_L, 0, 0), (1, 0, 0)))
    hipR = biggest(far(mid, (X_HIP_R, 0, 0), (-1, 0, 0)))
    pel = sorted(slab(mid, (1, 0, 0), X_HIP_R, X_HIP_L).decompose(), key=lambda c: -c.volume())
    for c in pel[1:]:                       # 髋壳伸进中间带的小块，并回髋壳
        b = c.bounding_box()
        if (b[0] + b[3]) / 2 > X0: hipL += c
        else: hipR += c
    S["hip_L"] = (hipL, "W", (0, 0, -1)); S["hip_R"] = (hipR, "W", (0, 0, -1))
    pelvis = Manifold.batch_boolean([pel[0]] + crotch, m3.OpType.Add)
    S["pelvis"] = (near(pelvis, (0, 0, Z_WAISTCOL), (0, 0, 1)), "W", (-1, 0, 0))
    S["waistcol"] = (far(pelvis, (0, 0, Z_WAISTCOL), (0, 0, 1)), "K", (0, 0, -1))
    # 躯干 / 颈 / 头
    torso, logo_info = logo_pad(slab(body, (0, 0, 1), Z_WAIST, Z_NECK0))
    S["torso"] = (torso, "W", (0, 0, -1))
    S["neck"] = (slab(body, (0, 0, 1), Z_NECK0, Z_NECK1), "K", (0, 0, -1))
    SRC["head_vol"] = far(body, (0, 0, Z_NECK1), (0, 0, 1)).volume()     # 原头不用，换成光环头
    S.update(make_halo_head())

    # 左臂：手 <- 腕 <- 前臂 <- 环 <- 肘 | 袖 | 肩
    r = armL
    S["hand_L"] = (far(r, CL + DL * TL["wrist1"], DL), "W", -DL); r = near(r, CL + DL * TL["wrist1"], DL)
    S["wrist_L"] = (far(r, CL + DL * TL["wrist0"], DL), "P", -DL); r = near(r, CL + DL * TL["wrist0"], DL)
    S["forearm_L"] = (far(r, CL + DL * TL["ring1"], DL), "W", -DL); r = near(r, CL + DL * TL["ring1"], DL)
    S["ring_L"] = (far(r, CL + DL * TL["ring0"], DL), "K", -DL); r = near(r, CL + DL * TL["ring0"], DL)
    S["elbow_L"] = (near(r, (0, 0, ZL_ELBOW), (0, 0, 1)), "W", (0, 0, 1))
    S["sleeve_L"] = (slab(r, (0, 0, 1), ZL_ELBOW, ZL_SLEEVE), "P", (0, 0, -1))
    S["shoulder_L"] = (far(r, (0, 0, ZL_SLEEVE), (0, 0, 1)), "W", (0, 0, -1))
    # 右臂
    r = armR
    S["hand_R"] = (far(r, CR + DR * TR["wrist1"], DR), "W", -DR); r = near(r, CR + DR * TR["wrist1"], DR)
    S["wrist_R"] = (far(r, CR + DR * TR["wrist0"], DR), "P", -DR); r = near(r, CR + DR * TR["wrist0"], DR)
    S["forearm_R"] = (far(r, CR + DR * TR["ring1"], DR), "W", -DR); r = near(r, CR + DR * TR["ring1"], DR)
    S["ring_R"] = (far(r, CR + DR * TR["ring0"], DR), "K", -DR); r = near(r, CR + DR * TR["ring0"], DR)
    S["elbow_R"] = (far(r, C2 + D2 * T2["sleeve1"], D2), "W", -D2)
    S["sleeve_R"] = (slab(r, D2, T2["sleeve0"], T2["sleeve1"], C2), "P", -D2)
    S["shoulder_R"] = (near(r, C2 + D2 * T2["sleeve0"], D2), "W", (1, 0, 0))

    out = {}
    for k, (m, col, down) in S.items():
        comps = [c for c in m.decompose() if c.volume() > 150.0]
        if len(comps) != 1:
            print(f"  !! {k}: {len(comps)} 个实体", [round(c.volume() / 1000, 1) for c in comps])
        out[k] = (Manifold.batch_boolean(comps, m3.OpType.Add), col, down)
    return out, logo_info


def logo_pad(torso):
    """在胸口压一块平底浅槽作为不干胶 logo 的定位区。返回 (新躯干, 定位信息)。"""
    L = LOGO
    xs = np.linspace(L["cx"] - L["w"] / 2 + 2, L["cx"] + L["w"] / 2 - 2, 9)
    zs = np.linspace(L["cz"] - L["h"] / 2 + 2, L["cz"] + L["h"] / 2 - 2, 5)
    pts = []
    for x in xs:
        for z in zs:
            col = slab(slab(torso, (1, 0, 0), x - 0.5, x + 0.5), (0, 0, 1), z - 0.5, z + 0.5)
            col = near(col, (0, 80, 0), (0, 1, 0))                 # 只看前表面
            pts.append((x, z, col.bounding_box()[1]))
    pts = np.array(pts)
    A = np.column_stack([np.ones(len(pts)), pts[:, 0], pts[:, 1]])
    coef, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)            # y = a + b x + c z
    resid = pts[:, 2] - A @ coef
    off = resid.max() + L["min_depth"]                              # 槽底平面：最浅处也有 min_depth
    n = np.array([-coef[1], 1.0, -coef[2]]); n /= np.linalg.norm(n)  # 指向体内
    p0 = np.array([L["cx"], coef[0] + coef[1] * L["cx"] + coef[2] * L["cz"] + off, L["cz"]])
    rect = CrossSection.square((L["w"] - 2 * L["r"], L["h"] - 2 * L["r"]), True).offset(L["r"], m3.JoinType.Round)
    Rinv = np.array([[1, 0, 0, L["cx"]], [0, 0, -1, 0], [0, 1, 0, L["cz"]]], float)   # 截面 (x,z) 沿 Y 拉伸
    prism = Manifold.extrude(rect, 300).translate((0, 0, -150)).transform(Rinv)
    cut = near(near(prism, p0, n), (0, 80, 0), (0, 1, 0))
    new = torso - cut
    info = dict(centre_xz=[L["cx"], L["cz"]], size_mm=[L["w"], round(L["h"] / abs(n[1]), 1)],
                depth_mm=[round(float(off - resid.max()), 2), round(float(off - resid.min()), 2)],
                tilt_deg=round(float(np.degrees(np.arctan(coef[2]))), 1),
                removed_cm3=round((torso.volume() - new.volume()) / 1000, 2))
    return new, info


SRC = {}
HEAD_C = np.array([7.5, 108.0, 1240.0])                    # 头部缩放中心
HALO = dict(c=(7.5, 150.0, 1262.0), R=74.0, tube=4.2, tilt=72.0, node=8.0, flat=3.8, bar=5.0)


def _ell(rx, ry, rz, x, y, z):
    return Manifold.sphere(1, 128).scale((rx, ry, rz)).translate((x, y, z))


def _scl(m, k):
    return m.translate(tuple(-HEAD_C)).scale((k, k, k)).translate(tuple(HEAD_C))


def make_halo_head():
    """光环头：亮黑头芯 + 白色甲片 (前顶 / 后顶 / 下后) + 紫色目线 + 脑后紫色光环 + 两根黑色支杆。"""
    cx = float(HEAD_C[0])
    h = Manifold.batch_hull([_ell(60, 76, 60, cx, 112, 1260), _ell(41, 46, 40, cx, 82, 1206),
                             _ell(30, 30, 16, cx, 126, 1184)])
    h = far(h, (0, 0, 1166), (0, 0, 1))
    unit = lambda v: np.asarray(v, float) / np.linalg.norm(v)
    capn, capp = unit((0, 0.45, 1)), np.array([cx, 100, 1262.0])
    gn, gp = unit((0, 1, -0.2)), np.array([cx, 130, 1240.0])
    chn, chp = unit((0, 0.5, -1)), np.array([cx, 96, 1222.0])

    def regions(m, d):                      # d>0: 各边界外扩 d (给座槽留间隙)
        cap = far(m, capp - capn * d, capn)
        cap_f = near(cap, gp - gn * (1.5 - d), gn)
        cap_r = far(cap, gp + gn * (1.5 - d), gn)
        ch = far(far(m, chp - chn * d, chn), (0, 74 - d, 0), (0, 1, 0))
        ch = near(ch, capp - capn * (3.0 - d), capn)
        return cap_f, cap_r, ch

    core0 = _scl(h, 0.955)
    cap_f, cap_r, cheek = regions(h - _scl(h, 0.90), 0.0)
    core = core0
    for seat in regions(_scl(h, 1.05) - _scl(h, 0.897), CLR + 0.05):
        core -= seat
    stub = lambda g, z0, z1: Manifold.cylinder(z1 - z0, 1, 1, 96).scale((42 + g, 31 + g, 1)).translate((7.6, 119, z0))
    core += stub(0, Z_NECK1, Z_NECK1 + 24)                                # 颈座，和头芯一体
    core += stub(0, Z_NECK1 + 24, Z_NECK1 + 45) ^ _scl(h, 0.93)           # 颈座往上延伸进头芯，给榫孔留壁厚
    cheek -= stub(CLR + 0.1, Z_NECK1 - 10, Z_NECK1 + 46)
    # 目线：从正面压进头芯的紫色条
    Rinv = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0]], float)
    rect = CrossSection.square((80, 6), True).translate((cx, 1243.5))
    prism = lambda cs: Manifold.extrude(cs, 400).translate((0, 0, -200)).transform(Rinv)
    eye = near((core0 - _scl(h, 0.86)) ^ prism(rect), (0, 80, 0), (0, 1, 0))
    core -= near((_scl(h, 1.05) - _scl(h, 0.857)) ^ prism(rect.offset(CLR, m3.JoinType.Miter)), (0, 80.3, 0), (0, 1, 0))
    # 光环：背面削平便于贴床打印，水平直径两端加节点接支杆；下端贴着头壳断开
    H = HALO; c = np.array(H["c"])
    ring = Manifold.revolve(CrossSection.circle(H["tube"], 48).translate((H["R"], 0)), 160)
    for sgn in (-1, 1):
        ring += Manifold.sphere(H["node"], 64).translate((sgn * H["R"], 0, 0))
    ring = far(ring, (0, 0, -H["flat"]), (0, 0, 1)).rotate((H["tilt"], 0, 0)).translate(tuple(c)) - _scl(h, 1.012)
    th = np.radians(H["tilt"]); ring_down = (0, float(np.sin(th)), float(-np.cos(th)))
    out = {}
    for sgn, nm in ((1, "L"), (-1, "R")):
        mid = (cx + sgn * 56.0, c[1], c[2])
        a = H["tilt"] - 90.0                                  # 支杆截面跟着光环平面转，节点孔壁才均匀
        bar = Manifold.cube((40.0, H["bar"], H["bar"]), True).rotate((a, 0, 0)).translate(mid)
        cut = Manifold.cube((40.6, H["bar"] + 2 * CLR, H["bar"] + 2 * CLR), True).rotate((a, 0, 0)).translate(mid)
        core -= cut; cap_r -= cut; ring -= cut
        out["halo_strut_" + nm] = (bar, "K", (0, float(np.sin(np.radians(a))), float(-np.cos(np.radians(a)))))
    out.update({"head": (core, "K", (0, 0, -1)), "head_cap_f": (cap_f, "W", (0, 0, -1)),
                "head_cap_r": (cap_r, "W", (0, 0, -1)), "head_cheek": (cheek, "W", (0, 0, -1)),
                "head_eye": (eye, "P", (0, 0, -1)), "halo": (ring, "P", ring_down)})
    return out


# ------------------------------------------------------------ 总装
def build(out, do_render=True):
    t0 = time.time()
    whole, scale = load_source()
    bb = whole.bounding_box()
    print(f"scale {scale:.4f}  tris {whole.num_tri()}  volume {whole.volume() / 1e6:.2f} L  "
          f"bbox {np.round(bb, 1)}", flush=True)
    seg, logo_info = segment(whole)
    solids = {k: v[0] for k, v in seg.items()}
    color = {k: v[1] for k, v in seg.items()}
    down = {k: v[2] for k, v in seg.items()}
    body_names = [k for k in solids if not k.startswith(("head", "halo"))]
    vsum = sum(solids[k].volume() for k in body_names) + logo_info["removed_cm3"] * 1000 + SRC["head_vol"]
    print(f"分件体积合计 / 原模型 = {vsum / whole.volume():.5f}", flush=True)

    # 重心 (按实体体积) 与底座
    def centroid(man):
        mesh = man.to_mesh(); v = np.asarray(mesh.vert_properties)[:, :3].astype(float)
        t = v[np.asarray(mesh.tri_verts)]
        sv = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])) / 6
        return (t.sum(1) / 4 * sv[:, None]).sum(0) / sv.sum(), sv.sum()
    com, _ = centroid(whole)
    bcx, bcy = float(round(com[0])), float(round(com[1]))
    solids["base"] = make_base(bcx, bcy); color["base"] = "W"

    # 背后支撑柱
    back = slab(solids["pelvis"], (0, 0, 1), POST_TIE_Z - 15, POST_TIE_Z + 15)
    back = slab(back, (1, 0, 0), X0 - 15, X0 + 15)
    y_back = back.bounding_box()[4]
    py0 = float(np.ceil(y_back)) + 1.0
    for i in range(3):
        seg_ = Manifold.cube((POST, POST, POST_Z[i + 1] - POST_Z[i])).translate(
            (X0 - POST / 2, py0, POST_Z[i]))
        solids[f"post_{i + 1}"] = seg_; color[f"post_{i + 1}"] = "K"; down[f"post_{i + 1}"] = (0, 1, 0)
    pc = np.array([X0, py0 + POST / 2, 0.0])

    # 榫条
    Z, Xp = (0, 0, 1), (1, 0, 0)
    T = []
    def add(*a, **k):
        t = auto_tenon(*a, **k)
        if t: T.append(t)
        return t
    for s, x in (("L", 112.0), ("R", -98.0)):
        add("footbase_" + s, solids, ["base", "foot_" + s], (x, 60, 0), Z, -15, 12)
        add("ankle_" + s, solids, ["foot_" + s, "shin_" + s], (x, 136, 0), Z, Z_ANKLE - 28, Z_ANKLE + 40)
        add("knee_" + s, solids, ["shin_" + s, "knee_" + s, "thigh_" + s], (x, 115, 0), Z, Z_SHIN - 40, Z_KNEE + 40, wall=3.5)
        add("hip_" + s, solids, ["thigh_" + s, "hip_" + s], (x + 10, 108, 0), Z, Z_THIGH - 40, Z_THIGH + 40)
    add("hipx_L", solids, ["pelvis", "hip_L"], (X_HIP_L, 120, 690), Xp, -30, 40)
    add("hipx_R", solids, ["pelvis", "hip_R"], (X_HIP_R, 120, 690), (-1, 0, 0), -30, 40)
    add("waist", solids, ["pelvis", "waistcol", "torso"], (X0, 128, 0), Z, Z_WAIST - 70, Z_WAIST + 70)
    add("neck", solids, ["torso", "neck", "head"], (X0, 118, 0), Z, Z_NECK0 - 40, Z_NECK1 + 33)
    add("shoulder_L", solids, ["torso", "shoulder_L"], (X_ARM_L, 114, 1068), Xp, -35, 35)
    add("shoulder_R", solids, ["torso", "shoulder_R"], (X_ARM_R, 116, 1070), (-1, 0, 0), -35, 35)
    add("sleeve_L", solids, ["elbow_L", "sleeve_L", "shoulder_L"], (176, 131, 0), Z, ZL_ELBOW - 35, ZL_SLEEVE + 35)
    add("sleeve_R", solids, ["shoulder_R", "sleeve_R", "elbow_R"], C2, D2, T2["sleeve0"] - 35, T2["sleeve1"] + 35)
    for s, C, D, TT in (("L", CL, DL, TL), ("R", CR, DR, TR)):
        add("ring_" + s, solids, ["elbow_" + s, "ring_" + s, "forearm_" + s], C, D, TT["ring0"] - 35, TT["ring1"] + 35)
        add("wrist_" + s, solids, ["forearm_" + s, "wrist_" + s, "hand_" + s], C, D, TT["wrist0"] - 35, TT["wrist1"] + 30)
    # 支撑柱：固定位置
    T.append(Tenon("post_base", 16, pc + (0, 0, -15), Z, 45, Xp, ["base", "post_1"]))
    T.append(Tenon("post_12", 16, pc + (0, 0, POST_Z[1] - 30), Z, 60, Xp, ["post_1", "post_2"]))
    T.append(Tenon("post_23", 16, pc + (0, 0, POST_Z[2] - 30), Z, 60, Xp, ["post_2", "post_3"]))
    T.append(Tenon("post_tie", 16, (X0, y_back - 45, POST_TIE_Z), (0, 1, 0), py0 + POST - (y_back - 45),
                   Z, ["pelvis", "post_3"]))

    # 壁厚复核：加厚的榫孔必须完全落在它穿过的零件里
    raw = dict(solids)
    for t in T:
        u = Manifold.batch_boolean([raw[p] for p in t.parts], m3.OpType.Add)
        c = t.cutter(grow=CLR + 2.5, extra=-1.0)
        miss = (c - u).volume() / c.volume()
        if miss > 0.01 and t.name != "post_tie":
            print(f"  !! {t.name}: 加厚榫孔有 {miss * 100:.1f}% 在实体外")
    for i, a in enumerate(T):
        for b in T[i + 1:]:
            if (a.cutter(grow=2.0) ^ b.cutter(grow=2.0)).volume() > 1:
                print(f"  !! 榫孔相碰: {a.name} / {b.name}")
    for t in T:
        cut = t.cutter()
        for p in t.parts:
            solids[p] -= cut

    # ---------------------------------------------------------- 导出
    world, rows = {}, []
    def export(name, man, col, dn, qty=1, note=""):
        pm = orient(man, dn)
        fn = os.path.join(out, "stl", COLOR_DIR[col], name + ".stl")
        ntri = write_stl(fn, pm)
        b = pm.bounding_box(); dims = (b[3] - b[0], b[4] - b[1], b[5] - b[2])
        fit = next((n for n, bed in BEDS if all(dims[i] <= bed[i] - 6 for i in range(3))), "TOO BIG")
        nb = len([c for c in pm.decompose() if c.volume() > 1])
        if nb != 1: print(f"  !! {name}: 导出后 {nb} 个实体")
        if fit == "TOO BIG": print(f"  !! {name}: 放不进任何机器 {np.round(dims)}")
        rows.append(dict(file=os.path.relpath(fn, out), name=name, color=col, qty=qty,
                         x=round(dims[0], 1), y=round(dims[1], 1), z=round(dims[2], 1),
                         volume_cm3=round(pm.volume() / 1000, 1), area_cm2=round(pm.surface_area() / 100, 1),
                         tris=ntri, printer=fit, note=note))
    NOTE = {"foot": "脚掌+脚踝一体，脚踝最细处很细：100% 填充", "knee": "膝关节", "pelvis": "骨盆，侧躺打印，4 层壁", "waistcol": "腰柱，承上身重量：6 层壁 + 40% 填充",
            "neck": "颈环", "sleeve": "上臂紫色段",
            "head_cap": "头顶白色甲片，扣进头芯的座槽并点胶，需支撑", "head_cheek": "下后白色甲片，从下往上套进头芯座槽并点胶",
            "head_eye": "紫色目线，从正面压进头芯的槽并点胶", "halo_strut": "光环支杆，平躺打印，100% 填充",
            "halo": "脑后紫色光环，削平的背面贴床，加 brim；两端节点套在支杆上并点胶",
            "ring": "肘下电机环", "wrist": "紫色腕环", "head": "亮黑头芯 (含颈座)，颈座朝下，需支撑", "torso": "躯干，最大件；胸口平底浅槽是 logo 贴纸位"}
    for k in seg:
        world[k] = (solids[k], color[k])
        export(k, solids[k], color[k], down[k], note=next((v for p, v in NOTE.items() if k.startswith(p)), ""))
    for i in range(3):
        k = f"post_{i + 1}"; world[k] = (solids[k], "K")
        export(k, solids[k], "K", down[k], note="背后支撑柱，平躺打印，4 层壁 + 40% 填充")
    base = solids["base"]
    for nm, sx, sy in (("base_front_L", 1, -1), ("base_front_R", -1, -1),
                       ("base_rear_L", 1, 1), ("base_rear_R", -1, 1)):
        tile = far(far(base, (bcx, 0, 0), (sx, 0, 0)), (0, bcy, 0), (0, sy, 0))
        world[nm] = (tile, "W"); export(nm, tile, "W", (0, 0, -1), note="底座 1/4，顶面朝上")
    counts = {}
    for t in T:
        key = (t.size, int(round(t.length - 1)))
        counts.setdefault(key, []).append(t.name)
        world["tenon_" + t.name] = (t.body(), "H")
    for (size, length), names in sorted(counts.items()):
        export(f"tenon_T{size}x{length}", make_tenon(size, length), "H", (0, 0, -1), qty=len(names),
               note="榫条，平躺打印，4 层壁 + 40% 填充；用于 " + " ".join(names))
    export("bowtie_key", Manifold.extrude(bowtie_cs(), 9.8), "H", (0, 0, -1), qty=4, note="底座蝴蝶键，从底面压入")
    used = sorted({t.size for t in T}, reverse=True)
    cp = Manifold.cube((20 + sum(s + 14 for s in used), 44, 20), True).translate((0, 0, 10))
    x = -cp.bounding_box()[3] + 14
    for s in used:
        cp -= Manifold.cube((s + 2 * CLR, s + 2 * CLR, 50), True).translate((x + s / 2, 0, 0)); x += s + 14
    export("fit_test_coupon", cp, "H", (0, 0, -1), note="先打这个，试各规格榫条的松紧")

    # ---------------------------------------------------------- 校验与报告
    names = list(world); bbs = {n: world[n][0].bounding_box() for n in names}
    bad = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A_, B_ = bbs[a], bbs[b]
            if any(A_[k] > B_[k + 3] or B_[k] > A_[k + 3] for k in range(3)):
                continue
            v = (world[a][0] ^ world[b][0]).volume()
            if v > 2.0: bad.append((a, b, round(v, 1)))
    print("装配干涉 (mm3):", bad if bad else "无", flush=True)

    def grams(vol_cm3, area_cm2):
        shell = min(area_cm2 * 0.10, vol_cm3)                 # 约 1 mm 表皮
        return (shell + (vol_cm3 - shell) * 0.12) * 1.24
    est = {}
    for r in rows:
        est[r["color"]] = est.get(r["color"], 0) + grams(r["volume_cm3"], r["area_cm2"]) * r["qty"]
    msum, mc = 0.0, np.zeros(3)
    for n, (man, col) in world.items():
        if n.startswith("base") or n.startswith("post"):
            continue
        c, vol = centroid(man)
        g = grams(man.volume() / 1000, man.surface_area() / 100); msum += g; mc += g * c
    comw = mc / msum
    margin = min(BASE_W / 2 - abs(comw[0] - bcx), BASE_D / 2 - abs(comw[1] - bcy))
    tip = float(np.degrees(np.arctan2(margin, comw[2] + BASE_T)))
    ztop_head = max(solids[k].bounding_box()[5] for k in ("head_cap_f", "head_cap_r"))
    summary = dict(source="MakerWorld 2512182 G1Static.stl", palette=PALETTE, logo_pad=logo_info, scale=round(scale, 4),
                   head_top_mm=round(ztop_head, 1), halo_top_mm=round(solids["halo"].bounding_box()[5], 1), raised_hand_top_mm=round(bb[5], 1),
                   width_mm=round(bb[3] - bb[0], 1), depth_mm=round(bb[4] - bb[1], 1),
                   base_mm=[BASE_W, BASE_D, BASE_T], base_centre=[bcx, bcy],
                   est_grams={k: round(v) for k, v in est.items()}, est_total_g=round(sum(est.values())),
                   centre_of_mass_mm=[round(float(c), 1) for c in comw], tip_angle_deg=round(tip, 1),
                   interference=bad, files=len(rows), pieces=sum(r["qty"] for r in rows),
                   tenons={t.name: f"T{t.size}x{int(round(t.length - 1))}" for t in T})
    os.makedirs(os.path.join(out, "docs"), exist_ok=True)
    with open(os.path.join(out, "docs", "parts.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(out, "docs", "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))
    print(f"build {time.time() - t0:.0f}s")
    if do_render:
        sys.path.insert(0, os.path.dirname(__file__))
        import render
        render.render_all(world, os.path.join(out, "docs"))
    return world, rows, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=ROOT)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    build(a.out, not a.no_render)
