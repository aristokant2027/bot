"""Build all printable STL files + reports for the HUMAN+ static figure.

usage: python3 build.py [--out ..] [--only name,name]
"""
import argparse, csv, json, os, struct, sys, time
import numpy as np
from skimage import measure
import manifold3d as m3
from manifold3d import Manifold, Mesh, CrossSection

sys.path.insert(0, os.path.dirname(__file__))
import figure as fg

SIMPLIFY_TOL = 0.04
COLOR_DIR = {"W": "1_white", "K": "2_black", "P": "3_purple", "H": "4_connectors"}
BEDS = [("A1 / X2D", (256, 256, 256)), ("A2L", (330, 320, 325))]


# ------------------------------------------------------------ mesh helpers
def sdf_to_manifold(fn, lo, hi, voxel, level=0.0):
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    n = np.ceil((hi - lo) / voxel).astype(int) + 1
    ax = [(lo[i] + voxel * np.arange(n[i])).astype(np.float32) for i in range(3)]
    P = (ax[0][:, None, None], ax[1][None, :, None], ax[2][None, None, :])
    d = np.ascontiguousarray(np.broadcast_to(fn(P), tuple(n))).astype(np.float32)
    d = d - level
    for s in (np.s_[0], np.s_[-1]):          # close the surface at the grid walls
        d[s, :, :] = np.maximum(d[s, :, :], 1); d[:, s, :] = np.maximum(d[:, s, :], 1)
        d[:, :, s] = np.maximum(d[:, :, s], 1)
    v, f, _, _ = measure.marching_cubes(d, 0.0, spacing=(voxel,) * 3)
    v = (v + lo).astype(np.float32)
    man = Manifold(Mesh(vert_properties=v, tri_verts=f.astype(np.uint32)))
    if man.status() != m3.Error.NoError or man.volume() < 0:
        man = Manifold(Mesh(vert_properties=v,
                            tri_verts=f[:, ::-1].astype(np.uint32).copy()))
    assert man.status() == m3.Error.NoError, man.status()
    assert man.volume() > 0
    return man.simplify(SIMPLIFY_TOL)


def keep(man, point, direction):
    n = np.asarray(direction, float); n /= np.linalg.norm(n)
    return man.trim_by_plane(tuple(n), float(np.dot(n, point)))


def frame(e1, axis):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    e1 = np.asarray(e1, float); e1 = e1 - axis * (e1 @ axis); e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return e1, e2, axis


def place(man, e1, axis, origin):
    e1, e2, ax = frame(e1, axis)
    M = np.column_stack([e1, e2, ax, np.asarray(origin, float)])
    return man.transform(M)


def perp(axis):
    axis = np.asarray(axis, float)
    c = np.eye(3)[np.argmin(np.abs(axis))]
    return c


def box_cutter(t, clr, extra=1.0):
    s = t.size + 2 * clr
    c = Manifold.cube((s, s, t.length + 2 * extra), True)
    axis = np.asarray(t.axis, float)
    return place(c, t.e1, axis, np.asarray(t.p0) + axis * t.length / 2)


def cyl_cutter(h):
    c = Manifold.cylinder(h.length, h.d / 2, h.d / 2, 48, True)
    return place(c, perp(h.axis), h.axis, h.p0)


def mirror(man):
    return man.mirror((1, 0, 0))


def orient(man, down):
    """Rotate so `down` faces the bed, drop on z=0, centre in XY.
    Returns (manifold, 3x4 matrix)."""
    e3 = -np.asarray(down, float); e3 /= np.linalg.norm(e3)
    e1 = perp(e3); e1 = e1 - e3 * (e1 @ e3); e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    R = np.vstack([e1, e2, e3])
    M = np.column_stack([R, np.zeros(3)])
    r = man.transform(M)
    b = r.bounding_box()
    t = np.array([-(b[0] + b[3]) / 2, -(b[1] + b[4]) / 2, -b[2]])
    M[:, 3] = t
    return man.transform(M), M


def write_stl(path, man):
    mesh = man.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3]; f = np.asarray(mesh.tri_verts)
    tri = v[f]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(nrm, axis=1, keepdims=True); nrm = nrm / np.maximum(ln, 1e-12)
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = nrm; rec["v"] = tri
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"HUMAN+ static figure".ljust(80, b" "))
        fh.write(struct.pack("<I", len(f))); fh.write(rec.tobytes())
    return len(f)


# ------------------------------------------------------- connector parts
def make_tenon(size, length, holes=True):
    e, c = 1.2, 2.0
    a = Manifold.cube((size, size - 2 * e, length - 2 * c), True)
    b = Manifold.cube((size - 2 * e, size, length - 2 * c), True)
    d = Manifold.cube((size - 2 * c, size - 2 * c, length), True)
    t = Manifold.batch_hull([a, b, d])
    if holes:
        for z in (-length / 2 + 15, length / 2 - 15):
            h = Manifold.cylinder(size + 2, fg.PIN_HOLE_TENON / 2,
                                  fg.PIN_HOLE_TENON / 2, 48, True)
            t -= h.rotate((90, 0, 0)).translate((0, 0, z))
    return t.rotate((90, 0, 0))            # lie flat on a side face


def make_pin(length, d=fg.PIN_D):
    a = Manifold.cylinder(length - 1.6, d / 2, d / 2, 48, True)
    b = Manifold.cylinder(length, d / 2 - 0.8, d / 2 - 0.8, 48, True)
    p = Manifold.batch_hull([a, b]).rotate((0, 90, 0))
    return p ^ Manifold.cube((length + 1, d + 1, d - 0.8), True)


def stadium(r, half, h):
    c = CrossSection.circle(r, 64)
    cs = CrossSection.batch_hull([c.translate((0, -half)), c.translate((0, half))])
    return Manifold.extrude(cs, h)


def bowtie_cs(grow=0.0):
    pts = [(-25, -13), (-4, -7), (4, -7), (25, -13), (25, 13), (4, 7), (-4, 7), (-25, 13)]
    cs = CrossSection([pts])
    return cs.offset(grow, m3.JoinType.Miter) if grow else cs


def make_base():
    c = CrossSection.circle(fg.BASE_R, 96)
    hx, hy = fg.BASE_W / 2 - fg.BASE_R, fg.BASE_D / 2 - fg.BASE_R
    def rr(inset):
        cc = CrossSection.circle(fg.BASE_R - inset, 96)
        return CrossSection.batch_hull([cc.translate((sx * hx, sy * hy))
                                        for sx in (-1, 1) for sy in (-1, 1)])
    lower = Manifold.extrude(rr(0), fg.BASE_T - 9).translate((0, 0, -fg.BASE_T))
    upper = Manifold.extrude(rr(9), 1).translate((0, 0, -1))
    base = Manifold.batch_hull([lower, upper]).translate((0, fg.BASE_CY, 0))
    for x, y, d in fg.BOWTIES:
        pk = Manifold.extrude(bowtie_cs(0.2), 10.6).translate((0, 0, -fg.BASE_T - 0.1))
        if d == "y":
            pk = pk.rotate((0, 0, 90))
        base -= pk.translate((x, y, 0))
    return base


def make_logo():
    """Returns (text_prism, plus_prism) pointing to -Y, centred on chest."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    fp = FontProperties(family="DejaVu Sans", weight="bold", style="oblique")
    tp = TextPath((0, 0), "HUMAN", size=14, prop=fp)
    polys = [np.asarray(p[:-1]) for p in tp.to_polygons() if len(p) > 3]
    cs = CrossSection(polys, m3.FillRule.EvenOdd)
    x0, y0, x1, y1 = cs.bounds()
    s = 4.2
    sq = CrossSection.square((s, s), True)
    cx, cy = x1 + 3.0 + 1.5 * s, (y0 + y1) / 2
    plus = CrossSection.batch_boolean(
        [sq.translate((cx + dx * s, cy + dy * s))
         for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))], m3.OpType.Add)
    total = (cx + 1.5 * s) - x0
    out = []
    for c in (cs, plus):
        p = Manifold.extrude(c.translate((-x0 - total / 2, -cy)), 110)
        out.append(p.rotate((90, 0, 0)).translate((0, 0, 1046)))
    return out


# ------------------------------------------------------------------ build
def build(out, only=None):
    t0 = time.time()
    world = {}      # name -> (manifold in world frame, color)
    solids = {}
    parts = {p.name: p for p in fg.PARTS}
    for p in fg.PARTS:
        if only and p.name not in only:
            continue
        ts = time.time()
        man = sdf_to_manifold(p.fn, p.lo, p.hi, p.voxel)
        for pt, d in p.clips:
            man = keep(man, pt, d)
        solids[p.name] = man
        print(f"  sdf {p.name:12s} tris={man.num_tri():7d} {time.time()-ts:5.1f}s",
              flush=True)
    solids["base"] = make_base()

    # joints
    for t in fg.TENONS:
        cut = box_cutter(t, fg.CLR)
        for name in t.parts:
            if name not in solids:
                continue
            centre = name == "base" or not parts[name].side
            solids[name] -= cut
            if centre and t.sym:
                solids[name] -= mirror(cut)
    for h in fg.HOLES:
        cut = cyl_cutter(h)
        for name in h.parts:
            if name not in solids:
                continue
            solids[name] -= cut
            if not parts[name].side and h.sym:
                solids[name] -= mirror(cut)

    # shoulder accent pad + pocket
    pad = stadium(fg.PAD_R, fg.PAD_Y, fg.PAD_TOP - fg.PAD_Z0 - 0.2).translate(
        (fg.PAD_X, 0, fg.PAD_Z0 + 0.2))
    if "shoulder" in solids:
        pocket = stadium(fg.PAD_R + 0.2, fg.PAD_Y, 20).translate((fg.PAD_X, 0, fg.PAD_Z0))
        solids["shoulder"] -= pocket

    # chest logo (debossed 1 mm) + optional colour-fill bodies
    logo = {}
    if "chest" in solids:
        cp = parts["chest"]
        inner = sdf_to_manifold(cp.fn, cp.lo, cp.hi, cp.voxel, level=-1.0)
        for nm, prism in zip(("chest_logo_text", "chest_logo_plus"), make_logo()):
            cutter = prism - inner              # 1 mm deep, follows the chest surface
            logo[nm] = cutter ^ solids["chest"]
            solids["chest"] -= cutter
        comps = solids["chest"].decompose()
        solids["chest"] = max(comps, key=lambda c: c.volume())
        print("  chest genus", solids["chest"].genus(), "bodies", len(comps))

    # ---- export
    rows = []
    def export(name, man, color, down, qty=1, note="", world_man=None, M=None):
        if M is None:
            pm, M = orient(man, down)
        else:
            pm = man.transform(M)
        fn = os.path.join(out, "stl", COLOR_DIR[color], name + ".stl")
        ntri = write_stl(fn, pm)
        b = pm.bounding_box()
        dims = (b[3] - b[0], b[4] - b[1], b[5] - b[2])
        fit = next((n for n, bed in BEDS if dims[0] <= bed[0] - 6 and dims[1] <= bed[1] - 6
                    and dims[2] <= bed[2] - 6), "TOO BIG")
        vol, area = pm.volume() / 1000, pm.surface_area() / 100   # cm3, cm2
        rows.append(dict(file=os.path.relpath(fn, out), name=name, color=color, qty=qty,
                         x=round(dims[0], 1), y=round(dims[1], 1), z=round(dims[2], 1),
                         volume_cm3=round(vol, 1), area_cm2=round(area, 1), tris=ntri,
                         printer=fit, genus=pm.genus(), note=note))
        comps = pm.decompose()
        if len(comps) > 1 and not name.startswith("chest_logo"):
            print("  !! multi-body", name, [round(c.volume(), 2) for c in comps])
        return M

    Mchest = None
    for p in fg.PARTS:
        if p.name not in solids:
            continue
        man = solids[p.name]
        if p.side:
            world[p.name + "_L"] = (man, p.color)
            world[p.name + "_R"] = (mirror(man), p.color)
            export(p.name + "_L", man, p.color, p.down, note=p.note)
            dn = np.array(p.down, float) * (-1, 1, 1)
            export(p.name + "_R", mirror(man), p.color, dn, note=p.note)
        else:
            world[p.name] = (man, p.color)
            M = export(p.name, man, p.color, p.down, note=p.note)
            if p.name == "chest":
                Mchest = M
    for nm, man in logo.items():
        col = "K" if nm.endswith("text") else "P"
        world[nm] = (man, col)
        export(nm, man, col, None, M=Mchest,
               note="可选: 与 chest.stl 同坐标, 在切片软件中作为同一对象的多色零件载入")
    world["shoulder_pad_L"] = (pad, "P"); world["shoulder_pad_R"] = (mirror(pad), "P")
    export("shoulder_pad", pad, "P", (0, 0, -1), qty=2, note="肩部紫色饰块, 压入/胶粘")

    # base tiles
    base = solids["base"]
    for nm, sx, sy in (("base_front_L", 1, -1), ("base_front_R", -1, -1),
                       ("base_rear_L", 1, 1), ("base_rear_R", -1, 1)):
        tile = keep(keep(base, (0, 0, 0), (sx, 0, 0)), (0, fg.BASE_CY, 0), (0, sy, 0))
        world[nm] = (tile, "W")
        export(nm, tile, "W", (0, 0, -1), note="底座 1/4, 顶面朝上")

    # connectors
    counts = {}
    for t in fg.TENONS:
        key = (t.size, t.length)
        counts[key] = counts.get(key, 0) + (2 if t.sym else 1)
        ten = make_tenon(t.size, t.length).rotate((-90, 0, 0))
        ten = place(ten, t.e1, t.axis, np.asarray(t.p0) + np.asarray(t.axis) * t.length / 2)
        world["tenon_" + t.name + ("_L" if t.sym else "")] = (ten, "H")
        if t.sym:
            world["tenon_" + t.name + "_R"] = (mirror(ten), "H")
    for (size, length), q in sorted(counts.items()):
        export(f"tenon_T{int(size)}x{int(length)}", make_tenon(size, length), "H",
               (0, 0, -1), qty=q, note="榫条, 平躺打印, 4 层壁 + 40% 填充")
    export("pin_P6x40", make_pin(40), "H", (0, 0, -1), qty=8, note="手臂横销")
    export("pin_P6x80", make_pin(80), "H", (0, 0, -1), qty=2, note="肩部横销")
    export("dowel_P6x20", make_pin(20), "H", (0, 0, -1), qty=7, note="定位销(膝盖4, 头部3)")
    export("bowtie_key", Manifold.extrude(bowtie_cs(), 9.8), "H", (0, 0, -1),
           qty=len(fg.BOWTIES), note="底座蝴蝶键, 从底面压入")
    # fit test coupon
    cp = Manifold.cube((74, 40, 22), True).translate((0, 0, 11))
    cp -= Manifold.cube((24 + 2 * fg.CLR,) * 2 + (50,), True).translate((-20, 0, 0))
    cp -= Manifold.cube((16 + 2 * fg.CLR,) * 2 + (50,), True).translate((10, 0, 0))
    cp -= Manifold.cylinder(50, fg.PIN_HOLE / 2, fg.PIN_HOLE / 2, 48, True).translate((28, 9, 0))
    cp -= Manifold.cylinder(50, fg.DOWEL_HOLE / 2, fg.DOWEL_HOLE / 2, 48, True).translate((28, -9, 0))
    export("fit_test_coupon", cp, "H", (0, 0, -1), note="先打这个, 试 T24/T16 榫条与 P6 销的松紧")

    # ---- reports
    os.makedirs(os.path.join(out, "docs"), exist_ok=True)
    with open(os.path.join(out, "docs", "parts.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- validation: interference between assembled parts
    names = list(world)
    bbs = {n: world[n][0].bounding_box() for n in names}
    bad = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a.startswith("chest_logo") or b.startswith("chest_logo"):
                continue
            A_, B_ = bbs[a], bbs[b]
            if any(A_[k] > B_[k + 3] or B_[k] > A_[k + 3] for k in range(3)):
                continue
            v = (world[a][0] ^ world[b][0]).volume()
            if v > 2.0:
                bad.append((a, b, round(v, 1)))
    print("interference (mm3):", bad if bad else "none")
    # ---- rough filament estimate + centre of mass (uniform shell/infill assumption)
    def grams(vol_cm3, area_cm2):
        shell = min(area_cm2 * 0.10, vol_cm3)          # ~1 mm skin
        return (shell + (vol_cm3 - shell) * 0.12) * 1.24
    est = {}
    for r in rows:
        if r["name"].startswith("chest_logo"):
            continue
        est[r["color"]] = est.get(r["color"], 0) + grams(r["volume_cm3"], r["area_cm2"]) * r["qty"]
    msum, mc = 0.0, np.zeros(3)
    for n, (man, col) in world.items():
        if n.startswith("chest_logo"):
            continue
        mesh = man.to_mesh(); v = np.asarray(mesh.vert_properties)[:, :3].astype(float)
        t = v[np.asarray(mesh.tri_verts)]
        sv = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])) / 6
        cen = (t.sum(1) / 4 * sv[:, None]).sum(0) / sv.sum()
        g = grams(man.volume() / 1000, man.surface_area() / 100)
        msum += g; mc += g * cen
    com = mc / msum
    summary = dict(height_mm=round(max(bb[5] for bb in bbs.values()), 1),
                   base_mm=[fg.BASE_W, fg.BASE_D, fg.BASE_T],
                   est_grams={k: round(v) for k, v in est.items()},
                   est_total_g=round(sum(est.values())),
                   centre_of_mass_mm=[round(float(c), 1) for c in com],
                   interference=bad, files=len(rows),
                   pieces=sum(r["qty"] for r in rows if not r["name"].startswith("chest_logo")))
    with open(os.path.join(out, "docs", "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    print(summary)
    ztop = max(bb[5] for bb in bbs.values())
    print(f"figure height above base: {ztop:.1f} mm; build {time.time()-t0:.0f}s")
    return world, rows, bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), ".."))
    ap.add_argument("--only", default="")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "docs"), exist_ok=True)
    world, rows, bad = build(a.out, set(a.only.split(",")) if a.only else None)
    if not a.no_render:
        import render
        render.render_all(world, os.path.join(a.out, "docs"))
