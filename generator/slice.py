#!/usr/bin/env python3
"""用 Bambu Studio 命令行把 stl/ 按机器和参数分组切片，输出带 G-code 的 .3mf。

用法:  python3 generator/slice.py --out <输出目录> [--only 组名前缀]
命令行不会展开预设的继承关系，所以这里先把官方预设 (机器 / 工艺 / PETG 耗材) 逐级展开成完整 json 再喂给它。
"""
import argparse, csv, glob, json, os, re, subprocess, sys, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = "/Applications/BambuStudio.app/Contents"
PROF = APP + "/Resources/profiles/BBL"
MACHINES = {   # 机器 -> (机器预设, 工艺预设, PETG 耗材预设)
    "X2D": ("Bambu Lab X2D 0.4 nozzle", "0.20mm Standard @BBL X2D", "Bambu PETG HF @BBL X2D 0.4 nozzle"),
    "A2L": ("Bambu Lab A2L 0.4 nozzle", "0.20mm Standard @BBL A2L", "Bambu PETG HF @BBL A2L"),
    "A1": ("Bambu Lab A1 0.4 nozzle", "0.20mm Standard @BBL A1", "Bambu PETG HF @BBL A1"),
}
SUP = {"enable_support": "1", "support_type": "tree(auto)", "support_top_z_distance": "0.22", "support_bottom_z_distance": "0.22"}
NOSUP = {"enable_support": "0"}
PROC = {       # 参数组 -> 对工艺预设的改动
    "shell": {"wall_loops": "3", "sparse_infill_density": "15%", **SUP},
    "flat": {"wall_loops": "3", "sparse_infill_density": "15%", **NOSUP},
    "conn": {"wall_loops": "4", "sparse_infill_density": "40%", **NOSUP},
    "strong": {"wall_loops": "8", "sparse_infill_density": "40%", **SUP},   # 脚踝最细处基本全是壁；100% 填充要多用约 0.8 kg
    "waist": {"wall_loops": "6", "sparse_infill_density": "40%", **SUP},
    "pelvis": {"wall_loops": "4", "sparse_infill_density": "15%", **SUP},
    "halo": {"wall_loops": "3", "sparse_infill_density": "15%", **NOSUP, "brim_type": "outer_only", "brim_width": "5"},
}
JOBS = [       # (文件名, 机器, 参数组, 颜色, [零件名或通配])
    ("x2d_01_先打_试配件", "X2D", "conn", "任意", ["fit_test_coupon", "tenon_T16x44", "tenon_T24x26:1"]),
    ("x2d_02_连接件和支撑柱", "X2D", "conn", "黑", ["tenon_*", "bowtie_key", "post_*"]),
    ("x2d_03_黑_脚和支杆_加强", "X2D", "strong", "黑", ["foot_*", "halo_strut_*"]),
    ("x2d_04_黑_腰柱", "X2D", "waist", "黑", ["waistcol"]),
    ("x2d_05_黑_头芯和关节", "X2D", "shell", "黑", ["head", "knee_*", "neck", "ring_*"]),
    ("x2d_06_紫_上臂腕环目线", "X2D", "shell", "紫", ["sleeve_*", "wrist_*", "head_eye"]),
    ("x2d_07_紫_光环", "X2D", "halo", "紫", ["halo"]),
    ("a2l_01_白_躯干", "A2L", "shell", "白", ["torso"]),
    ("a2l_02_白_骨盆", "A2L", "pelvis", "白", ["pelvis"]),
    ("a1_01_白_底座", "A1", "flat", "白", ["base_*"]),
    ("a1_02_白_腿", "A1", "shell", "白", ["shin_*", "thigh_*"]),
    ("a1_03_白_上身和头甲", "A1", "shell", "白", ["hip_*", "shoulder_*", "elbow_*", "hand_*", "forearm_*", "head_cap_*", "head_cheek"]),
]


def load_index():
    idx = {}
    for sub in ("machine", "process", "filament"):
        for f in glob.glob(f"{PROF}/{sub}/**/*.json", recursive=True):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            if d.get("name"):
                idx[(sub, d["name"])] = d
    return idx


def flatten(idx, sub, name):
    d = idx[(sub, name)]; out = {}
    if d.get("inherits"):
        out.update(flatten(idx, sub, d["inherits"]))
    for inc in d.get("include", []):
        out.update(flatten(idx, sub, inc))
    out.update(d)
    return out


def write_preset(idx, sub, name, path, over=None):
    d = flatten(idx, sub, name)
    for k in ("inherits", "include"):
        d.pop(k, None)
    d.update(over or {}); d["from"] = "system"
    json.dump(d, open(path, "w"), indent=1, ensure_ascii=False)
    return path


def expand(parts, rows):
    """零件名 / 通配 -> [(stl 路径, 数量)]；'名字:n' 强制数量。"""
    out = []
    for spec in parts:
        name, _, q = spec.partition(":")
        rx = re.compile("^" + re.escape(name).replace(r"\*", ".*") + "$")
        hit = [r for r in rows if rx.match(r["name"])]
        if not hit:
            raise SystemExit(f"找不到零件 {spec}")
        out += [(os.path.join(ROOT, r["file"]), int(q) if q else int(r["qty"])) for r in hit]
    return out


def gcode_stats(path):
    """从 3mf 里的各盘 G-code 头读出时长和耗材克数。"""
    plates = []
    with zipfile.ZipFile(path) as z:
        for n in sorted(z.namelist()):
            if n.endswith(".gcode"):
                head = z.open(n).read(4000).decode("utf8", "ignore")
                t = re.search(r"total estimated time: ([^\n;]+)", head)
                w = re.search(r"total filament weight \[g\] : ([\d.,]+)", head)
                sec = 0
                if t:
                    for v, u in re.findall(r"(\d+)([dhms])", t.group(1)):
                        sec += int(v) * {"d": 86400, "h": 3600, "m": 60, "s": 1}[u]
                plates.append((sec, sum(float(x) for x in w.group(1).split(",")) if w else 0.0))
    return plates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = os.path.join(a.out, "_presets"); os.makedirs(cfg, exist_ok=True)
    idx = load_index()
    rows = list(csv.DictReader(open(os.path.join(ROOT, "docs", "parts.csv"))))
    report = []
    for name, mach, proc, colour, parts in JOBS:
        if not name.startswith(a.only):
            continue
        m, p, f = MACHINES[mach]
        mj = write_preset(idx, "machine", m, f"{cfg}/{mach}_machine.json")
        pj = write_preset(idx, "process", p, f"{cfg}/{mach}_{proc}.json", {**PROC[proc], "curr_bed_type": "Textured PEI Plate"})
        fj = write_preset(idx, "filament", f, f"{cfg}/{mach}_petg.json")
        items = expand(parts, rows)
        out3 = os.path.join(a.out, name + ".gcode.3mf")
        if os.path.exists(out3):
            os.remove(out3)
        cmd = [APP + "/MacOS/BambuStudio", "--debug", "2", "--slice", "0", "--arrange", "1",
               "--load-settings", f"{mj};{pj}", "--load-filaments", fj,
               "--clone-objects", ",".join(str(q) for _, q in items),
               "--outputdir", a.out, "--export-3mf", os.path.basename(out3)] + [s for s, _ in items]
        r = subprocess.run(cmd, capture_output=True, text=True)
        log = (r.stdout + r.stderr).strip().splitlines()
        ok = r.returncode == 0 and os.path.exists(out3)
        st = gcode_stats(out3) if ok else []
        report.append(dict(job=name, machine=mach, colour=colour, process=proc, pieces=sum(q for _, q in items),
                           ok=ok, returncode=r.returncode, plates=len(st),
                           hours=round(sum(s for s, _ in st) / 3600, 1), grams=round(sum(g for _, g in st)),
                           plate_hours=[round(s / 3600, 1) for s, _ in st]))
        print(json.dumps(report[-1], ensure_ascii=False), flush=True)
        if not ok:
            print("\n".join(log[-15:]), flush=True)
    for fn in glob.glob(os.path.join(a.out, "plate_*.gcode")) + glob.glob(os.path.join(a.out, "result.json")):
        os.remove(fn)
    rp = os.path.join(a.out, "slice_report.json")
    if a.only and os.path.exists(rp):                 # 只重切一部分时，把结果并回原报告
        done = {r["job"]: r for r in report}
        report = [done.pop(r["job"], r) for r in json.load(open(rp))] + list(done.values())
    json.dump(report, open(rp, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
