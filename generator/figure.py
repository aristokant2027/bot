"""Geometry definition of the HUMAN+ "静弧" static display figure.

World frame (assembled pose), millimetres:
  +X = figure's left, -Y = front, +Z = up, Z=0 is the sole plane (top of base).
All left/right parts are defined for the LEFT (+X) side and mirrored.
"""
from dataclasses import dataclass, field
import numpy as np
from sdf import (sphere, ellipsoid, rbox, obox, rcone, cyl, plane, smin, smax,
                 union, inter, sub, groove)

# ---------------------------------------------------------------- constants
CLR = 0.20          # socket clearance per side for tenons (mm)
T24, T16 = 24.0, 16.0
PIN_D = 6.0
PIN_HOLE = 6.4      # sliding hole in shells
PIN_HOLE_TENON = 6.8
DOWEL_HOLE = 6.3    # snug hole for short alignment dowels

HIP_X = 82.0
SH = np.array([172.0, 0.0, 1040.0])          # left shoulder centre
ARM_TILT = np.radians(7.0)
AD = np.array([np.sin(ARM_TILT), 0, -np.cos(ARM_TILT)])   # arm axis (down/out)
AW = np.array([np.cos(ARM_TILT), 0, np.sin(ARM_TILT)])    # outward normal
AF = np.array([0.0, -1.0, 0.0])                           # forward


def A(t, out=0.0, fwd=0.0):
    """Point in the left-arm frame."""
    return SH + AD * t + AW * out + AF * fwd


def arm_t(P):
    return (P[0] - SH[0]) * AD[0] + (P[2] - SH[2]) * AD[2]


# arm stations (distance along arm axis from shoulder centre)
T_ELBOW0, T_ELBOW1 = 190.0, 240.0
T_WRIST0, T_WRIST1 = 430.0, 448.0

Z_SEAM = 1232.0     # head lower / upper split (top edge of visor)
Z_VIS0 = 1206.0


@dataclass
class Part:
    name: str
    color: str                  # W / K / P
    fn: object                  # sdf(P) -> d
    lo: tuple
    hi: tuple
    down: tuple                 # world direction that faces the bed
    clips: list = field(default_factory=list)   # (point, keep_direction)
    side: bool = False          # True -> exported as _L and mirrored _R
    voxel: float = 0.8
    note: str = ""


# ------------------------------------------------------------------- head
def head_d(P):
    return smin(ellipsoid(P, (0, -4, 1216), (71, 86, 84)),
                ellipsoid(P, (0, -14, 1182), (58, 70, 56)), 20)


def _ang_mm(P):
    """Angle from straight-ahead, expressed as arc length (mm) at r=80."""
    return np.arctan2(np.abs(P[0]), -(P[1] - 10.0)) * 80.0


def _deg(a):
    return np.radians(a) * 80.0


def head_lower(P):
    dh = head_d(P)
    recess = inter(-(dh + 11.2), (Z_VIS0 - 0.2) - P[2], _ang_mm(P) - _deg(78.3))
    return sub(dh, recess)


STRIPE_P0 = (0.0, 8.0, Z_SEAM)
STRIPE_N = (0.0, np.cos(np.radians(35)), -np.sin(np.radians(35)))


def head_upper(P):
    dh = head_d(P)
    g = plane(P, STRIPE_P0, STRIPE_N)
    slot = inter(-(dh + 4.2), np.abs(g) - 4.7, 4.0 - P[0])
    return sub(dh, slot)


def head_stripe(P):
    dh = head_d(P)
    g = plane(P, STRIPE_P0, STRIPE_N)
    return inter(dh - 0.3, -(dh + 4.0), np.abs(g) - 4.5, 4.3 - P[0],
                 (Z_SEAM + 0.3) - P[2])


def visor(P):
    dh = head_d(P)
    a = _ang_mm(P)
    body = inter(dh + 1.5, -(dh + 11.0), Z_VIS0 - P[2], P[2] - Z_SEAM,
                 a - _deg(78.0))
    pocket = inter(-(dh + 5.7), 1213.8 - P[2], P[2] - 1224.2,
                   _deg(9.6) - a, a - _deg(52.4))
    return sub(body, pocket)


def eye_strip(P):
    dh = head_d(P)
    a = _ang_mm(P)
    return inter(dh + 0.9, -(dh + 5.5), 1214.0 - P[2], P[2] - 1224.0,
                 _deg(10.0) - a, a - _deg(52.0), -P[0])


def neck(P):
    d = cyl(P, (0, 5, 1080), (0, 5, 1145), 31, 2)
    for z in (1099, 1108, 1117, 1126):
        d = groove(d, P[2] - z, 3.0, 2.0)
    return d


# ------------------------------------------------------------------ torso
def chest(P):
    ax = np.abs(P[0])
    Q = (ax, P[1], P[2])
    up = rbox(P, (0, 8, 1012), (120, 78, 78), 26)
    # shoulders slope down from the collar (trapezius line)
    up = smax(up, plane(Q, (40, 0, 1090), (0.22, 0, 1.0)), 12)
    lo = rbox(P, (0, 8, 935), (90, 66, 65), 24)
    d = smin(up, lo, 30)
    # pectoral plates: flat slabs with a 45-degree chamfered underside
    pec = ellipsoid(Q, (54, -56, 1012), (60, 22, 46))
    pec = smax(pec, plane(Q, (0, -78, 985), (0, -0.7, -0.7)), 3)
    pec = smax(pec, plane(Q, (4, 0, 0), (-1, 0, 0)), 3)
    d = smin(d, pec, 8)
    # back plate
    d = smin(d, rbox(P, (0, 80, 1000), (70, 12, 60), 8), 8)
    d = groove(d, P[0] + 1e3 * (P[2] > 960) + 1e3 * (P[1] > 0), 2.0, 1.5)
    d = groove(d, P[2] - 925 + 1e3 * (P[1] > 20), 2.0, 1.5)
    return d


def abdomen(P):
    d = rbox(P, (0, 8, 845), (80, 58, 52), 22)
    for z in (832, 858):
        d = groove(d, P[2] - z, 3.0, 2.5)
    d = groove(d, P[0] + 1e3 * (P[1] > 0), 3.0, 2.5)
    return d


def pelvis(P):
    ax = np.abs(P[0])
    Q = (ax, P[1], P[2])
    hipbox = rbox(P, (0, 6, 752), (141, 74, 64), 26)
    hipbox = smax(hipbox, plane(Q, (141, 0, 700), (1.0, 0, 0.30)), 10)   # taper to waist
    hipbox = inter(hipbox, 700.0 - P[2])
    crotch = rbox(P, (0, 6, 692), (23, 60, 44), 10)
    plate = rbox(P, (0, -66, 735), (34, 12, 50), 8)                     # front plate
    plate = smax(plate, plane(Q, (34, 0, 690), (0.8, 0, -0.6)), 4)
    d = union(hipbox, smin(crotch, plate, 6))
    d = groove(d, P[2] - 778, 2.0, 1.5)
    return d


# -------------------------------------------------------------------- leg
def hip(P):
    d = cyl(P, (HIP_X, 5, 650), (HIP_X, 5, 705), 53, 4)
    return groove(d, P[2] - 677.5, 4.0, 3.0)


def thigh(P):
    d = rcone(P, (HIP_X, 5, 665), (HIP_X, 0, 400), 59, 45)
    quad = ellipsoid(P, (HIP_X, -26, 550), (46, 40, 100))
    d = smin(d, quad, 14)
    d = groove(d, P[2] - 620, 2.0, 1.5)
    d = groove(d, P[2] - 440, 2.0, 1.5)
    return d


def knee(P):
    d = rbox(P, (HIP_X, 0, 370), (42, 38, 48), 8)
    disc = cyl(P, (HIP_X - 49, 0, 370), (HIP_X + 49, 0, 370), 30, 3)
    return union(d, disc)


def kneecap(P):
    d = rbox(P, (HIP_X, -46, 372), (30, 10, 36), 7)
    hexd = (np.abs(P[0] - HIP_X) * 0.5 + np.abs(P[2] - 372) * 0.866) - 30
    return smax(d, hexd, 4)


def shin(P):
    d = rcone(P, (HIP_X, 0, 335), (HIP_X, 8, 90), 45, 34)
    calf = ellipsoid(P, (HIP_X, 26, 250), (40, 36, 80))
    guard = ellipsoid(P, (HIP_X, -24, 215), (33, 28, 110))
    d = smin(smin(d, calf, 14), guard, 10)
    d = groove(d, P[2] - 300, 2.0, 1.5)
    d = groove(d, P[2] - 125, 2.0, 1.5)
    return d


def ankle(P):
    d = cyl(P, (HIP_X, 10, 55), (HIP_X, 10, 100), 31, 3)
    disc = cyl(P, (HIP_X - 39, 10, 78), (HIP_X + 39, 10, 78), 19, 3)
    return union(d, disc)


def foot(P):
    toe = rbox(P, (HIP_X, -52, 24), (41, 64, 40), 20)
    toe = smax(toe, plane(P, (0, -20, 62), (0, -0.30, 1.0)), 10)
    heel = rbox(P, (HIP_X, 22, 28), (39, 48, 44), 18)
    d = smin(toe, heel, 12)
    d = groove(d, P[2] - 12, 2.0, 1.5)
    return d


# -------------------------------------------------------------------- arm
def shoulder(P):
    return rbox(P, (125, 0, 1040), (30, 40, 42), 10)


def upper_arm(P):
    delt = ellipsoid(P, SH, (46, 46, 50))
    arm = rcone(P, A(20), A(T_ELBOW0), 41, 33)
    arm = smin(arm, ellipsoid(P, A(105, 0, 12), (36, 36, 60)), 10)
    d = smin(delt, arm, 12)
    d = groove(d, arm_t(P) - 62, 2.0, 1.5)
    return d


def elbow(P):
    d = cyl(P, A(T_ELBOW0 - 5), A(T_ELBOW1 + 5), 30, 2)
    c = A(215)
    disc = cyl(P, c + np.array([0, -33, 0]), c + np.array([0, 33, 0]), 22, 3)
    return union(d, disc)


def forearm(P):
    d = rcone(P, A(T_ELBOW1), A(T_WRIST0), 31, 22)
    bulge = rcone(P, A(262, 4), A(335, 2), 35, 27)
    d = smin(d, bulge, 10)
    d = groove(d, arm_t(P) - 400, 2.0, 1.5)
    return d


def wrist_band(P):
    return cyl(P, A(T_WRIST0 - 4), A(T_WRIST1 + 4), 23.5, 0.01)


def hand(P):
    stub = rcone(P, A(T_WRIST1 - 2), A(474), 20.5, 18)
    palm = obox(P, A(494), (AD, AF, AW), (30, 25, 14.5), 11)
    d = smin(stub, palm, 9)
    # fingers: index (front) .. pinky (back); curl towards the body (-AW)
    ys = (19.8, 6.6, -6.6, -19.8)          # fwd offsets
    Ls = (64, 72, 66, 52)
    fing = None
    for yf, L in zip(ys, Ls):
        p = A(516, 0, yf)
        r = 6.3
        ang = 0.0
        for frac, bend, r2 in ((0.45, 8, 6.1), (0.30, 14, 5.8), (0.25, 14, 5.5)):
            ang += np.radians(bend)
            dirv = AD * np.cos(ang) - AW * np.sin(ang)
            q = p + dirv * L * frac
            seg = rcone(P, p, q, r, r2)
            fing = seg if fing is None else np.minimum(fing, seg)
            p, r = q, r2
    d = smin(d, fing, 3)
    th = union(rcone(P, A(478, -2, 22), A(512, -7, 37), 8.5, 7.2),
               rcone(P, A(512, -7, 37), A(540, -13, 39), 7.2, 6.2))
    d = smin(d, th, 5)
    return d


# ----------------------------------------------------------- part registry
Zp, Zm = (0, 0, 1), (0, 0, -1)


def keepz(z0, z1):
    return [((0, 0, z0), Zp), ((0, 0, z1), Zm)]


def keept(t0, t1):
    c = []
    if t0 is not None:
        c.append((tuple(A(t0)), tuple(AD)))
    if t1 is not None:
        c.append((tuple(A(t1)), tuple(-AD)))
    return c


PARTS = [
    # --- head
    Part("head_lower", "W", head_lower, (-80, -100, 1125), (80, 92, 1240), Zp,
         keepz(1137, Z_SEAM), voxel=0.7, note="倒置打印(大面朝下)"),
    Part("head_upper", "W", head_upper, (-80, -100, 1225), (80, 92, 1308), Zm,
         [((0, 0, Z_SEAM), Zp)], voxel=0.7),
    Part("head_stripe", "P", head_stripe, (0, -30, 1225), (80, 92, 1308),
         STRIPE_N, voxel=0.4, note="头部左侧紫色饰条, 胶粘入槽"),
    Part("visor", "K", visor, (-85, -100, 1200), (85, 40, 1238), Zm, voxel=0.4,
         note="面罩, 从上方落入头下半, 被头顶盖压住"),
    Part("eye_strip", "P", eye_strip, (0, -100, 1208), (85, 20, 1230), Zm,
         side=True, voxel=0.3, note="眼灯条, 嵌入面罩"),
    Part("neck", "K", neck, (-40, -35, 1075), (40, 45, 1150), Zm,
         keepz(1090, 1137)),
    # --- torso
    Part("chest", "W", chest, (-130, -100, 860), (130, 96, 1100), Zm,
         keepz(885, 1090) + [((112, 0, 0), (-1, 0, 0)), ((-112, 0, 0), (1, 0, 0))],
         note="胸甲, 底面朝下; 胸肌下缘建议开支撑"),
    Part("abdomen", "K", abdomen, (-90, -60, 790), (90, 76, 900), Zm,
         keepz(805, 885)),
    Part("pelvis", "W", pelvis, (-150, -90, 635), (150, 90, 822), Zp,
         keepz(600, 805) + [], note="倒置打印(腰面朝下); 仅 A2L 可整体打印"),
    # --- leg (left, mirrored)
    Part("hip", "K", hip, (25, -52, 645), (140, 62, 710), Zm, keepz(655, 700),
         side=True),
    Part("thigh", "W", thigh, (12, -75, 395), (152, 72, 670), Zp,
         keepz(410, 655), side=True, note="倒置打印(髋端朝下)"),
    Part("knee", "K", knee, (25, -45, 322), (140, 45, 418), Zm,
         keepz(330, 410), side=True),
    Part("kneecap", "W", kneecap, (45, -62, 330), (120, -30, 414), (0, 1, 0),
         [((0, -38, 0), (0, -1, 0))], side=True, voxel=0.6),
    Part("shin", "W", shin, (28, -60, 85), (136, 70, 340), Zp,
         keepz(95, 330), side=True, note="倒置打印(膝端朝下)"),
    Part("ankle", "K", ankle, (35, -28, 55), (130, 48, 100), Zm,
         keepz(62, 95), side=True),
    Part("foot", "W", foot, (35, -122, -8), (130, 76, 70), Zm,
         keepz(0, 62), side=True),
    # --- arm (left, mirrored)
    Part("shoulder", "K", shoulder, (100, -45, 990), (150, 45, 1090),
         (-1, 0, 0), [((112, 0, 0), (1, 0, 0)), ((138, 0, 0), (-1, 0, 0))],
         side=True),
    Part("upper_arm", "W", upper_arm, (120, -56, 835), (250, 52, 1098),
         tuple(AD), [((138, 0, 0), (1, 0, 0))] + keept(None, T_ELBOW0),
         side=True, note="肘端朝下竖直打印"),
    Part("elbow", "K", elbow, (155, -40, 780), (245, 40, 870), tuple(AD),
         keept(T_ELBOW0, T_ELBOW1), side=True),
    Part("forearm", "W", forearm, (160, -40, 600), (270, 40, 815),
         tuple(-AD), keept(T_ELBOW1, T_WRIST0), side=True),
    Part("wrist_band", "P", wrist_band, (190, -30, 585), (265, 30, 625),
         tuple(-AD), keept(T_WRIST0, T_WRIST1), side=True, voxel=0.5),
    Part("hand", "W", hand, (175, -52, 440), (275, 40, 605), tuple(-AD),
         keept(T_WRIST1, None), side=True, voxel=0.5,
         note="腕面朝下, 手指朝上; 建议树状支撑"),
]


@dataclass
class Tenon:
    name: str
    size: float
    p0: tuple          # start point (world)
    axis: tuple
    length: float
    e1: tuple          # orientation of one side face
    parts: list        # part names crossed
    sym: bool = True   # mirror to the right side as well


TENONS = [
    Tenon("neck", T24, (0, 5, 1045), Zp, 140, (1, 0, 0),
          ["chest", "neck", "head_lower"], sym=False),
    Tenon("waist", T24, (42, 5, 770), Zp, 160, (1, 0, 0),
          ["pelvis", "abdomen", "chest"]),
    Tenon("hip", T24, (HIP_X, 5, 605), Zp, 140, (1, 0, 0),
          ["thigh", "hip", "pelvis"]),
    Tenon("knee", T24, (HIP_X, 0, 280), Zp, 180, (1, 0, 0),
          ["shin", "knee", "thigh"]),
    Tenon("ankle", T24, (HIP_X, 10, -30), Zp, 180, (1, 0, 0),
          ["base", "foot", "ankle", "shin"]),
    Tenon("shoulder", T24, (65, 0, 1040), (1, 0, 0), 110, (0, 1, 0),
          ["chest", "shoulder", "upper_arm"]),
    Tenon("elbow", T16, tuple(A(150)), tuple(AD), 130, (0, 1, 0),
          ["upper_arm", "elbow", "forearm"]),
    Tenon("wrist", T16, tuple(A(390)), tuple(AD), 100, (0, 1, 0),
          ["forearm", "wrist_band", "hand"]),
]


@dataclass
class Hole:
    name: str
    d: float
    p0: tuple
    axis: tuple
    length: float       # cutter length, centred on p0
    parts: list
    sym: bool = True


Yp = (0, 1, 0)
HOLES = [
    # cross pins (through holes along Y) that lock the hanging arm joints
    Hole("pin_shoulder", PIN_HOLE, (160, 0, 1040), Yp, 200, ["upper_arm"]),
    Hole("pin_elbow_up", PIN_HOLE, tuple(A(165)), Yp, 200, ["upper_arm"]),
    Hole("pin_elbow_lo", PIN_HOLE, tuple(A(265)), Yp, 200, ["forearm"]),
    Hole("pin_wrist_up", PIN_HOLE, tuple(A(405)), Yp, 200, ["forearm"]),
    Hole("pin_wrist_lo", PIN_HOLE, tuple(A(475)), Yp, 200, ["hand"]),
    # alignment dowels (blind, 10.5 mm each side)
    Hole("dowel_kneecap_a", DOWEL_HOLE, (HIP_X, -38, 356), Yp, 21,
         ["knee", "kneecap"]),
    Hole("dowel_kneecap_b", DOWEL_HOLE, (HIP_X, -38, 388), Yp, 21,
         ["knee", "kneecap"]),
    Hole("dowel_head_a", DOWEL_HOLE, (40, 12, Z_SEAM), Zp, 21,
         ["head_lower", "head_upper"]),
    Hole("dowel_head_c", DOWEL_HOLE, (0, 56, Z_SEAM), Zp, 21,
         ["head_lower", "head_upper"], sym=False),
]

# shoulder accent pad (stadium), left side
PAD_X, PAD_Y, PAD_R = 125.0, 17.0, 6.5
PAD_Z0, PAD_Z1, PAD_TOP = 1077.0, 1082.0, 1086.0

# base
BASE_W, BASE_D, BASE_T, BASE_CY, BASE_R = 480.0, 360.0, 35.0, -30.0, 30.0
BOWTIES = [  # (x, y, across-seam direction)
    (0, -165, "x"), (0, -75, "x"), (0, 15, "x"), (0, 105, "x"),
    (-180, BASE_CY, "y"), (-40, BASE_CY, "y"), (40, BASE_CY, "y"),
    (180, BASE_CY, "y"),
]
