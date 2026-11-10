"""Minimal signed-distance-field toolkit (numpy, float32).

Every function takes P = (X, Y, Z) broadcastable arrays in millimetres and
returns a distance field (negative inside).
"""
import numpy as np

F = np.float32


def _len(*c):
    s = c[0] * c[0]
    for ci in c[1:]:
        s = s + ci * ci
    return np.sqrt(s)


def sphere(P, c, r):
    return _len(P[0] - c[0], P[1] - c[1], P[2] - c[2]) - F(r)


def ellipsoid(P, c, r):
    x, y, z = P[0] - c[0], P[1] - c[1], P[2] - c[2]
    k0 = _len(x / r[0], y / r[1], z / r[2])
    k1 = _len(x / (r[0] ** 2), y / (r[1] ** 2), z / (r[2] ** 2))
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)


def rbox(P, c, h, r):
    """Axis-aligned box, half-size h, corner radius r."""
    qx = np.abs(P[0] - c[0]) - (h[0] - r)
    qy = np.abs(P[1] - c[1]) - (h[1] - r)
    qz = np.abs(P[2] - c[2]) - (h[2] - r)
    out = _len(np.maximum(qx, 0), np.maximum(qy, 0), np.maximum(qz, 0))
    ins = np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0)
    return out + ins - F(r)


def obox(P, c, axes, h, r):
    """Oriented rounded box. axes: 3 orthonormal vectors."""
    d = (P[0] - c[0], P[1] - c[1], P[2] - c[2])
    q = []
    for a, hh in zip(axes, h):
        q.append(np.abs(d[0] * a[0] + d[1] * a[1] + d[2] * a[2]) - (hh - r))
    out = _len(*[np.maximum(v, 0) for v in q])
    ins = np.minimum(np.maximum(q[0], np.maximum(q[1], q[2])), 0)
    return out + ins - F(r)


def rcone(P, a, b, r1, r2):
    """Round cone (capsule with two radii) between points a and b."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ba = b - a
    l2 = float(ba @ ba)
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = (P[0] - a[0], P[1] - a[1], P[2] - a[2])
    y = pa[0] * ba[0] + pa[1] * ba[1] + pa[2] * ba[2]
    z = y - l2
    vx = pa[0] * l2 - ba[0] * y
    vy = pa[1] * l2 - ba[1] * y
    vz = pa[2] * l2 - ba[2] * y
    x2 = vx * vx + vy * vy + vz * vz
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    d1 = np.sqrt(x2 + z2) * il2 - r2
    d2 = np.sqrt(x2 + y2) * il2 - r1
    d3 = (np.sqrt(x2 * a2 * il2) + y * rr) * il2 - r1
    return np.where(np.sign(z) * a2 * z2 > k, d1,
                    np.where(np.sign(y) * a2 * y2 < k, d2, d3)).astype(F)


def cyl(P, a, b, r, rnd=0.0):
    """Capped cylinder between a and b (optionally rounded edges)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ba = b - a
    L = float(np.linalg.norm(ba))
    u = ba / L
    pa = (P[0] - a[0], P[1] - a[1], P[2] - a[2])
    t = pa[0] * u[0] + pa[1] * u[1] + pa[2] * u[2]
    rad = np.sqrt(np.maximum(
        pa[0] * pa[0] + pa[1] * pa[1] + pa[2] * pa[2] - t * t, 0))
    dx = rad - (r - rnd)
    dy = np.abs(t - L / 2) - (L / 2 - rnd)
    out = _len(np.maximum(dx, 0), np.maximum(dy, 0))
    return (out + np.minimum(np.maximum(dx, dy), 0) - rnd).astype(F)


def plane(P, p0, n):
    """Signed distance to plane through p0; positive on the side n points."""
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    return ((P[0] - p0[0]) * n[0] + (P[1] - p0[1]) * n[1]
            + (P[2] - p0[2]) * n[2])


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def union(*d):
    r = d[0]
    for x in d[1:]:
        r = np.minimum(r, x)
    return r


def inter(*d):
    r = d[0]
    for x in d[1:]:
        r = np.maximum(r, x)
    return r


def sub(a, b):
    return np.maximum(a, -b)


def groove(d, g, w=2.0, depth=1.4):
    """Carve a panel line where |g| < w/2, `depth` deep into the surface."""
    return np.maximum(d, -np.maximum(np.abs(g) - w / 2, -(d + depth)))


def mirror_x(P):
    return (-P[0], P[1], P[2])
