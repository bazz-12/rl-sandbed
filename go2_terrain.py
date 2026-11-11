"""
go2_terrain.py - [P3.1] Terrain of the Go2 environment: the curriculum grid of tiles (rows = difficulty levels, columns
= terrain types), a planar slope for evaluation, and the ground height under any point. numpy and torch, no Genesis:
go2_env.py builds the Genesis entity from it, go2_metrics.py and go2_feet_plot.py read ground heights from it offline.

    python go2_terrain.py                                       # default grid: tile table + terrain_grid.png
    python go2_terrain.py --set num_levels=6 max_slope_deg=20   # any key of TERRAIN_DEFAULTS
    python go2_terrain.py --run logs/go2-p31                    # the grid a run trained on (its terrain.npz)
    python go2_terrain.py --check                               # self-checks (numpy only)
    python go2_terrain.py --check --genesis                     # + heights against Genesis' own contact surface

Grid layout (legged_gym Terrain.curiculum(), Rudin et al. 2022)
    x   one row of tiles per difficulty level l = 0 .. num_levels - 1, difficulty d = l / (num_levels - 1)
    y   one column per terrain type, in the order of the "types" dict, `copies` columns per type
    a flat border all around, so a robot that walks out of the grid stays on the ground
    the world origin is the corner of tile (0, 0): tile (l, c) spans x in [l T, (l + 1) T], y in [c T, (c + 1) T],
    T = tile_size; tile origin = its centre, at the ground height there (robots spawn around it)

Tile types at difficulty d (keys of TERRAIN_DEFAULTS; a pair (a, b) means a at d = 0 and b at d = 1, linear between)
    flat        no relief at any level: keeps the flat gait in the mix
    rough       random heights in +- rough_amp on a grid of rough_cell, linear in between (legged_gym random_uniform)
    hill        pyramid with 4 planar faces at slope d x max_slope_deg and a flat top of side `platform`; robots spawn
                on top, so walking away from the spawn point is DOWNHILL
    pit         the same pyramid upside down: robots spawn at the bottom, walking away is UPHILL
    wave        h = A sin(2 pi x / L) sin(2 pi y / L), A = wave_amp: bumps and dips, whole half-waves per tile
    obstacles   obstacle_count rectangles of side obstacle_size, raised or sunk by h or h/2 (h = obstacle_height),
                flat centre of side `platform` (legged_gym discrete_obstacles_terrain)
Every tile is exactly 0 on its outer ring of vertices (shared with its neighbours), so any two tiles join.

Why pyramids and not single inclines (Genesis' sloped_terrain): a 25 deg incline over an 8 m tile rises 3.7 m and leaves
a 3.7 m cliff to the next tile; legged_gym and Isaac Lab use pyramids for this reason. Their pyramids are bilinear
(h ~ x y: the gradient changes over a face), these have planar faces (h ~ distance to the nearest edge), so a hill at
level l is a d x max_slope_deg slope everywhere on its faces. A planar slope of any extent is mode "slope": a tilted
plane, for evaluation (go2_metrics.py --slope) and the bench (go2_fsm_sim.py --slope).

Heights follow Genesis 1.4.2's terrain conventions (genesis/utils/terrain.py, collider/narrowphase.py): vertex
height_field[i, j] is at x = x0 + i hs, y = y0 + j hs, and each cell is split along its (i+1, j)-(i, j+1) diagonal. The
lookup interpolates on the same triangles, so it returns the surface the feet stand on (--check --genesis compares it
with Genesis' TerrainEntity.get_terrain_height and with a sphere dropped on the terrain).

[P3.2] Every ground also gives its unit normal (normal_at: torch, normal_np: numpy), and contact_slip_speed() is the
one definition of foot slip used by go2_env.py (foot_slip penalty, privileged observation), go2_metrics.py (slip
tables) and go2_feet_plot.py: the speed of the foot sphere's contact point along the ground's tangent plane.
--check tests it (rolling, sliding, sinking on a 20 deg face); --check --genesis compares the normal with the contact
normal Genesis' solver computes when a sphere touches a sloped face.
"""

import argparse
import ast
import copy
import hashlib
import json
import math
import os

import numpy as np

TILE_TYPES = ("flat", "rough", "hill", "pit", "wave", "obstacles")
GENERATOR_VERSION = 1  # part of the cache key: bump it whenever a tile generator changes

TERRAIN_DEFAULTS = {
    "mode": "grid",  # "grid": tiles + curriculum (training); "slope": one tilted plane (evaluation, bench)
    # --- layout
    "types": {"flat": 0.10, "rough": 0.20, "hill": 0.175, "pit": 0.175, "wave": 0.15, "obstacles": 0.20},  # env shares
    "copies": 1,  # columns per type: more random instances of each type (wider grid)
    "num_levels": 10,  # rows of tiles (legged_gym num_rows)
    "tile_size": 8.0,  # [m] legged_gym
    "border": 5.0,  # [m] flat margin around the grid (legged_gym: 25 m; see the build cost in TEST_AND_TINKER_P3.md)
    "horizontal_scale": 0.1,  # [m] heightfield cell (legged_gym); 0.25 is Genesis' default, too coarse for rough
    "vertical_scale": 0.005,  # [m] unit of the heightfield passed to Genesis (heights stay floats, no rounding)
    "seed": 0,  # terrain generation only (independent of the training seed)
    # --- tiles
    "rough_amp": (0.005, 0.03),  # [m] half range of the random heights: 1 cm to 6 cm peak to peak
    "rough_cell": 0.2,  # [m] spacing of the random heights (legged_gym downsampled_scale)
    "max_slope_deg": 25.0,  # hill and pit faces at the top level (0 deg at level 0)
    "platform": 2.0,  # [m] side of the flat centre of hill, pit and obstacle tiles
    "slope_noise": 0.0,  # [m] half range of rough noise on hill and pit tiles (legged_gym "rough slope": 0.05)
    "wave_amp": (0.0, 0.15),  # [m] at 4 m wavelength: 0 to 13 deg steepest slope
    "wave_length": 4.0,  # [m] snapped so that a tile holds a whole number of half waves
    "obstacle_height": (0.02, 0.10),  # [m] the foot_clearance target is 7.5 cm
    "obstacle_size": (0.3, 1.0),  # [m] side of the rectangles
    "obstacle_count": 50,  # per tile
    # --- spawning (go2_env.py)
    "spawn_xy_noise": 0.5,  # [m] uniform +- around the tile centre (legged_gym: 1 m), inside the 2 m platforms
    "random_yaw": True,  # heading uniform in [-pi, pi) at each reset
    "spawn_tile": None,  # e.g. ["hill", 9]: every env on that tile, curriculum off (go2_teleop.py)
    # --- terrain curriculum (go2_env.py): Rudin et al. 2022, legged_gym _update_terrain_curriculum
    "curriculum": True,
    "max_init_level": 4,  # initial levels uniform in 0 .. max_init_level (legged_gym: 5)
    "up_distance": None,  # [m] walked this far from the spawn point -> next level; None = tile_size / 2
    "down_ratio": 0.5,  # progress along the commands < down_ratio x commanded distance -> previous level
    "down_min_cmd_dist": 1.0,  # [m] judge the progress only if the episode asked for at least this distance
    "fall_moves_down": True,  # an episode that ended by a fall (not a time-out) -> previous level
    # --- mode "slope"
    "slope_deg": 15.0,  # plane through the origin, rising along +x (negative: falling along +x)
}
# keys that change the generated heights (the cache key); spawning and curriculum keys do not
GENERATION_KEYS = ("types", "copies", "num_levels", "tile_size", "border", "horizontal_scale", "seed", "rough_amp",
                   "rough_cell", "max_slope_deg", "platform", "slope_noise", "wave_amp", "wave_length",
                   "obstacle_height", "obstacle_size", "obstacle_count")
FOOTPRINT_HALF = 0.4  # [m] half side of the square under a robot whose highest point sets the spawn height


def terrain_config(cfg):
    """TERRAIN_DEFAULTS updated with cfg (dict). None or {} = flat ground (returns None)."""
    if not cfg:
        return None
    unknown = set(cfg) - set(TERRAIN_DEFAULTS) - {"file"}
    if unknown:
        raise ValueError(f"unknown terrain keys {sorted(unknown)}; valid: {sorted(TERRAIN_DEFAULTS)}")
    out = copy.deepcopy(TERRAIN_DEFAULTS)
    out.update(copy.deepcopy(cfg))
    if out["mode"] not in ("grid", "slope"):
        raise ValueError(f"terrain mode must be 'grid' or 'slope', not {out['mode']!r}")
    if out["mode"] == "grid":
        bad = set(out["types"]) - set(TILE_TYPES)
        if bad or not out["types"] or min(out["types"].values()) <= 0:
            raise ValueError(f"terrain types must be a non-empty {{type: share > 0}} dict with types in {TILE_TYPES}")
        n = out["tile_size"] / out["horizontal_scale"]
        if abs(n - round(n)) > 1e-6:
            raise ValueError("tile_size must be a multiple of horizontal_scale")
    return out


def lerp(value, d):
    """value at difficulty d: a pair (a, b) -> a + d (b - a); a number -> itself."""
    if isinstance(value, (tuple, list)):
        return float(value[0]) + d * (float(value[1]) - float(value[0]))
    return float(value)


def cache_key(cfg):
    """Hash of everything that changes the heights (Genesis' Terrain(name=...) caches the same way)."""
    keys = {k: cfg[k] for k in GENERATION_KEYS}
    text = json.dumps(keys, sort_keys=True, default=list) + f"|v{GENERATOR_VERSION}"
    return hashlib.sha1(text.encode()).hexdigest()[:12]


# ======================================================================================================================
# Tiles
# ======================================================================================================================
def random_field(n, hs, cell, amp, rng):
    """[n+1, n+1] heights: uniform in [-amp, amp] on a grid of spacing `cell`, linear in between."""
    size = n * hs
    m = max(1, int(round(size / cell)))
    coarse = rng.uniform(-amp, amp, size=(m + 1, m + 1))
    x = np.linspace(0.0, m, n + 1)  # coarse coordinate of each fine vertex
    i0 = np.clip(np.floor(x).astype(int), 0, m - 1)
    f = x - i0
    rows = coarse[i0] * (1.0 - f)[:, None] + coarse[i0 + 1] * f[:, None]  # [n+1, m+1]
    return rows[:, i0] * (1.0 - f)[None, :] + rows[:, i0 + 1] * f[None, :]  # [n+1, n+1]


def tile_slope_deg(kind, d, cfg):
    """Steepest slope [deg] the tile type has by construction at difficulty d (fall threshold of go2_metrics.py)."""
    if kind in ("hill", "pit"):
        return d * cfg["max_slope_deg"]
    if kind == "wave":
        k = max(1, int(round(2 * cfg["tile_size"] / cfg["wave_length"])))
        return math.degrees(math.atan(lerp(cfg["wave_amp"], d) * 2 * math.pi / (2 * cfg["tile_size"] / k)))
    return 0.0  # rough and obstacles: local, not a slope the base follows


BODY_LENGTH = 0.4  # [m] about the Go2's front-to-rear foot spacing: relief h across it tilts the base by atan(h / L)


def tile_tilt_deg(kind, d, cfg):
    """Base tilt [deg] the tile's relief can cause without anything going wrong: the slope of hill, pit and wave faces,
    atan(relief / BODY_LENGTH) for rough and obstacle tiles. go2_metrics.py adds it to --fall_deg on terrain."""
    if kind == "rough":
        return math.degrees(math.atan(2.0 * lerp(cfg["rough_amp"], d) / BODY_LENGTH))
    if kind == "obstacles":
        return math.degrees(math.atan(lerp(cfg["obstacle_height"], d) / BODY_LENGTH))
    return tile_slope_deg(kind, d, cfg)


def make_tile(kind, d, cfg, rng):
    """Heights [m] of one tile, [n+1, n+1] vertices, at difficulty d. The outer ring of vertices is 0: neighbouring
    tiles share it, so every tile joins every other one, whatever their types."""
    h = _tile_heights(kind, d, cfg, rng)
    h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = 0.0
    return h


def _tile_heights(kind, d, cfg, rng):
    hs, size = cfg["horizontal_scale"], cfg["tile_size"]
    n = int(round(size / hs))
    u = np.arange(n + 1) * hs  # tile-local coordinate of the vertices [m], along x (rows) and y (columns)
    if kind == "flat":
        return np.zeros((n + 1, n + 1))
    if kind == "rough":
        return random_field(n, hs, cfg["rough_cell"], lerp(cfg["rough_amp"], d), rng)
    if kind in ("hill", "pit"):
        slope = math.tan(math.radians(d * cfg["max_slope_deg"]))
        to_edge = np.minimum(u, size - u)
        dist = np.minimum(to_edge[:, None], to_edge[None, :])  # distance to the nearest tile edge
        h = slope * np.minimum(dist, 0.5 * (size - cfg["platform"]))  # planar faces, flat platform on top
        if cfg["slope_noise"] > 0:
            h = h + random_field(n, hs, cfg["rough_cell"], cfg["slope_noise"], rng)
        return h if kind == "hill" else -h
    if kind == "wave":
        k = max(1, int(round(2 * size / cfg["wave_length"])))  # half waves per tile: edges at 0
        s = np.sin(math.pi * k * u / size)
        return lerp(cfg["wave_amp"], d) * s[:, None] * s[None, :]
    if kind == "obstacles":
        h_max = lerp(cfg["obstacle_height"], d)
        h = np.zeros((n + 1, n + 1))
        lo, hi = cfg["obstacle_size"]
        for _ in range(int(cfg["obstacle_count"])):
            wx, wy = rng.uniform(lo, hi, size=2)
            x0, y0 = rng.uniform(0.0, size - wx), rng.uniform(0.0, size - wy)
            i0, i1 = int(round(x0 / hs)), int(round((x0 + wx) / hs))
            j0, j1 = int(round(y0 / hs)), int(round((y0 + wy) / hs))
            h[i0:i1 + 1, j0:j1 + 1] = rng.choice([-h_max, -0.5 * h_max, 0.5 * h_max, h_max])
        p0, p1 = int(round(0.5 * (size - cfg["platform"]) / hs)), int(round(0.5 * (size + cfg["platform"]) / hs))
        h[p0:p1 + 1, p0:p1 + 1] = 0.0
        return h
    raise ValueError(f"unknown tile type {kind!r}")


# ======================================================================================================================
# Heights on Genesis' triangles (numpy and torch)
# ======================================================================================================================
def _cells(gx, gy, rows, cols, floor, clip):
    i = clip(floor(gx), 0, rows - 2)
    j = clip(floor(gy), 0, cols - 2)
    u = clip(gx - i, 0.0, 1.0)
    v = clip(gy - j, 0.0, 1.0)
    return i, j, u, v


def hf_height_np(hf, origin, hs, xy):
    """Heights [m] at world xy [..., 2] (numpy) on the triangulated heightfield hf [R, C] (vertex (0, 0) at origin)."""
    xy = np.asarray(xy, dtype=np.float64)
    i, j, u, v = _cells((xy[..., 0] - origin[0]) / hs, (xy[..., 1] - origin[1]) / hs, *hf.shape,
                        lambda a: np.floor(a), np.clip)
    i, j = i.astype(np.int64), j.astype(np.int64)
    h00, h10, h01, h11 = hf[i, j], hf[i + 1, j], hf[i, j + 1], hf[i + 1, j + 1]
    lower = u + v <= 1.0  # triangle (i, j), (i+1, j), (i, j+1); else (i+1, j), (i, j+1), (i+1, j+1)
    return np.where(lower, h00 + u * (h10 - h00) + v * (h01 - h00),
                    h11 + (1.0 - u) * (h01 - h11) + (1.0 - v) * (h10 - h11))


def hf_gradient_np(hf, origin, hs, xy):
    """(dh/dx, dh/dy) [..., 2] of the triangle under each point (numpy)."""
    xy = np.asarray(xy, dtype=np.float64)
    i, j, u, v = _cells((xy[..., 0] - origin[0]) / hs, (xy[..., 1] - origin[1]) / hs, *hf.shape,
                        lambda a: np.floor(a), np.clip)
    i, j = i.astype(np.int64), j.astype(np.int64)
    h00, h10, h01, h11 = hf[i, j], hf[i + 1, j], hf[i, j + 1], hf[i + 1, j + 1]
    lower = u + v <= 1.0
    gx = np.where(lower, h10 - h00, h11 - h01) / hs
    gy = np.where(lower, h01 - h00, h11 - h10) / hs
    return np.stack((gx, gy), axis=-1)


class _TorchHeightField:
    """The same lookup on torch tensors (go2_env.py, every step: no GPU-CPU sync)."""

    def __init__(self, hf, origin, hs, device):
        import torch

        self.torch = torch
        self.hf = torch.as_tensor(hf, dtype=torch.float32, device=device).contiguous()
        self.rows, self.cols = hf.shape
        self.flat = self.hf.reshape(-1)
        self.x0, self.y0, self.hs = float(origin[0]), float(origin[1]), float(hs)

    def _corners(self, xy):
        t = self.torch
        gx, gy = (xy[..., 0] - self.x0) / self.hs, (xy[..., 1] - self.y0) / self.hs
        i = t.clamp(t.floor(gx), 0, self.rows - 2)
        j = t.clamp(t.floor(gy), 0, self.cols - 2)
        u, v = t.clamp(gx - i, 0.0, 1.0), t.clamp(gy - j, 0.0, 1.0)
        k = i.long() * self.cols + j.long()
        return (self.flat[k], self.flat[k + self.cols], self.flat[k + 1], self.flat[k + self.cols + 1]), u, v

    def height(self, xy):
        (h00, h10, h01, h11), u, v = self._corners(xy)
        return self.torch.where(u + v <= 1.0, h00 + u * (h10 - h00) + v * (h01 - h00),
                                h11 + (1.0 - u) * (h01 - h11) + (1.0 - v) * (h10 - h11))

    def gradient(self, xy):
        (h00, h10, h01, h11), u, v = self._corners(xy)
        lower = u + v <= 1.0
        gx = self.torch.where(lower, h10 - h00, h11 - h01) / self.hs
        gy = self.torch.where(lower, h01 - h00, h11 - h10) / self.hs
        return self.torch.stack((gx, gy), dim=-1)


def _footprint(half, n=5):
    """[n*n, 2] offsets covering the square +-half (the robot's footprint at any heading)."""
    s = np.linspace(-half, half, n)
    return np.stack(np.meshgrid(s, s, indexing="ij"), axis=-1).reshape(-1, 2)


# ======================================================================================================================
# Grounds: one interface for flat ground, a tilted plane and the tile grid
# ======================================================================================================================
def normal_from_gradient(g):
    """[P3.2] Unit normal [..., 3] of the surface z = h(x, y) from its gradient g = (dh/dx, dh/dy) [..., 2]:
    (-dh/dx, -dh/dy, 1) / norm. numpy arrays or torch tensors (same result type)."""
    nx, ny = -g[..., 0], -g[..., 1]
    inv = (nx * nx + ny * ny + 1.0) ** -0.5
    if isinstance(g, np.ndarray):
        return np.stack((nx * inv, ny * inv, inv), axis=-1)
    import torch

    return torch.stack((nx * inv, ny * inv, inv), dim=-1)


def contact_slip_speed(v, w, n, radius):
    """[P3.2] Slip speed of a sphere on the ground: speed of its contact point along the ground's tangent plane.
    v, w [..., 3]: velocity of the sphere centre and angular velocity (world frame); n [..., 3]: unit ground normal
    under the sphere; radius [m]. The contact point is the centre - radius n, so its velocity is v + w x (-radius n);
    its component along n (sinking into the soft contact, lifting off) is removed. A sphere rolling without slipping
    gives 0. On flat ground (n = (0, 0, 1)) this is the horizontal speed (vx - R wy, vy + R wx) of the P1 metric.
    Written component by component so that numpy and torch inputs both work (go2_env.py, go2_metrics.py,
    go2_feet_plot.py use this one function)."""
    rx, ry, rz = -radius * n[..., 0], -radius * n[..., 1], -radius * n[..., 2]
    px = v[..., 0] + w[..., 1] * rz - w[..., 2] * ry
    py = v[..., 1] + w[..., 2] * rx - w[..., 0] * rz
    pz = v[..., 2] + w[..., 0] * ry - w[..., 1] * rx
    pn = px * n[..., 0] + py * n[..., 1] + pz * n[..., 2]
    tx, ty, tz = px - pn * n[..., 0], py - pn * n[..., 1], pz - pn * n[..., 2]
    return (tx * tx + ty * ty + tz * tz) ** 0.5


class FlatGround:
    """The plane z = 0 (every run before P3, and the nominal evaluation world)."""

    kind = "flat"

    def to(self, device):
        return self

    def height_np(self, xy):
        return np.zeros(np.shape(xy)[:-1])

    def gradient_np(self, xy):
        return np.zeros(np.shape(xy))

    def height(self, xy):
        return xy[..., 0] * 0.0

    def gradient(self, xy):
        return xy * 0.0

    def normal_np(self, xy):
        """[P3.2] Unit ground normal [..., 3] under the points xy [..., 2] (numpy)."""
        return normal_from_gradient(self.gradient_np(xy))

    def normal_at(self, xy):
        """[P3.2] Unit ground normal [..., 3] under the points xy [..., 2] (torch). On the grid it is the normal of
        the heightfield triangle under the point, i.e. of the face the foot stands on."""
        return normal_from_gradient(self.gradient(xy))

    def spawn_height(self, xy):
        return self.height(xy)

    def describe(self):
        return "flat ground"


class PlaneGround(FlatGround):
    """A plane through the origin at slope_deg, rising along +x (negative: falling along +x)."""

    kind = "slope"

    def __init__(self, cfg):
        self.cfg = cfg
        self.slope_deg = float(cfg["slope_deg"])
        self.tan = math.tan(math.radians(self.slope_deg))

    def normal(self):
        a = math.radians(self.slope_deg)
        return (-math.sin(a), 0.0, math.cos(a))  # Genesis Plane(normal=...): z = x tan(a)

    def height_np(self, xy):
        return np.asarray(xy)[..., 0] * self.tan

    def gradient_np(self, xy):
        g = np.zeros(np.shape(xy))
        g[..., 0] = self.tan
        return g

    def height(self, xy):
        return xy[..., 0] * self.tan

    def gradient(self, xy):
        g = xy * 0.0
        g[..., 0] = self.tan
        return g

    def spawn_height(self, xy):
        return (xy[..., 0] + FOOTPRINT_HALF * math.copysign(1.0, self.tan)) * self.tan  # uphill edge of the footprint

    def describe(self):
        return f"plane at {self.slope_deg:+.1f} deg (rising along +x)" if self.slope_deg else "flat plane"


class TerrainGrid(FlatGround):
    """The curriculum grid: num_levels rows (difficulty) x one column per type and copy, plus a flat border."""

    kind = "grid"

    def __init__(self, cfg=None, height_field=None):
        c = self.cfg = terrain_config({"mode": "grid", **(cfg or {})})  # {} or None: the defaults
        if c["mode"] != "grid":
            raise ValueError("TerrainGrid needs a terrain config in mode 'grid'")
        self.types = list(c["types"])
        shares = np.array([c["types"][t] for t in self.types], dtype=np.float64)
        copies = int(c["copies"])
        self.col_type = np.repeat(np.arange(len(self.types)), copies)  # type index of each column
        self.col_share = np.repeat(shares / shares.sum() / copies, copies)  # share of the envs per column
        self.num_levels, self.num_cols = int(c["num_levels"]), len(self.col_type)
        self.hs, self.size = float(c["horizontal_scale"]), float(c["tile_size"])
        self.n = int(round(self.size / self.hs))
        self.b = int(round(c["border"] / self.hs))
        self.origin = np.array([-self.b * self.hs, -self.b * self.hs])  # world xy of vertex (0, 0)
        self.key = cache_key(c)
        self.height_field = (self._generate() if height_field is None else np.asarray(height_field)).astype(np.float32)
        expected = (2 * self.b + self.num_levels * self.n + 1, 2 * self.b + self.num_cols * self.n + 1)
        if self.height_field.shape != expected:
            raise ValueError(f"height field {self.height_field.shape} does not match the config {expected}")
        # tile centres (world) and the steepest slope each tile has by construction
        lv, co = np.meshgrid(np.arange(self.num_levels), np.arange(self.num_cols), indexing="ij")
        xy = np.stack(((lv + 0.5) * self.size, (co + 0.5) * self.size), axis=-1)
        self.tile_origins = np.concatenate((xy, self.height_np(xy)[..., None]), axis=-1)  # [L, C, 3]
        self.tile_slope = np.array([[tile_slope_deg(self.types[self.col_type[cc]], self.difficulty(lvl), c)
                                     for cc in range(self.num_cols)] for lvl in range(self.num_levels)])
        self.tile_tilt = np.array([[tile_tilt_deg(self.types[self.col_type[cc]], self.difficulty(lvl), c)
                                    for cc in range(self.num_cols)] for lvl in range(self.num_levels)])
        self._torch = None

    def difficulty(self, level):
        return level / max(self.num_levels - 1, 1)

    def _generate(self):
        c = self.cfg
        hf = np.zeros((2 * self.b + self.num_levels * self.n + 1, 2 * self.b + self.num_cols * self.n + 1))
        for lvl in range(self.num_levels):
            for col in range(self.num_cols):
                rng = np.random.default_rng([int(c["seed"]), lvl, col])  # one stream per tile
                tile = make_tile(self.types[self.col_type[col]], self.difficulty(lvl), c, rng)
                r0, c0 = self.b + lvl * self.n, self.b + col * self.n
                hf[r0:r0 + self.n + 1, c0:c0 + self.n + 1] = tile
        return hf

    # --- files
    def save(self, path):
        cfg = {k: v for k, v in self.cfg.items() if k != "file"}
        np.savez_compressed(path, height_field=self.height_field, cfg=json.dumps(cfg, default=list),
                            generator_version=GENERATOR_VERSION)
        return path

    @classmethod
    def load(cls, path, cfg=None):
        """A saved grid. cfg (optional) replaces the spawning and curriculum keys, never the heights' keys."""
        d = np.load(path, allow_pickle=False)
        saved = json.loads(str(d["cfg"]))
        if cfg:
            saved.update({k: v for k, v in cfg.items() if k not in GENERATION_KEYS and k != "file"})
        return cls(saved, height_field=d["height_field"])

    @classmethod
    def cached(cls, cfg, cache_dir=None):
        """Generate once per config, then load from cache_dir (the role of Genesis' Terrain(name=...) cache)."""
        c = terrain_config(cfg)
        cache_dir = os.path.expanduser(cache_dir or os.environ.get("GO2_TERRAIN_CACHE", "~/.cache/go2_terrain"))
        path = os.path.join(cache_dir, f"go2_terrain_{cache_key(c)}.npz")
        if os.path.isfile(path):
            return cls.load(path, c)
        grid = cls(c)
        try:
            os.makedirs(cache_dir, exist_ok=True)
            grid.save(path)
        except OSError:
            pass  # read-only home: generating each time is fast enough
        return grid

    # --- heights
    def to(self, device):
        self._torch = _TorchHeightField(self.height_field, self.origin, self.hs, device)
        return self

    def height_np(self, xy):
        return hf_height_np(self.height_field, self.origin, self.hs, xy)

    def gradient_np(self, xy):
        return hf_gradient_np(self.height_field, self.origin, self.hs, xy)

    def height(self, xy):
        return self._torch.height(xy)

    def gradient(self, xy):
        return self._torch.gradient(xy)

    def spawn_height(self, xy):
        """Highest ground under the footprint square around each xy [N, 2] (torch)."""
        offsets = self._torch.torch.as_tensor(_footprint(FOOTPRINT_HALF), dtype=xy.dtype, device=xy.device)
        return self.height(xy[:, None, :] + offsets[None]).amax(dim=1)

    def column_of_type(self, kind, copy_idx=0):
        cols = np.flatnonzero(self.col_type == self.types.index(kind))
        return int(cols[copy_idx % len(cols)])

    def describe(self):
        x1, y1 = self.num_levels * self.size, self.num_cols * self.size
        return (f"terrain grid {self.num_levels} levels x {self.num_cols} columns of {self.size:g} m "
                f"({x1:g} x {y1:g} m + {self.cfg['border']:g} m border, {self.height_field.shape[0]} x "
                f"{self.height_field.shape[1]} vertices at {self.hs:g} m), types "
                + ", ".join(f"{t} {100 * s:.0f}%" for t, s in zip(self.types, self.col_share.reshape(
                    len(self.types), -1).sum(1)))
                + f", hill/pit faces up to {self.cfg['max_slope_deg']:g} deg, seed {self.cfg['seed']}")

    def tile_stats(self):
        """Per tile: measured steepest slope [deg] (95th percentile of the triangle gradients), height min / max."""
        stats = np.zeros((self.num_levels, self.num_cols, 3))
        hf = self.height_field
        for lvl in range(self.num_levels):
            for col in range(self.num_cols):
                r0, c0 = self.b + lvl * self.n, self.b + col * self.n
                t = hf[r0:r0 + self.n + 1, c0:c0 + self.n + 1].astype(np.float64)
                gx = np.diff(t, axis=0)[:, :-1] / self.hs  # lower triangle of each cell
                gy = np.diff(t, axis=1)[:-1, :] / self.hs
                slope = np.degrees(np.arctan(np.hypot(gx, gy)))
                stats[lvl, col] = (np.percentile(slope, 95), t.min(), t.max())
        return stats


def make_ground(terrain_cfg, run_dir=None):
    """The ground of an env config: FlatGround (no terrain), PlaneGround (mode "slope") or TerrainGrid (mode "grid").
    A grid comes from terrain_cfg["file"] when given, else from run_dir/terrain.npz when it exists (the exact grid
    a run trained on), else from the cache / generator."""
    c = terrain_config(terrain_cfg)
    if c is None:
        return FlatGround()
    if c["mode"] == "slope":
        return PlaneGround(c)
    path = c.get("file") or (os.path.join(run_dir, "terrain.npz") if run_dir else None)
    if path and os.path.isfile(path):
        return TerrainGrid.load(path, c)
    return TerrainGrid.cached(c)


# ======================================================================================================================
# Preview and self-checks
# ======================================================================================================================
def plot_grid(grid, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    ink, muted, hairline = "#0b0b0b", "#52514e", "#e1e0d9"
    # diverging blue <-> red around a gray 0 (reference palette): pits blue, hills red
    cmap = LinearSegmentedColormap.from_list("ground", ["#184f95", "#5598e7", "#f0efec", "#e87473", "#a83232"])
    hf = grid.height_field
    x0, y0 = grid.origin
    x1, y1 = x0 + (hf.shape[0] - 1) * grid.hs, y0 + (hf.shape[1] - 1) * grid.hs
    lim = max(float(np.abs(hf).max()), 0.05)
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * grid.num_cols + 2.6))
    im = ax.imshow(hf.T, origin="lower", extent=(x0, x1, y0, y1), cmap=cmap, norm=TwoSlopeNorm(0.0, -lim, lim),
                   interpolation="nearest")
    for k in range(grid.num_levels + 1):
        ax.axvline(k * grid.size, color=hairline, linewidth=0.6)
    for k in range(grid.num_cols + 1):
        ax.axhline(k * grid.size, color=hairline, linewidth=0.6)
    ax.set_xticks((np.arange(grid.num_levels) + 0.5) * grid.size, [str(k) for k in range(grid.num_levels)])
    ax.set_yticks((np.arange(grid.num_cols) + 0.5) * grid.size, [grid.types[t] for t in grid.col_type])
    ax.set_xlim(-0.25 * grid.size, (grid.num_levels + 0.25) * grid.size)
    ax.set_ylim(-0.25 * grid.size, (grid.num_cols + 0.25) * grid.size)
    ax.set_xlabel("difficulty level (tile rows along x)", fontsize=8, color=ink)
    ax.tick_params(labelsize=7, colors=muted)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("ground height [m]", fontsize=8, color=ink)
    cb.ax.tick_params(labelsize=7, colors=muted)
    fig.suptitle(grid.describe() + ". One colour scale: rough and obstacle relief (a few cm) barely shows, see the "
                 "printed table", fontsize=8, color=ink, x=0.01, ha="left", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def print_table(grid):
    stats = grid.tile_stats()
    print(grid.describe())
    print("steepest slope [deg] by construction / measured (95th percentile), and height range [m], per tile:")
    head = f"{'level':>5}  " + "  ".join(f"{grid.types[t]:>17s}" for t in grid.col_type)
    print(head)
    for lvl in range(grid.num_levels):
        cells = []
        for col in range(grid.num_cols):
            s, lo, hi = stats[lvl, col]
            cells.append(f"{grid.tile_slope[lvl, col]:4.1f}/{s:4.1f} {lo:+.2f}..{hi:+.2f}")
        print(f"{lvl:>5}  " + "  ".join(f"{c:>17s}" for c in cells))


def self_check(genesis=False):
    """Checks that need no simulator (and with genesis=True, the lookup against Genesis' contact surface)."""
    import torch

    cfg = terrain_config({"num_levels": 5, "border": 4.0})
    grid = TerrainGrid(cfg)
    rng = np.random.default_rng(0)
    hf = grid.height_field
    # 1) the lookup returns the vertex heights at the vertices
    ii = rng.integers(0, hf.shape[0], 200)
    jj = rng.integers(0, hf.shape[1], 200)
    xy = np.stack((grid.origin[0] + ii * grid.hs, grid.origin[1] + jj * grid.hs), axis=-1)
    assert np.abs(grid.height_np(xy) - hf[ii, jj]).max() < 1e-5
    print("ok   lookup = vertex heights at the vertices")
    # 2) torch = numpy anywhere, gradient = finite differences away from the cell diagonals
    pts = np.stack((rng.uniform(-3, grid.num_levels * grid.size + 3, 2000),
                    rng.uniform(-3, grid.num_cols * grid.size + 3, 2000)), axis=-1)
    grid.to("cpu")
    ht = grid.height(torch.as_tensor(pts, dtype=torch.float32)).numpy()
    assert np.abs(ht - grid.height_np(pts)).max() < 1e-4, np.abs(ht - grid.height_np(pts)).max()
    g = (pts - grid.origin) / grid.hs
    frac = g - np.floor(g)
    away = np.abs(frac.sum(-1) - 1.0) > 0.05
    away &= (frac > 0.05).all(-1) & (frac < 0.95).all(-1)
    e = 1e-4
    fd = np.stack(((grid.height_np(pts + [e, 0]) - grid.height_np(pts - [e, 0])) / (2 * e),
                   (grid.height_np(pts + [0, e]) - grid.height_np(pts - [0, e])) / (2 * e)), axis=-1)
    assert np.abs(fd - grid.gradient_np(pts))[away].max() < 1e-3
    print("ok   torch lookup = numpy lookup; gradient = finite differences")
    # 3) hill and pit faces have the nominal slope, and seamless types are 0 on their edges
    stats = grid.tile_stats()
    for col, t in enumerate(grid.col_type):
        kind = grid.types[t]
        for lvl in range(grid.num_levels):
            r0, c0 = grid.b + lvl * grid.n, grid.b + col * grid.n
            tile = hf[r0:r0 + grid.n + 1, c0:c0 + grid.n + 1]
            edges = np.concatenate((tile[0], tile[-1], tile[:, 0], tile[:, -1]))
            assert np.abs(edges).max() < 1e-6, (kind, lvl)
            if kind in ("hill", "pit") and grid.tile_slope[lvl, col] > 0:
                assert abs(stats[lvl, col, 0] - grid.tile_slope[lvl, col]) < 0.3, (kind, lvl, stats[lvl, col, 0])
    top = grid.tile_slope[-1, grid.column_of_type("hill")]
    print(f"ok   hill/pit faces at d x max_slope_deg (top level {top:.1f} deg); every tile edge at 0")
    # 4) the cache returns the same heights; a saved grid loads back
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        a = TerrainGrid.cached(cfg, d)
        b = TerrainGrid.cached(cfg, d)
        assert np.array_equal(a.height_field, b.height_field) and np.array_equal(a.height_field, hf)
        c = TerrainGrid.load(a.save(os.path.join(d, "t.npz")), {"spawn_xy_noise": 0.0})
        assert np.array_equal(c.height_field, hf) and c.cfg["spawn_xy_noise"] == 0.0
    print("ok   cache and save/load give the same heights")
    plane = PlaneGround(terrain_config({"mode": "slope", "slope_deg": 20.0}))
    assert abs(plane.height_np(np.array([1.0, 3.0])) - math.tan(math.radians(20.0))) < 1e-9
    print("ok   tilted plane: z = x tan(slope)")
    # 5) [P3.2] contact-point slip on the 20 deg plane: 0 when rolling or sinking, the speed when sliding along it;
    #    on flat ground the horizontal formula of the P1 metric; normals agree with Genesis' Plane(normal=...)
    n = plane.normal_np(np.zeros((1, 2)))[0]
    assert np.allclose(n, plane.normal()), (n, plane.normal())
    t = np.array([math.cos(math.radians(20.0)), 0.0, math.sin(math.radians(20.0))])  # uphill, along the surface
    s, R = 0.3, 0.023
    roll = contact_slip_speed(s * t, (s / R) * np.cross(n, t), n, R)
    slide = contact_slip_speed(s * t, np.zeros(3), n, R)
    sink = contact_slip_speed(-0.1 * n, np.zeros(3), n, R)
    v, w = rng.normal(size=(100, 3)), rng.normal(size=(100, 3))
    flat = contact_slip_speed(v, w, FlatGround().normal_np(np.zeros((100, 2))), R)
    p1 = np.hypot(v[:, 0] - R * w[:, 1], v[:, 1] + R * w[:, 0])
    assert abs(roll) < 1e-12 and abs(slide - s) < 1e-12 and abs(sink) < 1e-12 and np.abs(flat - p1).max() < 1e-12
    n100 = plane.normal_np(np.zeros((100, 2)))
    tt = contact_slip_speed(torch.as_tensor(v), torch.as_tensor(w), torch.as_tensor(n100), R)
    assert np.abs(tt.numpy() - contact_slip_speed(v, w, n100, R)).max() < 1e-12
    print("ok   [P3.2] contact slip: 0 rolling or sinking on a 20 deg face, the sliding speed when sliding; "
          "= the P1 horizontal formula on flat ground; torch = numpy")
    if genesis:
        genesis_check(grid)


def genesis_check(grid):
    """The lookup against Genesis: TerrainEntity.get_terrain_height, and spheres dropped onto the terrain."""
    import torch

    import genesis as gs

    from go2_env import TERRAIN_FAST_BUILD, skip_terrain_support_field  # build the terrain as go2_env.py does

    gs.init(backend=gs.cpu, logging_level="warning")
    if TERRAIN_FAST_BUILD:
        skip_terrain_support_field()
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.005), show_viewer=False)
    vs = grid.cfg["vertical_scale"]
    terrain = scene.add_entity(gs.morphs.Terrain(height_field=grid.height_field / vs, horizontal_scale=grid.hs,
                                                 vertical_scale=vs, pos=(*grid.origin, 0.0)))
    rng = np.random.default_rng(1)
    # [P3.2] spheres dropped at the centroid of a heightfield triangle (away from its edges), mostly on sloped faces,
    # so that the solver's contact normal at touchdown can be compared with the lookup's normal there
    cells = rng.integers([grid.b, grid.b], [grid.b + grid.num_levels * grid.n - 1, grid.b + grid.num_cols * grid.n - 1],
                         size=(4000, 2))
    centroid = grid.origin + (cells + 1.0 / 3.0) * grid.hs
    steep = np.degrees(np.arctan(np.linalg.norm(grid.gradient_np(centroid), axis=-1))) > 5.0
    pts = np.concatenate((centroid[steep][:8], centroid[~steep][:4]))
    r = 0.023
    balls = [scene.add_entity(gs.morphs.Sphere(radius=r, pos=(x, y, grid.height_np(np.array([x, y])) + 0.05)))
             for x, y in pts]
    scene.build()
    q = rng.uniform([0, 0], [grid.num_levels * grid.size, grid.num_cols * grid.size], size=(500, 2))
    hg = terrain.get_terrain_height(torch.as_tensor(q, dtype=torch.float32)).cpu().numpy()
    err = np.abs(hg - grid.height_np(q)).max()
    print(f"ok   lookup vs Genesis get_terrain_height, 500 points: max difference {1e3 * err:.3f} mm")
    assert err < 1e-3
    first_normal = [None] * len(balls)  # [P3.2] (angle to the lookup normal [deg], slope there [deg]) at touchdown
    for _ in range(300):
        scene.step()
        for k, ball in enumerate(balls):
            if first_normal[k] is not None:
                continue
            c = ball.get_contacts(with_entity=terrain)
            if len(c["normal"]) == 0:
                continue
            p = ball.get_pos().cpu().numpy()
            n = grid.normal_np(p[:2])
            cos = np.abs(c["normal"].cpu().numpy().reshape(-1, 3) @ n).max()  # sign: the solver's a -> b convention
            first_normal[k] = (np.degrees(np.arccos(min(cos, 1.0))), np.degrees(np.arccos(n[2])))
    rest = []
    for ball in balls[-4:]:  # the spheres dropped on faces below 5 deg (on steeper ones they roll away)
        p = ball.get_pos().cpu().numpy()
        n = grid.normal_np(p[:2])
        rest.append((p[2] - grid.height_np(p[:2])) * n[2] - r)  # gap along the normal, minus the radius
    print(f"     spheres at rest (faces below 5 deg): gap to the surface {1e3 * np.min(rest):+.1f} .. "
          f"{1e3 * np.max(rest):+.1f} mm (contact penetration of the soft solver; Genesis' default plane gives the "
          "same order)")
    seen = [x for x in first_normal if x is not None]
    worst = max(a for a, _ in seen)
    print(f"ok   [P3.2] contact normal at touchdown vs the lookup's normal (go2_env foot slip, go2_metrics slip_v): "
          f"max {worst:.2f} deg over {len(seen)} spheres, on faces of "
          f"{min(s for _, s in seen):.1f} .. {max(s for _, s in seen):.1f} deg")
    assert len(seen) == len(balls) and worst < 1.0, first_normal


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE", help="terrain keys, values as Python literals")
    p.add_argument("--run", type=str, default=None, help="a run folder: show the terrain.npz it trained on")
    p.add_argument("--png", type=str, default="terrain_grid.png")
    p.add_argument("--check", action="store_true", help="self-checks (numpy)")
    p.add_argument("--genesis", action="store_true", help="with --check: compare with Genesis (CPU)")
    args = p.parse_args()
    if args.check:
        self_check(genesis=args.genesis)
        return
    if args.run:
        grid = TerrainGrid.load(os.path.join(args.run, "terrain.npz"))
    else:
        cfg = {}
        for item in args.set:
            key, _, text = item.partition("=")
            try:
                cfg[key] = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                cfg[key] = text
        grid = TerrainGrid(cfg)
    print_table(grid)
    print(f"height map saved to {plot_grid(grid, args.png)}")


if __name__ == "__main__":
    main()
