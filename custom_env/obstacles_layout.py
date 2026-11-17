"""Seeded layout generation, with geometric margins used only for sampling."""

import math
from collections import deque

import numpy as np


def obstacle_corners(o):
    if o["shape"] == "cylinder":
        r = o["width"] / 2
        return [(o["x"] + r * math.cos(i * math.pi / 2),
                 o["y"] + r * math.sin(i * math.pi / 2)) for i in range(4)]
    c, s = math.cos(o["angle"]), math.sin(o["angle"])
    return [(o["x"] + c*u - s*v, o["y"] + s*u + c*v)
            for u, v in [(-o["width"]/2, -o["depth"]/2), (o["width"]/2, -o["depth"]/2),
                         (o["width"]/2, o["depth"]/2), (-o["width"]/2, o["depth"]/2)]]


def obstacle_distance(point, o):
    """Unsigned distance from a point to the actual footprint, including its interior."""
    x, y = point[0] - o["x"], point[1] - o["y"]
    if o["shape"] == "cylinder":
        return max(0., math.hypot(x, y) - o["width"] / 2)
    c, s = math.cos(o["angle"]), math.sin(o["angle"])
    return math.hypot(max(abs(c*x + s*y) - o["width"]/2, 0),
                      max(abs(-s*x + c*y) - o["depth"]/2, 0))


def obstacle_gap(a, b):
    if a["shape"] == "cylinder":
        return max(0., obstacle_distance((a["x"], a["y"]), b) - a["width"]/2)
    if b["shape"] == "cylinder":
        return max(0., obstacle_distance((b["x"], b["y"]), a) - b["width"]/2)
    pa, pb = obstacle_corners(a), obstacle_corners(b)
    separated = False
    for polygon in (pa, pb):
        for i, p in enumerate(polygon):
            q = polygon[(i+1) % 4]
            axis = (p[1]-q[1], q[0]-p[0])
            av = [x*axis[0] + y*axis[1] for x, y in pa]
            bv = [x*axis[0] + y*axis[1] for x, y in pb]
            separated |= max(av) < min(bv) or max(bv) < min(av)
    if not separated:
        return 0.
    return min([obstacle_distance(p, b) for p in pa] +
               [obstacle_distance(p, a) for p in pb])


def reachable_grid(cfg, obstacles):
    """Conservative four-neighbour grid; also clear each cell's half diagonal."""
    limit = cfg.arena_size - 0.1 - cfg.path_clearance
    count = math.ceil(2 * limit / cfg.connectivity_grid_step) + 1
    axis = np.linspace(-limit, limit, count)
    spacing = float(axis[1] - axis[0])
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    safety = spacing / math.sqrt(2)
    free = np.ones(xx.shape, dtype=bool)
    for o in obstacles:
        x, y = xx-o["x"], yy-o["y"]
        if o["shape"] == "cylinder":
            dist = np.maximum(0, np.hypot(x, y) - o["width"]/2)
        else:
            c, s = math.cos(o["angle"]), math.sin(o["angle"])
            dist = np.hypot(np.maximum(abs(c*x+s*y)-o["width"]/2, 0),
                            np.maximum(abs(-s*x+c*y)-o["depth"]/2, 0))
        free &= dist > cfg.path_clearance + safety
    start = (int(np.argmin(abs(axis-cfg.spawn_x))), int(np.argmin(abs(axis-cfg.spawn_y))))
    reached = set()
    queue = deque([start]) if free[start] else deque()
    while queue:
        p = queue.popleft()
        if p in reached:
            continue
        reached.add(p)
        for q in [(p[0]+1,p[1]), (p[0]-1,p[1]), (p[0],p[1]+1), (p[0],p[1]-1)]:
            if 0 <= q[0] < count and 0 <= q[1] < count and free[q] and q not in reached:
                queue.append(q)
    return axis, free, reached


def generate_layout(cfg, seed):
    """Use a private RNG so layouts are independent of policy/training RNG state."""
    rng = np.random.default_rng(seed)
    shapes = [("cylinder", cfg.pillar_diameter, cfg.pillar_diameter)] * cfg.pillar_count
    shapes += [("wall", cfg.wall_length, cfg.wall_thickness)] * cfg.wall_count
    spawn = (cfg.spawn_x, cfg.spawn_y)
    resource_limit = cfg.arena_size - cfg.resource_boundary_gap
    if resource_limit <= 0:
        raise ValueError("resource_boundary_gap leaves no resource placement area.")
    for _ in range(cfg.layout_attempts):
        obstacles = []
        for shape, width, depth in shapes:
            for _ in range(cfg.placement_attempts):
                o = dict(shape=shape, width=width, depth=depth,
                         x=float(rng.uniform(-cfg.arena_size, cfg.arena_size)),
                         y=float(rng.uniform(-cfg.arena_size, cfg.arena_size)),
                         angle=float(rng.uniform(0, math.pi)) if shape == "wall" else 0.)
                if any(abs(v) > cfg.arena_size - 0.1 - cfg.boundary_obstacle_gap
                       for p in obstacle_corners(o) for v in p):
                    continue
                if obstacle_distance(spawn, o) < cfg.spawn_obstacle_gap:
                    continue
                if any(obstacle_gap(o, other) < cfg.obstacle_spacing for other in obstacles):
                    continue
                obstacles.append(o)
                break
        if len(obstacles) != len(shapes):
            continue
        axis, free, reached = reachable_grid(cfg, obstacles)
        if not reached or len(reached) != int(free.sum()):
            continue
        resources = []
        kinds = ["food"] * cfg.num_food + ["water"] * cfg.num_water
        for _ in range(cfg.resource_placement_attempts):
            p = tuple(map(float, rng.uniform(-resource_limit, resource_limit, size=2)))
            if math.dist(p, spawn) < cfg.spawn_resource_gap:
                continue
            if any(obstacle_distance(p, o) < cfg.resource_obstacle_gap for o in obstacles):
                continue
            if any(math.dist(p, r[1:]) < cfg.object_spacing for r in resources):
                continue
            cell = tuple(int(np.argmin(abs(axis-v))) for v in p)
            if cell not in reached:
                continue
            resources.append((kinds[len(resources)], *p))
            if len(resources) == len(kinds):
                return dict(seed=int(seed), spawn=[*spawn, cfg.spawn_yaw],
                            obstacles=obstacles, resources=resources,
                            reachable_grid_cells=len(reached))
    raise ValueError("Could not generate a reachable obstacles layout within layout_attempts; "
                     "reduce resource/obstacle counts, dimensions, or sampling margins.")
