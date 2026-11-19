"""Seeded layout generation, with geometric margins used only for sampling."""

import heapq
import itertools
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


def _grid_shortest_distances(axis, free, source, goals):
    """Return 8-neighbour geodesic distances in metres from source to goals."""
    n = len(axis)
    step = float(axis[1] - axis[0])
    si = (int(np.argmin(abs(axis - source[0]))), int(np.argmin(abs(axis - source[1]))))
    goal_cells = [
        (int(np.argmin(abs(axis - g[0]))), int(np.argmin(abs(axis - g[1]))))
        for g in goals
    ]
    remaining = set(goal_cells)
    dist = np.full((n, n), math.inf, dtype=np.float64)
    dist[si] = 0.0
    pq = [(0.0, si[0], si[1])]
    diag = math.sqrt(2.0)
    nbrs = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, diag), (-1, 1, diag), (1, -1, diag), (1, 1, diag),
    )
    while pq and remaining:
        d, ux, uy = heapq.heappop(pq)
        if d > dist[ux, uy]:
            continue
        remaining.discard((ux, uy))
        if not remaining:
            break
        for dx, dy, w in nbrs:
            vx, vy = ux + dx, uy + dy
            if 0 <= vx < n and 0 <= vy < n and free[vx, vy]:
                if dx != 0 and dy != 0 and not (free[ux + dx, uy] and free[ux, uy + dy]):
                    continue
                nd = d + w
                if nd < dist[vx, vy]:
                    dist[vx, vy] = nd
                    heapq.heappush(pq, (nd, vx, vy))
    return [float(dist[g] * step) for g in goal_cells]


def tour_path_metrics(axis, free, spawn, resources):
    """Compute optimal obstacle-free tour distance and detour ratio."""
    points = [spawn] + [r[1:] for r in resources]
    k = len(resources)
    if k == 0:
        return 0.0, 0.0, 1.0
    geo = np.zeros((k + 1, k + 1), dtype=np.float64)
    for i in range(k + 1):
        if i + 1 <= k:
            dists = _grid_shortest_distances(axis, free, points[i], points[i + 1 :])
            for offset, d in enumerate(dists, start=1):
                j = i + offset
                geo[i, j] = geo[j, i] = d
    euc = np.zeros((k + 1, k + 1), dtype=np.float64)
    for i in range(k + 1):
        for j in range(i + 1, k + 1):
            d = math.dist(points[i], points[j])
            euc[i, j] = euc[j, i] = d

    if k <= 6:
        best_order = min(
            itertools.permutations(range(1, k + 1)),
            key=lambda order: geo[0, order[0]] + sum(
                geo[a, b] for a, b in zip(order, order[1:])
            ),
        )
    else:
        unvisited = set(range(1, k + 1))
        cur = 0
        best_order = []
        while unvisited:
            nxt = min(unvisited, key=lambda idx: geo[cur, idx])
            best_order.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
    best_geo = float(geo[0, best_order[0]] + sum(geo[a, b] for a, b in zip(best_order, best_order[1:])))
    best_euc = float(euc[0, best_order[0]] + sum(euc[a, b] for a, b in zip(best_order, best_order[1:])))
    detour_ratio = best_geo / max(best_euc, 1e-6)
    return best_geo, best_euc, float(detour_ratio)


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
                geo_dist, euc_dist, detour_ratio = tour_path_metrics(axis, free, spawn, resources)
                if obstacles and detour_ratio < cfg.min_detour_ratio:
                    break
                return dict(seed=int(seed), spawn=[*spawn, cfg.spawn_yaw],
                            obstacles=obstacles, resources=resources,
                            reachable_grid_cells=len(reached),
                            geodesic_shortest_distance=geo_dist,
                            euclidean_shortest_distance=euc_dist,
                            detour_ratio=detour_ratio)
    raise ValueError("Could not generate a reachable obstacles layout within layout_attempts; "
                     "reduce resource/obstacle counts, dimensions, or sampling margins.")
