"""Das Labyrinth und seine Täuschungsmaße.

Raster `grid[i, j]` (True = Wand). Der Roboter startet unten links, das Ziel liegt oben links; dazwischen steht eine Querwand mit einer Öffnung. Je weiter die Öffnung vom Start-Ziel-Weg entfernt liegt, desto
länger der Umweg, der zunächst **vom Ziel wegführt**. Eine Suche, die nur den Abstand zum Ziel kennt (die Fitness), läuft an die Wand unter dem Ziel und bleibt dort stehen: ein lokales Optimum.

Täuschungsmaße (alle exakt, ohne Suche): kürzester Weg (Breitensuche) gegen Luftlinie (**Umwegfaktor**), **Fitness-Distanz-Korrelation** (Jones und Forrest 1995: Korrelation zwischen Luftlinie zum Ziel und
wahrem Weg zum Ziel über alle freien Zellen) und die **lokalen Minima der Fitness** (freie Zellen, deren Luftlinie zum Ziel kleiner ist als die aller freien Nachbarzellen: dort bleibt die Suche hängen)."""

from collections import deque

import numpy as np

import aur_constants as C


def make_maze(level):
    """Labyrinth der Stufe `level`: 0 freies Feld; 1 bis 3 eine Querwand (Zeile j = 12) mit Öffnung links (Spalten 3 bis 5), in der Mitte (11 bis 13), rechts (19 bis 21); 4 zwei Wände (j = 7 mit Öffnung rechts,
    j = 12 mit Öffnung links, j = 17 mit Öffnung rechts): eine Schlangenlinie."""
    n = C.N
    g = np.zeros((n, n), bool)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = True
    if level in (1, 2, 3):
        g[1:n - 1, 12] = True
        gap = {1: 3, 2: 11, 3: 19}[level]
        g[gap:gap + C.GAP, 12] = False
    elif level == 4:
        for j in (7, 12, 17):
            g[1:n - 1, j] = True
        g[19:19 + C.GAP, 7] = False
        g[3:3 + C.GAP, 12] = False
        g[19:19 + C.GAP, 17] = False
    elif level != 0:
        raise ValueError(f"unbekannte Stufe: {level}")
    return g


def cell_of(point):
    return int(point[0]), int(point[1])


def bfs_distances(grid, target):
    """Wahre Weglänge (in Zellen, 4er-Nachbarschaft) von jeder freien Zelle zur Zielzelle `target`; unerreichbare und belegte Zellen = unendlich."""
    n = grid.shape[0]
    dist = np.full(grid.shape, np.inf)
    dist[target] = 0.0
    q = deque([target])
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < n and 0 <= b < n and not grid[a, b] and dist[a, b] == np.inf:
                dist[a, b] = dist[i, j] + 1
                q.append((a, b))
    return dist


def shortest_path(grid, start=None, goal=None):
    """Kürzester Weg als Liste von Zellen von der Start- zur Zielzelle (Breitensuche), None ohne Weg."""
    start = cell_of(C.START) if start is None else start
    goal = cell_of(C.GOAL) if goal is None else goal
    dist = bfs_distances(grid, goal)
    if dist[start] == np.inf:
        return None
    path, cur = [start], start
    while cur != goal:
        i, j = cur
        cur = min(((i + di, j + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))), key=lambda c: dist[c])
        path.append(cur)
    return path


def euclid_field(grid, goal=None):
    """Luftlinie jeder Zelle (Mittelpunkt) zum Ziel; das ist, was die Fitness „sieht“."""
    goal = C.GOAL if goal is None else goal
    xs = np.arange(grid.shape[0]) + 0.5
    return np.hypot(xs[:, None] - goal[0], xs[None, :] - goal[1])


def detour_ratio(grid):
    """Kürzester Weg (Zellen) geteilt durch die Luftlinie von Start zu Ziel (Mittelpunkte)."""
    path = shortest_path(grid)
    return (len(path) - 1) / float(np.hypot(C.START[0] - C.GOAL[0], C.START[1] - C.GOAL[1]))


def fitness_distance_correlation(grid):
    """Korrelation (Pearson) zwischen Luftlinie zum Ziel und wahrem Weg zum Ziel über alle freien, erreichbaren Zellen. 1 = Fitness führt zum Ziel, nahe 0 oder negativ = täuschend."""
    true = bfs_distances(grid, cell_of(C.GOAL))
    mask = np.isfinite(true) & ~grid
    return float(np.corrcoef(euclid_field(grid)[mask], true[mask])[0, 1])


def local_minima(grid):
    """Lokale Minima der Fitness: freie, vom Start erreichbare Zellen, deren Luftlinie zum Ziel höchstens so groß ist wie die ihrer freien Nachbarzellen (8er-Nachbarschaft) – außer der Zielzelle.
    Gibt eine Liste (Zelle, Luftlinie) zurück, nach Luftlinie sortiert."""
    n = grid.shape[0]
    reach = np.isfinite(bfs_distances(grid, cell_of(C.START)))
    eu = euclid_field(grid)
    goal = cell_of(C.GOAL)
    out = []
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if grid[i, j] or not reach[i, j] or (i, j) == goal:
                continue
            better = [eu[i + di, j + dj] < eu[i, j] - 1e-12 for di in (-1, 0, 1) for dj in (-1, 0, 1)
                      if (di or dj) and not grid[i + di, j + dj]]
            if not any(better):
                out.append(((i, j), float(eu[i, j])))
    return sorted(out, key=lambda x: x[1])


def trap_distance(grid):
    """Luftlinie des tiefsten lokalen Minimums ohne Ziel (None bei freiem Feld): so nah läuft die Fitness-Suche ans Ziel, ohne es zu erreichen."""
    minima = local_minima(grid)
    return minima[0][1] if minima else None
