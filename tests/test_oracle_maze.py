"""Orakel-Tests: unabhängige Gegenproben für die beiden Kerne (Breitensuche und Kinematik).

* Breitensuche gegen SciPys `csgraph.shortest_path` auf dem Zellengraphen (ein anderer Algorithmus und eine andere Implementierung).
* Die vektorisierte Populations-Simulation gegen eine skalare Neuimplementierung, die nur aus der Beschreibung der Kinematik geschrieben ist (Schleife je Individuum und Schritt, `math` statt NumPy).
* Fitness-Distanz-Korrelation gegen eine Rechnung mit den SciPy-Weglängen."""

import math

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

import aur_constants as C
import aur_maze as M
import aur_robot as R


def scipy_distances(grid, target):
    """Weglängen zur Zielzelle über SciPy: Graph aus freien Zellen mit 4er-Kanten, Gewicht 1."""
    n = grid.shape[0]
    idx = lambda i, j: i * n + j
    rows, cols = [], []
    for i in range(n):
        for j in range(n):
            if grid[i, j]:
                continue
            for di, dj in ((1, 0), (0, 1)):
                a, b = i + di, j + dj
                if a < n and b < n and not grid[a, b]:
                    rows += [idx(i, j), idx(a, b)]
                    cols += [idx(a, b), idx(i, j)]
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n * n, n * n)).tocsr()
    dist = shortest_path(graph, method="D", unweighted=True, indices=idx(*target))
    return dist.reshape(n, n)


@pytest.mark.parametrize("level", list(C.LEVELS))
def test_breadth_first_search_agrees_with_scipy_on_every_free_cell(level):
    grid = M.make_maze(level)
    mine = M.bfs_distances(grid, M.cell_of(C.GOAL))
    ref = scipy_distances(grid, M.cell_of(C.GOAL))
    free = ~grid
    assert np.array_equal(np.isfinite(mine[free]), np.isfinite(ref[free]))
    ok = free & np.isfinite(ref)
    assert mine[ok] == pytest.approx(ref[ok], abs=0)


@pytest.mark.parametrize("level", list(C.LEVELS))
def test_shortest_path_length_agrees_with_scipy(level):
    grid = M.make_maze(level)
    ref = scipy_distances(grid, M.cell_of(C.GOAL))[M.cell_of(C.START)]
    assert len(M.shortest_path(grid)) - 1 == int(ref)


@pytest.mark.parametrize("level", list(C.LEVELS))
def test_fitness_distance_correlation_agrees_with_a_scipy_based_computation(level):
    grid = M.make_maze(level)
    ref = scipy_distances(grid, M.cell_of(C.GOAL))
    mask = np.isfinite(ref) & ~grid
    xs = np.arange(C.N) + 0.5
    eu = np.array([[math.hypot(xs[i] - C.GOAL[0], xs[j] - C.GOAL[1]) for j in range(C.N)] for i in range(C.N)])
    assert M.fitness_distance_correlation(grid) == pytest.approx(np.corrcoef(eu[mask], ref[mask])[0, 1], abs=1e-12)


def scalar_run(genome, grid):
    """Skalare Referenz-Kinematik: eine Schleife je Schritt, nur `math`; gleiche Regeln wie in der Beschreibung (Drehung, Fahrt, Stillstand an belegten Zellen und am Rand)."""
    n = grid.shape[0]
    x, y = C.START
    heading = C.START_HEADING
    for turn, speed in genome:
        heading += turn * C.TURN_MAX
        nx = x + speed * C.SPEED_MAX * math.cos(heading)
        ny = y + speed * C.SPEED_MAX * math.sin(heading)
        inside = 0 <= nx < n and 0 <= ny < n
        if inside and not grid[int(nx), int(ny)]:
            x, y = nx, ny
    return x, y


@pytest.mark.parametrize("level", [0, 3, 4])
def test_vectorised_simulation_agrees_with_the_scalar_reference(level):
    grid = M.make_maze(level)
    genomes = R.random_genomes(np.random.default_rng(100 + level), 25)
    fast = R.simulate(genomes, grid)
    slow = np.array([scalar_run(g, grid) for g in genomes])
    assert fast == pytest.approx(slow, abs=1e-9)


def test_the_path_controller_agrees_with_the_scalar_reference_too():
    for level in C.LEVELS:
        grid = M.make_maze(level)
        genome = R.controller_from_path(M.shortest_path(grid))
        x, y = scalar_run(genome, grid)
        assert math.hypot(x - C.GOAL[0], y - C.GOAL[1]) < 0.5
