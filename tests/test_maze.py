"""Labyrinth und Täuschungsmaße: Aufbau der Stufen, Breitensuche und kürzester Weg von Hand, Luftlinie, Umwegfaktor, Fitness-Distanz-Korrelation, lokale Minima."""

import numpy as np
import pytest

import aur_constants as C
import aur_maze as M


def hand_grid():
    """5 × 5, Rand und die Zelle (2, 2) belegt: innen 3 × 3 mit einem Pfosten in der Mitte."""
    g = np.zeros((5, 5), bool)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = True
    g[2, 2] = True
    return g


def test_every_maze_has_a_wall_border_and_a_free_start_and_goal():
    for level in C.LEVELS:
        g = M.make_maze(level)
        assert g.shape == (C.N, C.N) and g[0].all() and g[-1].all() and g[:, 0].all() and g[:, -1].all()
        assert not g[M.cell_of(C.START)] and not g[M.cell_of(C.GOAL)]


def test_level_structure_by_hand():
    assert M.make_maze(0).sum() == 4 * C.N - 4                                              # nur der Rand
    g3 = M.make_maze(3)
    assert g3[1:C.N - 1, 12].sum() == (C.N - 2) - C.GAP and not g3[19:22, 12].any() and g3[18, 12] and g3[22, 12] and g3[3, 12]
    g1 = M.make_maze(1)
    assert not g1[3:6, 12].any() and g1[6, 12]
    g4 = M.make_maze(4)
    assert not g4[19:22, 7].any() and not g4[3:6, 12].any() and not g4[19:22, 17].any() and g4[3, 7] and g4[12, 12] and g4[3, 17]
    for j in (7, 12, 17):
        assert g4[1:C.N - 1, j].sum() == (C.N - 2) - C.GAP


def test_unknown_level_is_rejected():
    with pytest.raises(ValueError):
        M.make_maze(7)


def test_bfs_distances_by_hand():
    """Ziel (1, 1): (1, 2) und (2, 1) = 1; (1, 3) und (3, 1) = 2; (2, 3) und (3, 2) = 3; (3, 3) = 4; der Pfosten bleibt unendlich."""
    d = M.bfs_distances(hand_grid(), (1, 1))
    assert d[1, 1] == 0 and d[1, 2] == d[2, 1] == 1 and d[1, 3] == d[3, 1] == 2 and d[2, 3] == d[3, 2] == 3 and d[3, 3] == 4 and d[2, 2] == np.inf and d[0, 0] == np.inf


def test_shortest_path_by_hand_and_no_path():
    g = hand_grid()
    path = M.shortest_path(g, start=(3, 3), goal=(1, 1))
    assert len(path) == 5 and path[0] == (3, 3) and path[-1] == (1, 1) and (2, 2) not in path
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(path, path[1:]))
    sealed = g.copy()
    sealed[1, 2] = sealed[2, 1] = True
    assert M.shortest_path(sealed, start=(3, 3), goal=(1, 1)) is None and M.bfs_distances(sealed, (1, 1))[3, 3] == np.inf


def test_shortest_path_lengths_of_the_levels():
    """Weg in Zellen: 17 (frei), 17 (Öffnung links, genau auf dem Weg), 33 (Mitte: 8 Zellen seitlich hin und zurück), 49 (rechts: 16 hin und zurück), 77 (drei Wände)."""
    lengths = [len(M.shortest_path(M.make_maze(lv))) - 1 for lv in C.LEVELS]
    assert lengths == [17, 17, 33, 49, 77]


def test_euclid_field_by_hand():
    eu = M.euclid_field(M.make_maze(0))
    assert eu[3, 20] == pytest.approx(0.0)
    assert eu[6, 16] == pytest.approx(5.0)                                                  # Mitte (6.5, 16.5) zum Ziel (3.5, 20.5): 3-4-5
    assert eu[3, 11] == pytest.approx(9.0)


def test_detour_ratio_by_hand():
    assert M.detour_ratio(M.make_maze(0)) == pytest.approx(1.0) and M.detour_ratio(M.make_maze(3)) == pytest.approx(49 / 17)


def test_fitness_distance_correlation_is_one_in_a_straight_corridor():
    """Ein Gang entlang x = 3 (alles andere Wand): Luftlinie zum Ziel = Weg zum Ziel, Korrelation genau 1."""
    g = np.ones((C.N, C.N), bool)
    g[3, 1:C.N - 1] = False
    assert M.fitness_distance_correlation(g) == pytest.approx(1.0)


def test_fitness_distance_correlation_falls_with_the_detour():
    f = [M.fitness_distance_correlation(M.make_maze(lv)) for lv in C.LEVELS]
    assert f[0] == pytest.approx(f[1], abs=1e-3) and f[0] > 0.95 and f[0] > f[2] > f[3] and f[3] < 0.7 and f[4] < 0.7


def test_local_minima_of_the_levels():
    """Kein Minimum bei freiem Feld und bei Öffnung links; Stufe 2 und 3: Zelle (3, 11) unter der Wand, Luftlinie 9; Stufe 4: (3, 16) mit 4 und (3, 6) mit 14."""
    assert M.local_minima(M.make_maze(0)) == [] and M.local_minima(M.make_maze(1)) == []
    for lv in (2, 3):
        assert M.local_minima(M.make_maze(lv)) == [((3, 11), pytest.approx(9.0))]
    assert M.local_minima(M.make_maze(4)) == [((3, 16), pytest.approx(4.0)), ((3, 6), pytest.approx(14.0))]
    assert M.trap_distance(M.make_maze(0)) is None and M.trap_distance(M.make_maze(3)) == pytest.approx(9.0) and M.trap_distance(M.make_maze(4)) == pytest.approx(4.0)


def test_cell_of():
    assert M.cell_of((3.5, 20.5)) == (3, 20) and M.cell_of((0.99, 7.0)) == (0, 7)
