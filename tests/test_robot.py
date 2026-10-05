"""Roboter: Kinematik und Kollision von Hand, Population gleich Einzelfahrten, Fahrten, Grenzen der Gene, Lösbarkeit über den Pfad-Regler."""

import math

import numpy as np
import pytest

import aur_constants as C
import aur_maze as M
import aur_robot as R


def zeros():
    return np.zeros((1, C.STEPS, 2))


def test_stationary_genome_stays_at_the_start():
    end = R.simulate(zeros(), M.make_maze(0))
    assert end[0] == pytest.approx(C.START)


def test_one_step_by_hand():
    """Erster Schritt mit Tempo 1 und ohne Drehung: Richtung π/4, Strecke 0.7: (3.5 + 0.7·cos π/4, 3.5 + 0.7·sin π/4); alle weiteren Schritte Tempo 0."""
    g = zeros()
    g[0, 0, 1] = 1.0
    end = R.simulate(g, M.make_maze(0))
    d = 0.7 * math.cos(math.pi / 4)
    assert end[0] == pytest.approx([3.5 + d, 3.5 + d])


def test_turning_changes_the_heading_by_turn_max():
    """Drehgen 1 in Schritt 1 (Richtung π/4 + 0.6), dann Tempo 1: Position = Start + 0.7·(cos, sin)(π/4 + 0.6)."""
    g = zeros()
    g[0, 0, 0] = 1.0
    g[0, 1, 1] = 1.0
    end = R.simulate(g, M.make_maze(0))
    h = math.pi / 4 + 0.6
    assert end[0] == pytest.approx([3.5 + 0.7 * math.cos(h), 3.5 + 0.7 * math.sin(h)])


def test_collision_stops_the_robot_at_the_wall():
    """Zwei Schritte auf Kurs π/2 drehen (Tempo 0), dann 20 Schritte Tempo 1 senkrecht nach oben bei x = 3.5: Die Wand von Stufe 3 liegt in Zeile 12 (y ab 12). Letzte erlaubte Position: y = 3.5 + 12·0.7 = 11.9."""
    g = zeros()
    g[0, 0, 0] = 1.0
    g[0, 1, 0] = (math.pi / 2 - (math.pi / 4 + 0.6)) / 0.6
    g[0, 2:22, 1] = 1.0
    end = R.simulate(g, M.make_maze(3))
    assert end[0] == pytest.approx([3.5, 11.9])
    free = R.simulate(g, M.make_maze(0))
    assert free[0, 1] == pytest.approx(3.5 + 20 * 0.7) and free[0, 0] == pytest.approx(3.5)


def test_robot_cannot_leave_the_arena():
    g = zeros()
    g[0, :, 1] = 1.0
    end = R.simulate(g, M.make_maze(0))
    assert (end > 1.0).all() and (end < C.N - 1).all()


def test_population_equals_single_runs():
    rng = np.random.default_rng(3)
    genomes = R.random_genomes(rng, 12)
    grid = M.make_maze(3)
    batch = R.simulate(genomes, grid)
    single = np.vstack([R.simulate(genomes[i:i + 1], grid) for i in range(12)])
    assert batch == pytest.approx(single, abs=1e-12)


def test_trajectories_start_at_the_start_and_end_at_the_end_position():
    genomes = R.random_genomes(np.random.default_rng(1), 5)
    grid = M.make_maze(2)
    end, track = R.simulate(genomes, grid, trajectories=True)
    assert track.shape == (5, C.STEPS + 1, 2) and (track[:, 0] == np.array(C.START)).all() and track[:, -1] == pytest.approx(end)
    steps = np.hypot(*np.diff(track, axis=1).transpose(2, 0, 1))
    assert steps.max() <= C.SPEED_MAX + 1e-9


def test_random_genomes_and_clipping():
    g = R.random_genomes(np.random.default_rng(0), 50)
    assert g.shape == (50, C.STEPS, 2) and g[..., 0].min() >= -1 and g[..., 0].max() <= 1 and g[..., 1].min() >= 0 and g[..., 1].max() <= 1
    wild = np.full((2, 3, 2), 5.0)
    wild[..., 0] = -5.0
    out = R.clip_genomes(wild)
    assert (out[..., 0] == -1).all() and (out[..., 1] == 1).all()


def test_distance_to_goal_by_hand():
    assert R.distance_to_goal(np.array([[3.5, 17.5], [3.5, 20.5], [6.5, 16.5]])) == pytest.approx([3.0, 0.0, 5.0])


@pytest.mark.parametrize("level", list(C.LEVELS))
def test_a_controller_from_the_shortest_path_reaches_the_goal(level):
    """Lösbarkeit: die Aufgabe liegt im Suchraum. Auch die Schlangenlinie braucht höchstens 136 der 140 Schritte."""
    g = M.make_maze(level)
    genome = R.controller_from_path(M.shortest_path(g))
    assert genome.shape == (C.STEPS, 2) and (np.abs(genome[:, 0]) <= 1 + 1e-12).all() and ((genome[:, 1] >= 0) & (genome[:, 1] <= 1 + 1e-12)).all()
    assert R.distance_to_goal(R.simulate(genome[None], g))[0] < 0.5


def test_the_controller_for_the_serpentine_uses_almost_all_steps():
    genome = R.controller_from_path(M.shortest_path(M.make_maze(4)))
    used = int((np.abs(genome).sum(axis=1) > 0).sum())
    assert used == 136 and used <= C.STEPS


def test_a_path_that_needs_more_steps_than_available_is_rejected():
    path = [(1, 1)]
    for k in range(60):
        path.append((path[-1][0] + (1 if k % 2 == 0 else 0), path[-1][1] + (0 if k % 2 == 0 else 1)))
    with pytest.raises(ValueError):
        R.controller_from_path(path)
