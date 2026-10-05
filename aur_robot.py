"""Der Hof-Roboter: offene Steuerfolge (kein Sensor), Kinematik mit Kollision, Simulation ganzer Populationen auf einmal und eine Steuerfolge, die einem gegebenen Zellenpfad folgt (Lösbarkeitsnachweis).

**Kinematik.** Zustand (x, y, Richtung). Je Schritt t: Richtung += Drehgen_t · TURN_MAX, dann Fahrt um Tempogen_t · SPEED_MAX in die neue Richtung. Landet der neue Punkt in einer belegten Zelle (Wand oder Rand), bleibt der Roboter
stehen (die Richtung bleibt geändert). **Genom:** Feld (STEPS, 2), Spalte 0 Drehgen ∈ [−1, 1], Spalte 1 Tempogen ∈ [0, 1]. **Verhalten:** die Endposition."""

import math

import numpy as np

import aur_constants as C


def clip_genomes(g):
    """Gene in ihre Grenzen zwingen (Drehung [−1, 1], Tempo [0, 1])."""
    g[..., 0] = np.clip(g[..., 0], -1.0, 1.0)
    g[..., 1] = np.clip(g[..., 1], 0.0, 1.0)
    return g


def random_genomes(rng, count, steps=None):
    """Zufällige Steuerfolgen der Länge `steps` (Standard: C.STEPS): Drehgene gleichverteilt in [−1, 1], Tempogene in [0, 1]."""
    steps = C.STEPS if steps is None else steps
    g = np.empty((count, steps, 2))
    g[..., 0] = rng.uniform(-1.0, 1.0, (count, steps))
    g[..., 1] = rng.uniform(0.0, 1.0, (count, steps))
    return g


def simulate(genomes, grid, trajectories=False):
    """Lässt jede Steuerfolge (P, Schritte, 2) im Labyrinth `grid` fahren; die Länge der Steuerfolge ergibt sich aus der Form von `genomes`. Gibt die Endpositionen (P, 2) zurück; mit `trajectories=True` zusätzlich alle Positionen (P, Schritte + 1, 2)."""
    genomes = np.asarray(genomes, float)
    count = len(genomes)
    n = grid.shape[0]
    pos = np.tile(np.array(C.START, float), (count, 1))
    heading = np.full(count, C.START_HEADING)
    track = [pos.copy()] if trajectories else None
    for t in range(genomes.shape[1]):
        heading = heading + genomes[:, t, 0] * C.TURN_MAX
        step = (genomes[:, t, 1] * C.SPEED_MAX)[:, None] * np.column_stack([np.cos(heading), np.sin(heading)])
        new = pos + step
        idx = np.clip(new.astype(int), 0, n - 1)
        blocked = grid[idx[:, 0], idx[:, 1]] | (new[:, 0] < 0) | (new[:, 1] < 0) | (new[:, 0] >= n) | (new[:, 1] >= n)
        pos = np.where(blocked[:, None], pos, new)
        if trajectories:
            track.append(pos.copy())
    return (pos, np.stack(track, axis=1)) if trajectories else pos


def distance_to_goal(positions):
    """Luftlinie jeder Position zum Ziel."""
    return np.hypot(positions[:, 0] - C.GOAL[0], positions[:, 1] - C.GOAL[1])


def controller_from_path(path):
    """Steuerfolge, die einem Zellenpfad (Liste von Zellen) folgt: Ecken des Pfades als Zwischenziele (Zellmitten), vor jeder Strecke auf Kurs drehen (Tempo 0), dann in gleich langen Schritten fahren.
    Gibt das Genom (STEPS, 2) zurück oder wirft ValueError, wenn die Schritte nicht reichen. Beweist, dass die Aufgabe im Suchraum lösbar ist."""
    pts = [np.array([path[0][0] + 0.5, path[0][1] + 0.5])]
    for k in range(1, len(path) - 1):
        d1 = (path[k][0] - path[k - 1][0], path[k][1] - path[k - 1][1])
        d2 = (path[k + 1][0] - path[k][0], path[k + 1][1] - path[k][1])
        if d1 != d2:
            pts.append(np.array([path[k][0] + 0.5, path[k][1] + 0.5]))
    pts.append(np.array([path[-1][0] + 0.5, path[-1][1] + 0.5]))
    genome, heading, cur = [], C.START_HEADING, np.array(C.START, float)
    for target in pts[1:]:
        vec = target - cur
        want = math.atan2(vec[1], vec[0])
        delta = (want - heading + math.pi) % (2 * math.pi) - math.pi
        while abs(delta) > 1e-12:
            turn = max(-1.0, min(1.0, delta / C.TURN_MAX))
            genome.append((turn, 0.0))
            heading += turn * C.TURN_MAX
            delta = (want - heading + math.pi) % (2 * math.pi) - math.pi
            if abs(delta) < 1e-9:
                delta = 0.0
        length = float(np.hypot(*vec))
        moves = max(1, math.ceil(length / C.SPEED_MAX - 1e-12))
        genome.extend([(0.0, length / (moves * C.SPEED_MAX))] * moves)
        cur = target
    if len(genome) > C.STEPS:
        raise ValueError(f"Steuerfolge braucht {len(genome)} Schritte, erlaubt sind {C.STEPS}")
    genome.extend([(0.0, 0.0)] * (C.STEPS - len(genome)))
    return np.array(genome, float)
