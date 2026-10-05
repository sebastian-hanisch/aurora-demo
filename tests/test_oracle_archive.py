"""Orakel-Tests für Archiv und Läufe: unabhängige Gegenproben.

* **Direktheit, Merkmale, Endzellen und Wegzellen** gegen eine skalare Neuberechnung der Fahrt nur aus der Beschreibung der Kinematik (eine Schleife je Individuum und Schritt, `math` statt NumPy).
* **MAP-Elites mit handgemachter und einmal gelernter Beschreibung als Ganzes** gegen eine skalare Neuimplementierung der Buchführung mit Wörterbüchern (Zelle -> Qualität, Steuerfolge, Endposition, berührte Zellen), mit skalar berechneten Merkmalen und skalar berechnetem Code
  (Identität, PCA und Autoencoder in reinen Schleifen) und Brute-Force-Suche des nächsten Schwerpunkts; dieselben Zufallszahlen, dieselbe Reihenfolge.
* **Die AURORA-Schleife als Ganzes** (Aufwärmen, erstes Lernen, Neulernen, Neueintragen) gegen dieselbe Buchführung mit Wörterbüchern; das Lernen selbst (`fit_space`) ist in `test_learn.py` und `test_space.py` geprüft und wird hier gemeinsam benutzt.
* **Restweg** gegen SciPys Breitensuche."""

import math

import numpy as np
import pytest

import aur_constants as C
import aur_evaluation as E
import aur_maze as M
import aur_robot as R
import aur_search as S
import aur_space as SP
from test_oracle_maze import scipy_distances


def scalar_track(genome, grid):
    n = grid.shape[0]
    x, y = C.START
    heading = C.START_HEADING
    out = [(x, y)]
    for turn, speed in genome:
        heading += turn * C.TURN_MAX
        nx = x + speed * C.SPEED_MAX * math.cos(heading)
        ny = y + speed * C.SPEED_MAX * math.sin(heading)
        if 0 <= nx < n and 0 <= ny < n and not grid[int(nx), int(ny)]:
            x, y = nx, ny
        out.append((x, y))
    return out


def clip01(v):
    return min(max(v, 0.0), 1.0)


def scalar_features(variant, genome, pts):
    """Merkmale der Variante aus den Punkten der Fahrt, skalar."""
    steps = len(genome)
    if C.VARIANT_MODE[variant] != "hand":
        out = []
        for i in range(1, C.RAW_POINTS + 1):
            px, py = pts[int(steps * i / C.RAW_POINTS)]
            out += [clip01(px / C.N), clip01(py / C.N)]
        return out
    kind = C.VARIANT_METHOD[variant]
    if kind == "style":
        driven = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
        return [clip01(driven / C.STYLE_MAX_LENGTH), (sum(t for t, _ in genome) / steps + 1.0) / 2.0]
    parts = C.HAND_PARTS[kind]
    out = []
    for i in range(1, parts + 1):
        px, py = pts[int(steps * i / parts)]
        out += [clip01(px / C.N), clip01(py / C.N)]
    return out


def scalar_eval(variant, genome, grid):
    """(Merkmale, Direktheit, Endposition, Menge der berührten 2 × 2-Zellen) einer Fahrt, skalar."""
    pts = scalar_track(genome, grid)
    driven = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
    ex, ey = pts[-1]
    q = math.hypot(ex - C.START[0], ey - C.START[1]) / driven if driven > 1e-9 else 0.0
    blocks = lambda x, y: min(max(int(x // 2), 0), 11) * 12 + min(max(int(y // 2), 0), 11)
    return scalar_features(variant, genome, pts), q, (ex, ey), frozenset(blocks(x, y) for x, y in pts)


def scalar_code(space, f):
    """Standardisierter Code eines Merkmalsvektors mit reinen Schleifen (Identität, PCA oder Autoencoder)."""
    enc = space.encoder
    if enc.kind == "identity":
        code = list(f)
    elif enc.kind == "pca":
        code = [sum((f[j] - enc.mean[j]) * axis[j] for j in range(len(f))) for axis in enc.axes]
    else:
        z = [(f[j] - enc.mean[j]) / enc.scale[j] for j in range(len(f))]
        for i, (w, b) in enumerate(zip(enc.weights, enc.biases)):
            z = [sum(z[j] * w[j][k] for j in range(len(z))) + b[k] for k in range(w.shape[1])]
            if i in (0, 2):
                z = [math.tanh(v) for v in z]
            if i == 1:
                code = list(z)
                break
    return [(c - m) / s for c, m, s in zip(code, space.mu, space.sd)]


def scalar_cell(space, f):
    code = scalar_code(space, f)
    dist = [sum((a - b) ** 2 for a, b in zip(code, c)) for c in space.centroids]
    return dist.index(min(dist))


def insert(result, cell, item):
    if cell not in result or item[0] > result[cell][0] + S.QUALITY_TIE:
        result[cell] = item


def metrics(result):
    if not result:
        return 0, 0.0, 0, 0
    ends = {(min(max(int(v[3][0] // 2), 0), 11), min(max(int(v[3][1] // 2), 0), 11)) for v in result.values()}
    return len(result), sum(v[0] for v in result.values()), len(ends), len(set().union(*[v[4] for v in result.values()]))


def reference_run(grid, seed, variant, space, steps, gens, pop):
    """Buchführung mit Wörterbüchern. Eintrag je Zelle: (Qualität, Steuerfolge, Merkmale, Endposition, berührte Zellen). Gibt (Archiv, Kurven) zurück."""
    rng = np.random.default_rng(seed)
    mode, method, d = C.VARIANT_MODE[variant], SP.variant_method(variant), C.VARIANT_DIM[variant]
    genomes = R.random_genomes(rng, pop, steps)
    result, warm, warm_feats, curves = {}, [], None, []
    for gen in range(gens):
        info = [scalar_eval(variant, g, grid) for g in genomes]
        items = [(q, genomes[k].copy(), f, end, path) for k, (f, q, end, path) in enumerate(info)]
        started = True
        if mode != "aur":
            for it in items:
                insert(result, scalar_cell(space, it[2]), it)
        elif gen < C.WARMUP_GENS - 1:
            warm += items
            started = False
        elif gen == C.WARMUP_GENS - 1:
            warm += items
            warm_feats = np.array([w[2] for w in warm])
            space = SP.fit_space(method, d, warm_feats)
            for it in warm:
                insert(result, scalar_cell(space, it[2]), it)
        else:
            if gen in C.RETRAIN_GENS:
                data = np.vstack([warm_feats, np.array([result[c][2] for c in sorted(result)]), np.array([it[2] for it in items])])
                space = SP.fit_space(method, d, data)
                old, result = result, {}
                for c in sorted(old, key=lambda c: -old[c][0]):
                    result.setdefault(scalar_cell(space, old[c][2]), old[c])
            for it in items:
                insert(result, scalar_cell(space, it[2]), it)
        curves.append(metrics(result) if started else (0, 0.0, 0, 0))
        if not started:
            genomes = R.random_genomes(rng, pop, steps)
            continue
        keys = sorted(result)
        pick = rng.integers(0, len(keys), pop)
        parents = np.array([result[keys[i]][1] for i in pick])
        mask = rng.random(parents.shape) < C.MUT_RATE
        genomes = R.clip_genomes(parents + rng.normal(0.0, C.MUT_SIGMA, parents.shape) * mask)
    return result, curves


@pytest.mark.parametrize("steps", [5, 40, 140])
@pytest.mark.parametrize("level", [0, 3, 4])
def test_directness_agrees_with_the_scalar_reference_for_every_genome_length(level, steps):
    grid = M.make_maze(level)
    genomes = R.random_genomes(np.random.default_rng(300 + level + steps), 12, steps)
    _, track = R.simulate(genomes, grid, trajectories=True)
    assert S.directness(track) == pytest.approx([scalar_eval("hand_end", g, grid)[1] for g in genomes], abs=1e-10)


@pytest.mark.parametrize("variant", ["hand_end", "hand_way4", "hand_style", "fixed_ae2"])
def test_the_features_agree_with_the_scalar_reference(variant):
    grid = M.make_maze(3)
    genomes = R.random_genomes(np.random.default_rng(11), 10, 37)
    _, track = R.simulate(genomes, grid, trajectories=True)
    assert SP.features(variant, genomes, track) == pytest.approx(np.array([scalar_eval(variant, g, grid)[0] for g in genomes]), abs=1e-10)


def test_the_path_masks_agree_with_the_scalar_reference():
    grid = M.make_maze(3)
    genomes = R.random_genomes(np.random.default_rng(12), 10, 60)
    _, track = R.simulate(genomes, grid, trajectories=True)
    masks = S.path_masks(track)
    for g, m in zip(genomes, masks):
        assert set(np.nonzero(m)[0]) == set(scalar_eval("hand_end", g, grid)[3])


@pytest.mark.parametrize("variant", ["hand_end", "hand_style", "fixed_pca8", "fixed_ae2"])
def test_the_scalar_code_and_cell_agree_with_the_vectorised_assignment(variant):
    space = E.get_space(variant, 0)
    grid = M.make_maze(0)
    genomes = R.random_genomes(np.random.default_rng(13), 40, 140)
    f = SP.features(variant, genomes, R.simulate(genomes, grid, trajectories=True)[1])
    assert space.code(f) == pytest.approx(np.array([scalar_code(space, row) for row in f]), abs=1e-9)
    assert list(space.assign(f)) == [scalar_cell(space, row) for row in f]


@pytest.mark.parametrize("variant,level", [("hand_end", 0), ("hand_way4", 3), ("hand_style", 3), ("fixed_pca2", 0), ("fixed_pca8", 3), ("fixed_ae2", 0), ("fixed_ae8", 3)])
def test_map_elites_with_each_stored_space_agrees_with_a_dictionary_based_reimplementation(variant, level):
    grid = M.make_maze(level)
    space = E.get_space(variant, level)
    steps, pop, gens = 12, 25, 10
    ref, curves = reference_run(grid, 5, variant, space, steps, gens, pop)
    res = S.run_search(grid, 5, variant, space=space, steps=steps, gens=gens, pop=pop, keep_archive=True)
    assert res.coverage == [c[0] for c in curves] and res.qd_score == pytest.approx([c[1] for c in curves], abs=1e-8) and res.end_cells == [c[2] for c in curves] and res.path_cells == [c[3] for c in curves]
    keys = sorted(ref)
    assert res.elite_quality == pytest.approx([ref[c][0] for c in keys], abs=1e-10) and res.elite_ends == pytest.approx(np.array([ref[c][3] for c in keys]), abs=1e-10)


@pytest.mark.parametrize("variant", ["aur_pca2", "aur_ae2", "aur_pca8"])
def test_the_aurora_loop_agrees_with_a_dictionary_based_reimplementation(variant, monkeypatch):
    monkeypatch.setattr(C, "K", 24)
    monkeypatch.setattr(C, "WARMUP_GENS", 3)
    monkeypatch.setattr(C, "RETRAIN_GENS", (4, 7))
    grid = M.make_maze(3)
    steps, pop, gens = 10, 30, 10
    ref, curves = reference_run(grid, 7, variant, None, steps, gens, pop)
    res = S.run_search(grid, 7, variant, steps=steps, gens=gens, pop=pop, keep_archive=True)
    assert res.retrains == 2
    assert res.coverage == [c[0] for c in curves] and res.qd_score == pytest.approx([c[1] for c in curves], abs=1e-8) and res.end_cells == [c[2] for c in curves] and res.path_cells == [c[3] for c in curves]
    keys = sorted(ref)
    assert res.elite_quality == pytest.approx([ref[c][0] for c in keys], abs=1e-10) and res.elite_ends == pytest.approx(np.array([ref[c][3] for c in keys]), abs=1e-10)


def test_best_progress_agrees_with_scipys_breadth_first_search():
    grid = M.make_maze(4)
    res = S.run_search(grid, 2, "hand_end", space=E.get_space("hand_end", 0), gens=6)
    ref = scipy_distances(grid, M.cell_of(C.GOAL))
    end = R.simulate(R.random_genomes(np.random.default_rng(2), C.POP, 140), grid)
    assert res.best_progress[0] == min(ref[int(x), int(y)] for x, y in end) and all(b <= a for a, b in zip(res.best_progress, res.best_progress[1:]))
