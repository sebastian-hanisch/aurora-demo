"""Auswertung: Täuschungsmaße je Stufe (exakt), Lösbarkeitsnachweis, Live-Paar (zwei Beschreibungen mit demselben Seed), Lesen und Aggregieren der vorgerechneten Studie, Mann-Whitney-Test.
Die Studie (11 Varianten × 2 Stufen × 20 Läufe) und die Räume der handgemachten und einmal gelernten Beschreibungen werden beim Bau gerechnet (`generate_precomputed.py`) und hier nur gelesen."""

import json
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

import aur_constants as C
import aur_maze as M
import aur_robot as R
import aur_search as S
import aur_space as SP

try:
    from threadpoolctl import threadpool_limits
except ImportError:                                  # ohne threadpoolctl läuft alles weiter, nur ohne Begrenzung der BLAS-Threads
    def threadpool_limits(*args, **kwargs):
        return nullcontext()

PRECOMPUTED_PATH = Path(__file__).resolve().parent / "precomputed_sweep.json"
SPACES_PATH = Path(__file__).resolve().parent / "spaces.npz"


def level_metrics(level):
    """Exakte Täuschungsmaße der Stufe: kürzester Weg (Zellen), Luftlinie Start–Ziel, Umwegfaktor, Fitness-Distanz-Korrelation, lokale Minima (Zahl), Abstand des tiefsten Minimums zum Ziel, freie Zellen."""
    g = M.make_maze(level)
    path = M.shortest_path(g)
    return {"path": len(path) - 1, "straight": float(np.hypot(C.START[0] - C.GOAL[0], C.START[1] - C.GOAL[1])), "detour": M.detour_ratio(g), "fdc": M.fitness_distance_correlation(g),
            "minima": len(M.local_minima(g)), "trap": M.trap_distance(g), "free": int((~g).sum())}


def solvability(level):
    """Lösbarkeitsnachweis: eine aus dem kürzesten Pfad abgeleitete Steuerfolge fährt in diesem Labyrinth ins Ziel. Gibt (Abstand am Ende, benötigte Schritte) zurück."""
    g = M.make_maze(level)
    genome = R.controller_from_path(M.shortest_path(g))
    end = R.simulate(genome[None], g)
    return float(R.distance_to_goal(end)[0]), int((np.abs(genome).sum(axis=1) > 0).sum())


@lru_cache(maxsize=1)
def _spaces_file():
    with np.load(SPACES_PATH) as f:
        return {key: f[key] for key in f.files}


def space_key(variant, level):
    return f"{variant}__{level}"


def store_space(out, variant, level, space):
    """Ablegen eines Raums in das Wörterbuch für `spaces.npz`."""
    for name, arr in space.arrays().items():
        out[f"{space_key(variant, level)}__{name}"] = arr


def get_space(variant, level):
    """Vorgerechneter Raum einer gespeicherten Variante (hand_*, fixed_*) für die Stufe `level`."""
    prefix = space_key(variant, level) + "__"
    stored = {k[len(prefix):]: v for k, v in _spaces_file().items() if k.startswith(prefix)}
    if not stored:
        raise ValueError(f"kein vorgerechneter Raum für {variant}, Stufe {level}")
    return SP.Space.from_arrays(SP.variant_method(variant), stored)


def live_run(level, gens, seed, variant):
    """Ein Lauf mit der Beschreibung `variant`; das Archiv wird gespeichert."""
    g = M.make_maze(level)
    space = None if C.VARIANT_MODE[variant] == "aur" else get_space(variant, level)
    with threadpool_limits(limits=1, user_api="blas"):          # die Matrizen sind klein; viele BLAS-Threads bremsen auf geteilten Rechnern (Streamlit Cloud) das Training
        return S.run_search(g, seed, variant, space=space, gens=gens, keep_archive=True)


def live_pair(level, gens, seed, a, b):
    """Derselbe Seed und dieselbe Stufe: zwei Beschreibungen nebeneinander. Gibt die beiden Läufe zurück."""
    return live_run(level, gens, seed, a), live_run(level, gens, seed, b)


@lru_cache(maxsize=1)
def load_precomputed():
    return json.loads(PRECOMPUTED_PATH.read_text(encoding="utf-8"))


def success_count(runs, budget=None):
    """Zahl der Läufe, die den Zielkreis erreichten (höchstens `budget` Evaluationen, wenn angegeben)."""
    return sum(1 for r in runs if r is not None and (budget is None or r <= budget))


def median_solved(runs):
    """Median der Evaluationen bis zum Erfolg über die erfolgreichen Läufe (None, wenn kein Lauf erfolgreich war)."""
    ok = [r for r in runs if r is not None]
    return float(np.median(ok)) if ok else None


def mannwhitney_p(values_a, values_b):
    """Zweiseitiger Mann-Whitney-Test auf zwei Zahlenlisten (etwa die verschiedenen Endzellen je Lauf)."""
    return float(mannwhitneyu(values_a, values_b, alternative="two-sided").pvalue)


def study_cell(pre, level, variant):
    """Die Läufe (20 Seeds) einer Zelle der Studie."""
    return pre["cells"][str(level)][variant]


def level_info(pre, level):
    return pre["levels"][str(level)]
