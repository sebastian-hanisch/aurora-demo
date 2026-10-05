"""Rechnet beim Bau vor (nicht in der App): `python generate_precomputed.py [Prozesse] [--keep-spaces]` schreibt `spaces.npz` und `precomputed_sweep.json`.

  spaces.npz   die Räume der handgemachten und der einmal gelernten Beschreibungen (Encoder-Gewichte, Standardisierung, 256 Schwerpunkte), je Stufe (freies Feld, Stufe 3)
  cells        je Stufe × Variante (11) 20 Läufe MAP-Elites mit 100 Kindern je Generation, 400 Generationen (40 000 Evaluationen):
               Evaluationen bis zum ersten Erfolg (None = nie), wahrer Restweg, beste Direktheit, belegte Zellen, QD-Score, verschiedene Endzellen, Wegzellen und Rekonstruktionsfehler am Ende,
               mittlerer Verlauf von Abdeckung, QD-Score, Endzellen und Wegzellen
  levels       exakte Täuschungsmaße der beiden Stufen"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):          # vor numpy: ein BLAS-Thread je Prozess, sonst bremsen sich die Arbeiter aus
    os.environ.setdefault(_var, "1")

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import aur_constants as C
import aur_evaluation as E
import aur_maze as M
import aur_search as S
import aur_space as SP


def _space_task(args):
    variant, level = args
    return variant, level, SP.train_space(variant, level)


def _task(args):
    level, variant, seed = args
    space = None if C.VARIANT_MODE[variant] == "aur" else E.get_space(variant, level)
    res = S.run_search(M.make_maze(level), seed, variant, space=space, gens=C.STUDY_GENS)
    return (level, variant, seed, res.first_solved, res.best_progress[-1], res.best_quality[-1], res.coverage, res.qd_score, res.end_cells, res.path_cells, res.recon)


def _curve(rows, index):
    return [round(float(x), 3) for x in np.mean([r[index] for r in rows], axis=0)]


def main(workers, keep_spaces):
    t0 = time.time()
    if keep_spaces and E.SPACES_PATH.exists():
        print("Räume aus spaces.npz übernommen", flush=True)
    else:
        tasks = [(v, lv) for v in C.STORED_VARIANTS for lv in C.STUDY_LEVELS]
        out = {}
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for variant, level, space in ex.map(_space_task, tasks, chunksize=1):
                E.store_space(out, variant, level, space)
        np.savez_compressed(E.SPACES_PATH, **out)
        print(f"{len(tasks)} Räume gerechnet nach {time.time() - t0:.0f} s", flush=True)
    tasks = [(lv, v, s) for lv in C.STUDY_LEVELS for v in C.VARIANTS for s in range(C.STUDY_SEEDS)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_task, tasks, chunksize=2))
    cells = {}
    for lv in C.STUDY_LEVELS:
        cells[str(lv)] = {}
        for v in C.VARIANTS:
            rows = sorted([r for r in res if r[0] == lv and r[1] == v], key=lambda r: r[2])
            cells[str(lv)][v] = {"runs": [r[3] for r in rows], "progress": [r[4] for r in rows], "best_quality": [round(r[5], 6) for r in rows], "coverage": [r[6][-1] for r in rows],
                                 "qd": [round(r[7][-1], 6) for r in rows], "end_cells": [r[8][-1] for r in rows], "path_cells": [r[9][-1] for r in rows],
                                 "recon": [None if r[10] is None else round(r[10], 8) for r in rows], "mean_coverage": _curve(rows, 6), "mean_qd": _curve(rows, 7),
                                 "mean_end_cells": _curve(rows, 8), "mean_path_cells": _curve(rows, 9)}
    data = {"budget": C.STUDY_BUDGET, "levels": {str(lv): E.level_metrics(lv) for lv in C.STUDY_LEVELS}, "cells": cells}
    E.PRECOMPUTED_PATH.write_text(json.dumps(data), encoding="utf-8")
    print(f"fertig in {time.time() - t0:.0f} s -> {E.PRECOMPUTED_PATH}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(int(args[0]) if args else min(14, os.cpu_count() or 1), "--keep-spaces" in sys.argv)
