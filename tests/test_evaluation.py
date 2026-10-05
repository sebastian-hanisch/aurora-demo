"""Auswertung: Täuschungsmaße je Stufe, Lösbarkeit, Live-Paar, Hilfsfunktionen der Studie, Vollständigkeit und Stimmigkeit der vorgerechneten Datei (inklusive Neurechnung einzelner Läufe)."""

import numpy as np
import pytest
from scipy.stats import mannwhitneyu

import aur_constants as C
import aur_evaluation as E
import aur_maze as M
import aur_search as S

PRE = E.load_precomputed()


def test_level_metrics_by_hand():
    m0, m3 = E.level_metrics(0), E.level_metrics(3)
    assert m0["path"] == 17 and m0["straight"] == pytest.approx(17.0) and m0["detour"] == pytest.approx(1.0) and m0["minima"] == 0 and m0["trap"] is None and m0["free"] == 22 * 22
    assert m3["path"] == 49 and m3["detour"] == pytest.approx(49 / 17) and m3["minima"] == 1 and m3["trap"] == pytest.approx(9.0) and m3["free"] == 22 * 22 - 19


@pytest.mark.parametrize("level", list(C.LEVELS))
def test_every_level_is_solvable_within_the_step_budget(level):
    dist, steps = E.solvability(level)
    assert dist < 0.5 and steps <= C.STEPS


def test_live_pair_runs_two_descriptions_from_the_same_start_and_keeps_their_archives():
    a, b = E.live_pair(0, 12, 5, "hand_end", "aur_ae2")
    assert len(a.coverage) == len(b.coverage) == 12 and a.elite_ends.shape == (a.coverage[-1], 2) and b.elite_codes.shape == (b.coverage[-1], 2)
    assert a.recon is None and b.recon is not None


def test_live_run_equals_a_direct_run_with_the_stored_space():
    res = E.live_run(3, 8, 2, "fixed_pca8")
    direct = S.run_search(M.make_maze(3), 2, "fixed_pca8", space=E.get_space("fixed_pca8", 3), gens=8)
    assert res.coverage == direct.coverage and res.qd_score == direct.qd_score and res.end_cells == direct.end_cells


def test_the_live_run_limits_the_blas_threads_to_one_while_it_computes(monkeypatch):
    """Auf geteilten Rechnern bremsen viele BLAS-Threads das Training der Autoencoder stark: der Live-Lauf rechnet mit einem Thread."""
    from threadpoolctl import threadpool_info
    seen = []
    real = S.run_search

    def spy(*args, **kwargs):
        seen.append([lib["num_threads"] for lib in threadpool_info() if lib.get("user_api") == "blas"])
        return real(*args, **kwargs)

    monkeypatch.setattr(S, "run_search", spy)
    E.live_run(0, 10, 1, "aur_ae2")
    assert seen and all(n == 1 for libs in seen for n in libs)


def test_success_helpers_by_hand():
    runs = [100, None, 500, 50, None]
    assert E.success_count(runs) == 3 and E.success_count(runs, 100) == 2 and E.median_solved(runs) == 100.0 and E.median_solved([None, None]) is None


def test_mann_whitney_agrees_with_scipy():
    x, y = [1.0, 2.0, 3.0, 4.0, 5.0], [3.5, 4.5, 5.5, 6.5, 7.5]
    assert E.mannwhitney_p(x, y) == pytest.approx(mannwhitneyu(x, y, alternative="two-sided").pvalue) and E.mannwhitney_p(x, x) == pytest.approx(1.0)


def test_the_study_covers_every_level_and_variant():
    assert set(PRE["levels"]) == {str(lv) for lv in C.STUDY_LEVELS} == set(PRE["cells"]) and PRE["budget"] == C.STUDY_BUDGET
    for lv in C.STUDY_LEVELS:
        assert set(PRE["cells"][str(lv)]) == set(C.VARIANTS)
        for v in C.VARIANTS:
            c = E.study_cell(PRE, lv, v)
            n = C.STUDY_SEEDS
            assert len(c["runs"]) == len(c["progress"]) == len(c["best_quality"]) == len(c["coverage"]) == len(c["qd"]) == len(c["end_cells"]) == len(c["path_cells"]) == len(c["recon"]) == n
            assert len(c["mean_coverage"]) == len(c["mean_end_cells"]) == len(c["mean_path_cells"]) == len(c["mean_qd"]) == C.STUDY_GENS
            assert max(c["coverage"]) <= C.K and max(c["end_cells"]) <= 144 and max(c["path_cells"]) <= 144 and min(c["qd"]) >= 0 and all(0.0 <= q <= 1.0 + 1e-9 for q in c["best_quality"])
            assert all(e <= cov for e, cov in zip(c["end_cells"], c["coverage"])) and all(q <= cov + 1e-9 for q, cov in zip(c["qd"], c["coverage"]))
            assert all((r is None) == (C.VARIANT_MODE[v] == "hand") for r in c["recon"])


def test_the_stored_level_metrics_equal_the_exact_ones():
    for lv in C.STUDY_LEVELS:
        stored, fresh = E.level_info(PRE, lv), E.level_metrics(lv)
        for key in ("path", "straight", "detour", "fdc", "minima", "free"):
            assert stored[key] == pytest.approx(fresh[key])


@pytest.mark.parametrize("level,variant,seed", [(0, "hand_end", 0), (3, "hand_style", 3), (0, "hand_way4", 7), (3, "fixed_pca2", 5), (0, "fixed_pca8", 11), (3, "fixed_ae2", 2), (0, "fixed_ae8", 4)])
def test_a_study_run_with_a_stored_space_is_reproduced_exactly_by_a_fresh_run_with_the_same_seed(level, variant, seed):
    """Hand- und einmal gelernte Beschreibungen haben einen gespeicherten Raum: alle gespeicherten Werte eines Laufs stimmen mit einem frischen Lauf überein."""
    res = S.run_search(M.make_maze(level), seed, variant, space=E.get_space(variant, level), gens=C.STUDY_GENS)
    c = E.study_cell(PRE, level, variant)
    assert res.first_solved == c["runs"][seed] and res.best_progress[-1] == c["progress"][seed] and res.best_quality[-1] == pytest.approx(c["best_quality"][seed], abs=1e-6)
    assert res.coverage[-1] == c["coverage"][seed] and res.qd_score[-1] == pytest.approx(c["qd"][seed], abs=1e-5) and res.end_cells[-1] == c["end_cells"][seed] and res.path_cells[-1] == c["path_cells"][seed]


def within(value, values, k=4.0, floor=1.0):
    """Liegt `value` höchstens k Standardabweichungen (mindestens `floor`) vom Mittel der gespeicherten Werte?"""
    values = np.asarray(values, float)
    return abs(value - values.mean()) <= k * max(float(values.std(ddof=1)), floor)


@pytest.mark.parametrize("level,variant,seed", [(0, "aur_pca2", 1), (3, "aur_ae2", 2), (0, "aur_ae8", 3), (3, "aur_pca8", 4)])
def test_an_aurora_study_run_falls_into_the_stored_distribution_when_run_again(level, variant, seed):
    """Die AURORA-Läufe lernen im Lauf (Singulärzerlegung, Autoencoder-Training): BLAS und LAPACK rechnen je Plattform in den letzten Stellen anders, und eine Zelle am Rand kippt dann; ein frischer Lauf ist deshalb ein anderer Zug aus derselben Verteilung.
    Die Zahl der Endzellen und Wegzellen liegt innerhalb von 4 Standardabweichungen der 20 gespeicherten Läufe."""
    res = S.run_search(M.make_maze(level), seed, variant, gens=C.STUDY_GENS)
    c = E.study_cell(PRE, level, variant)
    assert within(res.end_cells[-1], c["end_cells"], floor=3.0) and within(res.path_cells[-1], c["path_cells"], floor=3.0) and within(res.qd_score[-1], c["qd"], floor=3.0) and res.retrains == len(C.RETRAIN_GENS)


def test_the_mean_curves_are_the_mean_over_the_runs():
    c = E.study_cell(PRE, 0, "hand_way4")
    assert all(y >= x for x, y in zip(c["mean_coverage"], c["mean_coverage"][1:]))              # mit festem Raum schrumpft die Abdeckung nie
    assert c["mean_coverage"][-1] == pytest.approx(np.mean(c["coverage"]), abs=1e-3) and c["mean_end_cells"][-1] == pytest.approx(np.mean(c["end_cells"]), abs=1e-3)


def test_in_aurora_runs_the_first_generations_have_no_archive_in_the_mean_curves():
    c = E.study_cell(PRE, 0, "aur_pca2")
    assert c["mean_coverage"][:C.WARMUP_GENS - 1] == [0.0] * (C.WARMUP_GENS - 1) and c["mean_coverage"][C.WARMUP_GENS - 1] > 0
