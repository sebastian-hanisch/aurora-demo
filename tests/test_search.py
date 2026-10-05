"""Archiv, Neueintragen und Lauf (Hand, einmal gelernt, AURORA-Schleife) von Hand und an kleinen Instanzen."""

import numpy as np
import pytest

import aur_constants as C
import aur_evaluation as E
import aur_maze as M
import aur_robot as R
import aur_search as S

FREE = M.make_maze(0)


class FakeSpace:
    """Raum mit vorgegebener Zuordnung Merkmal -> Zelle: die erste Merkmalsspalte ist die Zelle."""
    def assign(self, X):
        return np.asarray(X)[:, 0].astype(int)


def test_directness_of_a_straight_slow_ride_is_one_and_of_standing_still_zero():
    g = np.zeros((1, 140, 2))
    g[:, :, 1] = 0.1
    assert S.directness(R.simulate(g, FREE, trajectories=True)[1])[0] == pytest.approx(1.0)
    assert S.directness(R.simulate(np.zeros((1, 140, 2)), FREE, trajectories=True)[1])[0] == 0.0


def test_directness_of_an_out_and_back_route_by_hand():
    """(3.5, 3.5) -> (3.5, 5.5) -> (4.5, 5.5) -> (4.5, 3.5): gefahren 2 + 1 + 2 = 5, Luftlinie Start–Ende 1: Direktheit 0.2."""
    assert S.directness(np.array([[[3.5, 3.5], [3.5, 5.5], [4.5, 5.5], [4.5, 3.5]]]))[0] == pytest.approx(0.2)


def test_block_index_by_hand():
    assert list(S.block_index(np.array([[3.5, 3.5], [23.9, 0.1], [2.0, 2.0], [1.99, 4.0], [30.0, -2.0]]))) == [1 * 12 + 1, 11 * 12 + 0, 1 * 12 + 1, 0 * 12 + 2, 11 * 12 + 0]
    assert S.BLOCKS == 12 and C.END_CELLS_TOTAL == 144


def test_path_masks_mark_every_two_by_two_block_a_ride_touches_by_hand():
    """Eine Fahrt von (0.5, 0.5) über (2.5, 0.5) nach (2.5, 4.5) berührt die Blöcke (0, 0), (1, 0), (1, 1), (1, 2) = 0, 12, 13, 14."""
    track = np.array([[[0.5, 0.5], [2.5, 0.5], [2.5, 2.5], [2.5, 4.5]]])
    mask = S.path_masks(track)
    assert mask.shape == (1, 144) and list(np.nonzero(mask[0])[0]) == [0, 12, 13, 14]


def test_reinsertion_plan_keeps_the_first_of_each_new_cell_by_hand():
    assert S.reinsertion_plan([7, 3, 5, 9], [2, 2, 4, 2]) == {2: 7, 4: 5}
    assert S.reinsertion_plan([], []) == {}


def test_insert_fills_an_empty_cell_replaces_with_better_keeps_better_and_ignores_equal_or_rounding():
    arch = S.Archive(1, 1, k=4)
    genome = lambda v: np.full((1, 1, 2), v)
    feats, ends, masks = np.zeros((1, 1)), np.zeros((1, 2)), np.zeros((1, 144), bool)
    for q, v in ((0.4, 1.0), (0.6, 2.0), (0.5, 3.0), (0.6, 4.0), (0.6 + 1e-15, 5.0)):
        arch.insert(np.array([1]), np.array([q]), genome(v), feats, ends, masks)
    assert arch.quality[1] == 0.6 and arch.genomes[1, 0, 0] == 2.0 and list(arch.filled) == [1]
    arch.insert(np.array([1]), np.array([0.6 + 1e-9]), genome(6.0), feats, ends, masks)
    assert arch.genomes[1, 0, 0] == 6.0


def test_a_child_with_quality_zero_fills_an_empty_cell_and_a_batch_goes_in_order():
    arch = S.Archive(1, 1, k=2)
    z = lambda n: (np.zeros((n, 1)), np.zeros((n, 2)), np.zeros((n, 144), bool))
    arch.insert(np.array([0]), np.array([0.0]), np.full((1, 1, 2), 5.0), *z(1))
    assert arch.quality[0] == 0.0 and list(arch.filled) == [0]
    arch.insert(np.array([0, 0, 0]), np.array([0.3, 0.9, 0.5]), np.array([[[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]]]), *z(3))
    assert arch.quality[0] == 0.9 and arch.genomes[0, 0, 0] == 2.0


def test_archive_metrics_by_hand():
    arch = S.Archive(1, 1, k=4)
    assert arch.metrics() == (0, 0.0, 0, 0)
    masks = np.zeros((2, 144), bool)
    masks[0, [0, 1]], masks[1, [1, 2]] = True, True
    arch.insert(np.array([0, 3]), np.array([0.5, 0.25]), np.zeros((2, 1, 2)), np.zeros((2, 1)), np.array([[0.5, 0.5], [1.5, 1.5]]), masks)
    assert arch.metrics() == (2, 0.75, 1, 3)                       # beide Enden im Block (0, 0); die Wege berühren 0, 1, 2
    arch.insert(np.array([2]), np.array([0.25]), np.zeros((1, 1, 2)), np.zeros((1, 1)), np.array([[10.5, 10.5]]), np.zeros((1, 144), bool))
    assert arch.metrics()[2] == 2


def test_rebuilding_the_archive_in_a_new_space_keeps_the_better_of_two_colliding_elites_by_hand():
    arch = S.Archive(1, 1, k=4)
    feats = np.array([[0.0], [1.0], [1.0]])                         # neue Zelle = Merkmal: Elite aus alter Zelle 0 → 0, aus 1 → 1, aus 2 → 1 (Kollision)
    arch.insert(np.array([0, 1, 2]), np.array([0.2, 0.5, 0.8]), np.array([[[0.0, 0.0]], [[1.0, 1.0]], [[2.0, 2.0]]]), feats, np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]]), np.zeros((3, 144), bool))
    new = arch.rebuilt(FakeSpace())
    assert list(new.filled) == [0, 1] and new.quality[0] == 0.2 and new.quality[1] == 0.8 and new.genomes[1, 0, 0] == 2.0 and new.ends[1] == pytest.approx([2.5, 2.5])


def test_every_run_has_one_entry_per_generation_and_the_quality_series_never_get_worse():
    for variant in ("hand_end", "fixed_pca2", "aur_pca2"):
        space = None if C.VARIANT_MODE[variant] == "aur" else E.get_space(variant, 0)
        res = S.run_search(FREE, 1, variant, space=space, gens=30, pop=60)
        for series in (res.coverage, res.qd_score, res.end_cells, res.path_cells, res.best_quality, res.best_progress):
            assert len(series) == 30
        assert all(b >= a - 1e-9 for a, b in zip(res.best_quality, res.best_quality[1:])) and all(b <= a for a, b in zip(res.best_progress, res.best_progress[1:]))
        assert res.evaluations == 30 * 60 and max(res.coverage) <= C.K and max(res.end_cells) <= 144 and max(res.path_cells) <= 144


def test_with_a_fixed_space_coverage_and_the_qd_score_never_shrink():
    res = S.run_search(FREE, 2, "hand_way4", space=E.get_space("hand_way4", 0), gens=40, pop=40)
    assert all(b >= a for a, b in zip(res.coverage, res.coverage[1:])) and all(b >= a - 1e-9 for a, b in zip(res.qd_score, res.qd_score[1:]))


def test_the_aurora_loop_has_no_archive_before_the_first_learned_space_and_retrains_on_schedule():
    res = S.run_search(FREE, 3, "aur_pca2", gens=120)
    assert res.coverage[:C.WARMUP_GENS - 1] == [0] * (C.WARMUP_GENS - 1) and res.qd_score[:4] == [0.0] * 4 and res.end_cells[:4] == [0] * 4 and res.path_cells[:4] == [0] * 4
    assert res.coverage[C.WARMUP_GENS - 1] > 0 and res.retrains == sum(1 for g in C.RETRAIN_GENS if g < 120) == 4


def test_the_warm_up_inserts_all_random_rides_of_the_first_generations():
    """Die ersten 5 Generationen sind 500 Zufallsfahrten; nach der 5. sind höchstens 256 Zellen belegt (K) und mindestens die Hälfte (500 Fahrten, k-Means-Zellen aus denselben Daten)."""
    res = S.run_search(FREE, 3, "aur_ae2", gens=C.WARMUP_GENS)
    assert 128 <= res.coverage[-1] <= C.K and res.retrains == 0


def test_the_first_children_after_the_warm_up_come_from_the_elites_not_from_random_rides():
    """Nach dem Aufwärmen bleibt die Zahl der Endzellen nicht bei Zufallsniveau: nach 40 Generationen liegt der QD-Score über dem nach dem Aufwärmen."""
    res = S.run_search(FREE, 3, "aur_pca2", gens=40)
    assert res.qd_score[-1] > res.qd_score[C.WARMUP_GENS - 1]


def test_a_run_is_reproduced_exactly_by_a_second_run_with_the_same_seed_and_differs_for_another_seed():
    kw = dict(gens=15, pop=60, steps=30)
    space = E.get_space("hand_end", 0)
    a, b, c = (S.run_search(FREE, s, "hand_end", space=space, **kw) for s in (5, 5, 6))
    assert a.coverage == b.coverage and a.qd_score == b.qd_score and a.end_cells == b.end_cells and a.qd_score != c.qd_score
    x, y = (S.run_search(FREE, 5, "aur_pca2", **kw) for _ in range(2))
    assert x.qd_score == y.qd_score and x.path_cells == y.path_cells


def test_the_archive_does_not_use_the_goal(monkeypatch):
    """Das Archiv kennt das Ziel nicht: ein verschobenes Ziel ändert Abdeckung, QD-Score, Endzellen, Wegzellen und beste Direktheit nicht; nur die Messgrößen Erfolg und Restweg ändern sich."""
    space = E.get_space("hand_end", 0)
    a = S.run_search(FREE, 2, "hand_end", space=space, gens=10, pop=40)
    monkeypatch.setattr(C, "GOAL", (20.5, 3.5))
    b = S.run_search(FREE, 2, "hand_end", space=space, gens=10, pop=40)
    assert a.coverage == b.coverage and a.qd_score == b.qd_score and a.end_cells == b.end_cells and a.path_cells == b.path_cells and a.best_quality == b.best_quality
    assert a.best_progress != b.best_progress


def test_first_solved_counts_evaluations_up_to_the_first_child_in_the_goal_circle():
    res = S.run_search(FREE, 1, "hand_end", space=E.get_space("hand_end", 0), gens=200)
    genomes = R.random_genomes(np.random.default_rng(1), C.POP, C.STEPS)
    near = R.distance_to_goal(R.simulate(genomes, FREE)) < C.SUCCESS_RADIUS
    if near.any():
        assert res.first_solved == int(np.argmax(near)) + 1
    else:
        assert res.first_solved is None or res.first_solved > C.POP


def test_the_variants_need_or_forbid_a_space():
    with pytest.raises(ValueError):
        S.run_search(FREE, 1, "hand_end", gens=2)
    with pytest.raises(ValueError):
        S.run_search(FREE, 1, "aur_ae2", space=E.get_space("hand_end", 0), gens=2)


def test_the_stored_archive_has_one_end_position_one_quality_and_one_code_per_filled_cell():
    res = S.run_search(FREE, 1, "aur_ae2", gens=30, pop=60, keep_archive=True)
    assert res.elite_ends.shape == (res.coverage[-1], 2) and res.elite_quality.shape == (res.coverage[-1],) and res.elite_codes.shape == (res.coverage[-1], 2)
    assert res.elite_quality.sum() == pytest.approx(res.qd_score[-1]) and (res.elite_ends >= 0).all() and (res.elite_ends < C.N).all()
    assert res.recon is not None and res.recon > 0
    assert S.run_search(FREE, 1, "hand_end", space=E.get_space("hand_end", 0), gens=5, pop=40).elite_ends is None


def test_hand_variants_have_no_reconstruction_error_and_learned_ones_do():
    assert S.run_search(FREE, 1, "hand_style", space=E.get_space("hand_style", 0), gens=5, pop=40).recon is None
    assert S.run_search(FREE, 1, "fixed_pca8", space=E.get_space("fixed_pca8", 0), gens=5, pop=40).recon > 0


def test_a_single_cell_space_keeps_the_most_direct_ride_of_everything_it_has_seen(monkeypatch):
    import aur_learn as L
    import aur_space as SP
    monkeypatch.setattr(C, "K", 1)
    space = SP.Space(L.IdentityEncoder(), np.zeros(2), np.ones(2), np.zeros((1, 2)))
    res = S.run_search(FREE, 1, "hand_end", space=space, gens=15, pop=40)
    assert res.coverage[-1] == 1 and res.qd_score[-1] == pytest.approx(res.best_quality[-1])
