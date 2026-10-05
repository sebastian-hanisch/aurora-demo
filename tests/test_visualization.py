"""Abbildungen: gesperrte Achsen, Spuren, Beschriftungen, Farben, Fehlerbalken."""

import numpy as np
import pytest

import aur_constants as C
import aur_evaluation as E
import aur_maze as M
import aur_visualization as V

PRE = E.load_precomputed()


def _locked(fig):
    return all(ax.fixedrange for ax in fig.select_xaxes()) and all(ax.fixedrange for ax in fig.select_yaxes())


def test_all_charts_lock_their_axes():
    grid = M.make_maze(0)
    a, b = E.live_pair(0, 10, 1, "hand_end", "aur_ae2")
    figs = [V.build_elite_map(grid, a.elite_ends, a.elite_quality), V.build_elite_map(grid, b.elite_ends, b.elite_quality), V.build_latent_map(b.elite_codes, b.space.centroids, b.elite_ends[:, 0]),
            V.build_end_cells_curve(a.end_cells, b.end_cells, "hand_end", "aur_ae2"), V.build_path_cells_curve(a.path_cells, b.path_cells, "hand_end", "aur_ae2"),
            *[V.build_end_cells_chart(PRE, lv) for lv in C.STUDY_LEVELS], *[V.build_path_cells_chart(PRE, lv) for lv in C.STUDY_LEVELS],
            *[V.build_study_curve(PRE, lv) for lv in C.STUDY_LEVELS], *[V.build_recon_scatter(PRE, lv) for lv in C.STUDY_LEVELS]]
    assert all(_locked(f) for f in figs)


def test_elite_map_has_one_marker_per_elite_coloured_by_quality_and_walls_in_the_maze_resolution():
    grid = M.make_maze(3)
    res = E.live_run(3, 10, 1, "hand_way4")
    fig = V.build_elite_map(grid, res.elite_ends, res.elite_quality)
    ends = next(t for t in fig.data if t.name == "Endpunkte der Elites")
    assert len(ends.x) == res.coverage[-1] and list(ends.marker.color) == list(res.elite_quality) and ends.marker.cmin == 0 and ends.marker.cmax == 1
    walls = np.array(fig.data[0].z).T
    assert (walls[grid] == 1.0).all() and np.isnan(walls[~grid]).all()


def test_latent_map_shows_centroids_and_elites_coloured_by_the_end_x():
    res = E.live_run(0, 10, 1, "fixed_pca2")
    fig = V.build_latent_map(res.elite_codes, res.space.centroids, res.elite_ends[:, 0])
    cen, eli = fig.data[0], fig.data[1]
    assert cen.name == "Schwerpunkte" and len(cen.x) == C.K and eli.name == "Elites" and len(eli.x) == res.coverage[-1] and list(eli.marker.color) == list(res.elite_ends[:, 0]) and eli.marker.cmax == C.N


def test_pair_curves_name_both_descriptions_and_the_colours_are_fixed():
    fig = V.build_end_cells_curve([1, 2], [3, 4], "hand_end", "aur_ae2")
    assert [t.name for t in fig.data] == [C.VARIANT_LABELS["hand_end"], C.VARIANT_LABELS["aur_ae2"]] and list(fig.data[0].y) == [1, 2] and list(fig.data[1].y) == [3, 4]
    assert fig.data[0].line.color == V.VARIANT_COLORS["hand_end"] and fig.data[1].line.color == V.VARIANT_COLORS["aur_ae2"]
    same = V.build_path_cells_curve([1, 2], [1, 2], "hand_style", "hand_style")
    assert same.data[1].line.dash == "dot"


def test_variant_colours_cover_every_variant():
    assert set(V.VARIANT_COLORS) == set(C.VARIANTS) and len(set(V.VARIANT_COLORS.values())) == len(C.VARIANTS)


@pytest.mark.parametrize("key,builder", [("end_cells", V.build_end_cells_chart), ("path_cells", V.build_path_cells_chart)])
def test_the_bar_charts_have_one_bar_per_variant_and_error_bars_equal_the_standard_deviation(key, builder):
    fig = builder(PRE, 3)
    assert [t.name for t in fig.data] == [C.VARIANT_LABELS[v] for v in C.VARIANTS]
    vals = np.array(E.study_cell(PRE, 3, "fixed_ae2")[key], float)
    t = fig.data[C.VARIANTS.index("fixed_ae2")]
    assert t.y[0] == pytest.approx(vals.mean()) and t.error_y.array[0] == pytest.approx(vals.std(ddof=1))


def test_study_curve_has_one_line_per_variant_over_the_evaluations():
    fig = V.build_study_curve(PRE, 0)
    assert [t.name for t in fig.data] == [C.VARIANT_LABELS[v] for v in C.VARIANTS] and fig.data[0].x[0] == C.POP and fig.data[0].x[-1] == C.STUDY_BUDGET


def test_the_recon_scatter_has_one_point_per_learned_variant_and_leaves_the_hand_made_out():
    fig = V.build_recon_scatter(PRE, 0)
    assert len(fig.data) == 8 and fig.layout.xaxis.type == "log"
    t = next(t for t in fig.data if t.name == C.VARIANT_LABELS["fixed_ae2"])
    assert t.x[0] == pytest.approx(np.mean(E.study_cell(PRE, 0, "fixed_ae2")["recon"])) and t.y[0] == pytest.approx(np.mean(E.study_cell(PRE, 0, "fixed_ae2")["end_cells"]))
