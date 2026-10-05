"""Merkmale, Encoder-Räume, Zellzuordnung und die gespeicherten Räume von Hand und gegen Brute-Force-Gegenproben."""

import numpy as np
import pytest

import aur_constants as C
import aur_evaluation as E
import aur_learn as L
import aur_maze as M
import aur_robot as R
import aur_space as SP

FREE = M.make_maze(0)


def slow_straight(speed=0.1, turn=0.0):
    """Steuerfolge: Tempo `speed`, Drehgen `turn` in jedem Schritt (Anfangsrichtung 45°); gibt (Steuerfolge, Fahrt) zurück."""
    g = np.zeros((1, 140, 2))
    g[:, :, 0], g[:, :, 1] = turn, speed
    return g, R.simulate(g, FREE, trajectories=True)[1]


def test_the_raw_features_of_a_slow_straight_ride_lie_at_equal_distances_by_hand():
    """Nach 7, 14, …, 140 Schritten ist der Roboter je 7 · 0.07 · cos 45° weiter; die Werte stehen als x, y, x, y, … in der Reihenfolge der Positionen."""
    _, track = slow_straight()
    raw = SP.raw_features(track)
    assert raw.shape == (1, 40) and raw[0] == pytest.approx([(3.5 + 7 * i * 0.07 * np.sqrt(0.5)) / C.N for i in range(1, 21) for _ in (0, 1)])


def test_hand_features_end_and_way4_by_hand():
    _, track = slow_straight()
    end = (3.5 + 140 * 0.07 * np.sqrt(0.5)) / C.N
    assert SP.hand_features("end", None, track)[0] == pytest.approx([end, end])
    assert SP.hand_features("way4", None, track)[0] == pytest.approx([(3.5 + n * 0.07 * np.sqrt(0.5)) / C.N for n in (35, 70, 105, 140) for _ in (0, 1)])


def test_the_style_features_by_hand():
    """Weglänge 140 · 0.07 = 9.8, größtmöglich 140 · 0.7 = 98: 0.1; mittleres Drehgen 0 → 0.5. Volle Drehung (Gen +1) auf der Stelle: Länge 0, Drehung 1; Stillstand und nur Drehgen −1: Drehung 0."""
    g, track = slow_straight()
    assert SP.hand_features("style", g, track)[0] == pytest.approx([0.1, 0.5])
    g2, t2 = slow_straight(speed=0.0, turn=1.0)
    assert SP.hand_features("style", g2, t2)[0] == pytest.approx([0.0, 1.0])
    g3, t3 = slow_straight(speed=0.0, turn=-1.0)
    assert SP.hand_features("style", g3, t3)[0] == pytest.approx([0.0, 0.0])
    g4, t4 = slow_straight(speed=1.0)
    assert SP.hand_features("style", g4, t4)[0][0] <= 1.0


def test_the_style_features_say_nothing_about_the_place():
    """Zwei Fahrten mit gleicher Weglänge und gleichem mittleren Drehgen, aber an verschiedenen Orten: die eine fährt geradeaus, die andere dreht 70 Schritte nach links und 70 nach rechts (Drehung gleich 0, sie bleibt in Startnähe)."""
    g = np.zeros((2, 140, 2))
    g[:, :, 1] = 0.1
    g[1, :70, 0], g[1, 70:, 0] = 1.0, -1.0
    track = R.simulate(g, FREE, trajectories=True)[1]
    style = SP.hand_features("style", g, track)
    assert style[0] == pytest.approx(style[1]) and np.linalg.norm(track[0, -1] - track[1, -1]) > 5.0


def test_the_feature_dispatch_follows_the_variant():
    g, track = slow_straight()
    assert SP.features("hand_end", g, track).shape == (1, 2) and SP.features("hand_way4", g, track).shape == (1, 8) and SP.features("hand_style", g, track).shape == (1, 2)
    for v in C.VARIANTS[3:]:
        assert SP.features(v, g, track).shape == (1, 40)


def test_features_are_in_the_unit_square_and_the_clipping_works():
    track = np.tile(np.array([30.0, -3.0]), (1, 21, 1))
    assert list(SP.hand_features("end", None, track)[0]) == [1.0, 0.0] and list(SP.raw_features(track)[0][:2]) == [1.0, 0.0]


def test_random_features_are_reproducible_and_depend_on_the_seed_and_the_maze():
    a = SP.random_features("hand_way4", FREE, 200, 5)
    assert a.shape == (200, 8) and (a == SP.random_features("hand_way4", FREE, 200, 5)).all() and not (a == SP.random_features("hand_way4", FREE, 200, 6)).all()
    assert not (a == SP.random_features("hand_way4", M.make_maze(3), 200, 5)).all()


def test_k_means_returns_k_finite_centroids_and_is_reproducible_and_rejects_too_few_points():
    Z = np.random.default_rng(1).normal(size=(500, 3))
    c = SP.kmeans_centroids(Z, 20)
    assert c.shape == (20, 3) and np.isfinite(c).all() and (c == SP.kmeans_centroids(Z, 20)).all()
    with pytest.raises(ValueError):
        SP.kmeans_centroids(Z[:10], 20)


def test_k_means_survives_identical_points():
    c = SP.kmeans_centroids(np.zeros((300, 2)), 10)
    assert c.shape == (10, 2) and np.isfinite(c).all()


def test_the_cell_assignment_is_the_brute_force_nearest_centroid_in_the_standardised_code():
    X = np.random.default_rng(2).normal(size=(400, 40))
    for method, d in (("pca", 2), ("pca", 8), ("identity", 40)):
        space = SP.fit_space(method, d, X)
        Z = np.random.default_rng(3).normal(size=(120, 40))
        code = space.code(Z)
        brute = np.array([np.argmin(((space.centroids - z) ** 2).sum(axis=1)) for z in code])
        assert space.dim == (d if method == "pca" else 40) and (space.assign(Z) == brute).all() and space.centroids.shape == (C.K, space.dim)


def test_fitted_codes_are_standardised_on_the_training_data():
    X = np.random.default_rng(4).normal(size=(600, 40)) * np.linspace(1, 5, 40)
    code = SP.fit_space("pca", 4, X).code(X)
    assert code.mean(axis=0) == pytest.approx(np.zeros(4), abs=1e-9) and code.std(axis=0) == pytest.approx(np.ones(4), abs=1e-9)


def test_the_standardisation_makes_a_narrow_feature_count_as_much_as_a_wide_one():
    """Fahrstil: die Drehung schwankt bei Zufallsfahrten nur um ±0.025, die Länge viel stärker; nach der Standardisierung haben beide gleiche Streuung, die Zellen teilen also beide Richtungen."""
    X = SP.random_features("hand_style", FREE, 3000, 1)
    assert X[:, 1].std() < 0.3 * X[:, 0].std() or X[:, 1].std() < 0.1
    code = SP.fit_space("identity", 2, X).code(X)
    assert code.std(axis=0) == pytest.approx([1.0, 1.0], abs=1e-9)


@pytest.mark.parametrize("method,d", [("pca", 2), ("ae", 2), ("identity", 40)])
def test_a_space_survives_the_array_round_trip(method, d):
    X = np.random.default_rng(5).normal(size=(400, 40))
    space = SP.fit_space(method, d, X)
    again = SP.Space.from_arrays(method, space.arrays())
    Z = np.random.default_rng(6).normal(size=(50, 40))
    assert (space.assign(Z) == again.assign(Z)).all()


def test_the_variant_tables_are_consistent():
    assert len(C.VARIANTS) == 11 and set(C.VARIANT_LABELS) == set(C.VARIANT_MODE) == set(C.VARIANT_METHOD) == set(C.VARIANT_DIM) == set(C.VARIANTS)
    assert [SP.variant_method(v) for v in C.VARIANTS] == ["identity"] * 3 + ["pca", "pca", "ae", "ae"] * 2 and [C.VARIANT_DIM[v] for v in C.VARIANTS] == [2, 8, 2, 2, 8, 2, 8, 2, 8, 2, 8]
    assert set(C.VARIANT_MODE.values()) == set(C.GROUP_LABELS) and C.STORED_VARIANTS == C.VARIANTS[:7]


def test_the_stored_spaces_cover_every_stored_variant_and_level_and_have_the_right_shape():
    for lv in C.STUDY_LEVELS:
        for v in C.STORED_VARIANTS:
            space = E.get_space(v, lv)
            assert space.centroids.shape == (C.K, C.VARIANT_DIM[v]) and np.isfinite(space.centroids).all()
            X = SP.random_features(v, M.make_maze(lv), 100, 99)
            assert space.assign(X).shape == (100,) and space.assign(X).max() < C.K
    with pytest.raises(ValueError):
        E.get_space("aur_ae2", 0)


def test_a_stored_pca_space_is_what_a_fresh_training_gives():
    """PCA hängt nur von der Singulärzerlegung ab: bis auf das Vorzeichen der Achsen dieselben Achsen, dieselbe Standardisierung."""
    stored, fresh = E.get_space("fixed_pca2", 0), SP.train_space("fixed_pca2", 0)
    cos = np.abs(np.diag(stored.encoder.axes @ fresh.encoder.axes.T))
    assert cos == pytest.approx([1.0, 1.0], abs=1e-6) and stored.sd == pytest.approx(fresh.sd, rel=1e-6)


def test_a_stored_autoencoder_space_is_as_good_as_a_fresh_training():
    """BLAS rechnet je Plattform minimal anders und das Training verstärkt das: die Neurechnung erreicht einen Rekonstruktionsfehler innerhalb von 25 % des gespeicherten, nicht dieselben Gewichte."""
    stored, fresh = E.get_space("fixed_ae2", 0), SP.train_space("fixed_ae2", 0)
    X = SP.random_features("fixed_ae2", FREE, 2000, 4242)
    assert L.reconstruction_error(fresh.encoder, X) == pytest.approx(L.reconstruction_error(stored.encoder, X), rel=0.25)
