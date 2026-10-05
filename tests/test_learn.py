"""PCA und Autoencoder von Hand und gegen unabhängige Gegenproben (scikit-learn, Differenzenquotienten)."""

import numpy as np
import pytest
from sklearn.decomposition import PCA

import aur_learn as L


def line_data(n=200, seed=0):
    t = np.random.default_rng(seed).normal(size=n)
    return np.column_stack([t, 2 * t]) + 5.0


def test_pca_of_points_on_a_line_finds_the_line_and_reconstructs_exactly_by_hand():
    X = line_data()
    enc = L.PCAEncoder.fit(X, 1)
    assert abs(float(enc.axes[0] @ np.array([1.0, 2.0]) / np.sqrt(5.0))) == pytest.approx(1.0) and enc.mean == pytest.approx(X.mean(axis=0))
    assert L.reconstruction_error(enc, X) == pytest.approx(0.0, abs=1e-20)
    assert enc(X).shape == (200, 1) and enc(X).std() == pytest.approx(np.sqrt(5.0) * np.std(X[:, 0]) , rel=1e-9)


def test_pca_agrees_with_scikit_learn_in_subspace_and_in_reconstruction_error():
    X = np.random.default_rng(1).normal(size=(300, 12)) @ np.random.default_rng(2).normal(size=(12, 12))
    for d in (2, 5):
        ours, ref = L.PCAEncoder.fit(X, d), PCA(n_components=d).fit(X)
        cos = np.linalg.svd(ours.axes @ ref.components_.T, compute_uv=False)
        assert cos == pytest.approx(np.ones(d), abs=1e-9)
        assert L.reconstruction_error(ours, X) == pytest.approx(float(((ref.inverse_transform(ref.transform(X)) - X) ** 2).mean()), rel=1e-9)
        assert ours(X).var(axis=0, ddof=1) == pytest.approx(ref.explained_variance_, rel=1e-9)


def test_pca_codes_are_uncorrelated_and_sorted_by_variance():
    X = np.random.default_rng(3).normal(size=(500, 6)) * np.array([5, 4, 3, 2, 1, 0.5])
    codes = L.PCAEncoder.fit(X, 4)(X)
    cov = np.cov(codes.T)
    assert np.abs(cov - np.diag(np.diag(cov))).max() < 1e-9 and list(np.diag(cov)) == sorted(np.diag(cov), reverse=True)


def test_pca_with_all_axes_reconstructs_without_error():
    X = np.random.default_rng(4).normal(size=(50, 5))
    assert L.reconstruction_error(L.PCAEncoder.fit(X, 5), X) == pytest.approx(0.0, abs=1e-20)


def test_the_gradient_agrees_with_finite_differences():
    rng = np.random.default_rng(5)
    Z = rng.normal(size=(7, 6))
    weights, biases = L.init_parameters(L.layer_sizes(6, 2, width=4), seed=1)
    for b in biases:
        b += rng.normal(size=b.shape) * 0.1
    loss, gw, gb = L.loss_and_gradients(weights, biases, Z)
    assert loss == pytest.approx(float(((L.forward(weights, biases, Z)[-1] - Z) ** 2).mean()))
    eps = 1e-6
    for params, grads in ((weights, gw), (biases, gb)):
        for p, g in zip(params, grads):
            for index in [tuple(rng.integers(0, s) for s in p.shape) for _ in range(5)]:
                old = p[index]
                p[index] = old + eps
                up = L.loss_and_gradients(weights, biases, Z)[0]
                p[index] = old - eps
                down = L.loss_and_gradients(weights, biases, Z)[0]
                p[index] = old
                assert g[index] == pytest.approx((up - down) / (2 * eps), rel=1e-5, abs=1e-9)


def test_the_layer_sizes_and_the_initialisation_by_hand():
    assert L.layer_sizes(40, 8) == [40, 16, 8, 16, 40]
    w, b = L.init_parameters(L.layer_sizes(40, 2), seed=0)
    assert [x.shape for x in w] == [(40, 16), (16, 2), (2, 16), (16, 40)] and all(np.abs(x).max() <= np.sqrt(6 / (a + c)) for x, (a, c) in zip(w, [(40, 16), (16, 2), (2, 16), (16, 40)])) and all((x == 0).all() for x in b)
    assert all((x == y).all() for x, y in zip(w, L.init_parameters(L.layer_sizes(40, 2), seed=0)[0])) and not (w[0] == L.init_parameters(L.layer_sizes(40, 2), seed=1)[0][0]).all()


def curved_data(n=400, seed=0):
    """Ein Kreisbogen in 3 Dimensionen (eine gekrümmte Linie): ein Wert genügt für eine gute Beschreibung, aber nicht linear."""
    t = np.random.default_rng(seed).uniform(0, np.pi, n)
    return np.column_stack([np.cos(t), np.sin(t), 0.2 * t]) + np.random.default_rng(seed + 1).normal(0, 0.01, (n, 3))


def test_the_autoencoder_learns_a_curve_that_pca_cannot_with_one_code_value():
    X = curved_data()
    ae, pca = L.AutoEncoder.fit(X, 1, epochs=1500), L.PCAEncoder.fit(X, 1)
    assert L.reconstruction_error(ae, X) < 0.35 * L.reconstruction_error(pca, X)


def test_training_lowers_the_loss_and_is_deterministic_for_a_seed_but_not_across_seeds():
    X = np.random.default_rng(6).normal(size=(200, 10)) @ np.random.default_rng(7).normal(size=(10, 10))
    untrained = L.AutoEncoder.fit(X, 2, epochs=0)
    a, b, c = (L.AutoEncoder.fit(X, 2, epochs=100, seed=s) for s in (0, 0, 1))
    assert a.loss(X) < 0.9 * untrained.loss(X)
    assert all((x == y).all() for x, y in zip(a.weights, b.weights)) and not (a.weights[0] == c.weights[0]).all()


def test_the_loss_is_the_mean_squared_error_in_standardised_units_by_hand():
    X = np.array([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]])
    ae = L.AutoEncoder.fit(X, 1, epochs=0)
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    assert ae.loss(X) == pytest.approx(float(((L.forward(ae.weights, ae.biases, Z)[-1] - Z) ** 2).mean()))


@pytest.mark.parametrize("make", [lambda X: L.PCAEncoder.fit(X, 3), lambda X: L.AutoEncoder.fit(X, 3, epochs=20), lambda X: L.IdentityEncoder()])
def test_the_arrays_round_trip_gives_the_same_encoder(make):
    X = np.random.default_rng(8).normal(size=(60, 7))
    enc = make(X)
    again = type(enc).from_arrays(enc.arrays())
    assert (enc(X) == again(X)).all()


def test_the_identity_encoder_returns_the_features():
    X = np.random.default_rng(9).normal(size=(5, 3))
    assert (L.IdentityEncoder()(X) == X).all() and L.IdentityEncoder().arrays() == {}


def test_the_code_has_the_requested_number_of_values_and_a_constant_feature_does_not_break_the_standardisation():
    X = np.random.default_rng(10).normal(size=(80, 6))
    X[:, 2] = 3.0
    ae = L.AutoEncoder.fit(X, 2, epochs=10)
    assert ae(X).shape == (80, 2) and np.isfinite(ae(X)).all() and ae.scale[2] == 1.0
