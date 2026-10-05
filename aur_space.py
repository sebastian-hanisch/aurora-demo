"""Verhaltensräume: Merkmale einer Fahrt (handgemacht oder Rohdaten), Encoder (Identität, PCA, Autoencoder) und die Zellen im Code (CVT mit K Schwerpunkten aus einem k-Means auf den Codes der Trainingsdaten).

Ein **Raum** ist alles, was aus Merkmalen eine Zelle macht: Code = Encoder(Merkmale), standardisiert mit Mittel und Standardabweichung der Trainingscodes; Zelle = nächster der K Schwerpunkte (euklidisch, KD-Baum). Handgemachte Beschreibungen haben die Identität als Encoder."""

import warnings

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

import aur_constants as C
import aur_learn as L
import aur_maze as M
import aur_robot as R


def raw_features(track):
    """Rohdaten für die gelernten Beschreibungen: die Positionen nach jedem 20. Teil der Fahrt (Schritt = Schritte · i / 20 abgerundet), durch N geteilt, auf [0, 1] begrenzt; Form (P, 40). `track` hat die Form (P, Schritte + 1, 2)."""
    steps = track.shape[1] - 1
    idx = [int(steps * i / C.RAW_POINTS) for i in range(1, C.RAW_POINTS + 1)]
    return np.clip(track[:, idx, :].reshape(len(track), -1) / C.N, 0.0, 1.0)


def hand_features(kind, genomes, track):
    """Handgemachte Beschreibung. `end`: Endposition / N (2 Werte). `way4`: Positionen nach 1/4, 2/4, 3/4, 4/4 der Fahrt / N (8 Werte). `style`: Weglänge geteilt durch die größtmögliche Weglänge und mittleres Drehgen
    verschoben auf [0, 1] (2 Werte); sie sagt nichts über den Ort, an dem die Fahrt endet."""
    if kind == "style":
        driven = np.linalg.norm(np.diff(track, axis=1), axis=2).sum(axis=1)
        return np.column_stack([np.clip(driven / C.STYLE_MAX_LENGTH, 0.0, 1.0), (np.asarray(genomes)[:, :, 0].mean(axis=1) + 1.0) / 2.0])
    parts = C.HAND_PARTS[kind]
    steps = track.shape[1] - 1
    idx = [int(steps * i / parts) for i in range(1, parts + 1)]
    return np.clip(track[:, idx, :].reshape(len(track), -1) / C.N, 0.0, 1.0)


def features(variant, genomes, track):
    """Merkmale, die der Encoder der Variante bekommt: handgemachte Beschreibung bzw. Rohdaten."""
    if C.VARIANT_MODE[variant] == "hand":
        return hand_features(C.VARIANT_METHOD[variant], genomes, track)
    return raw_features(track)


def random_features(variant, grid, n, seed, batch=100):
    """Merkmale von `n` zufälligen Steuerfolgen im Labyrinth `grid` (in Gruppen zu `batch` gefahren)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n // batch):
        genomes = R.random_genomes(rng, batch)
        out.append(features(variant, genomes, R.simulate(genomes, grid, trajectories=True)[1]))
    return np.vstack(out)


def kmeans_centroids(Z, k=None):
    """`k` Schwerpunkte per k-Means (k-means++-Start, feste Iterationszahl, fester Zufall). Leere Cluster behalten ihren alten Schwerpunkt; NaN aus Duplikaten im Start werden durch Datenpunkte ersetzt."""
    k = C.K if k is None else k
    if len(Z) < k:
        raise ValueError(f"zu wenige Punkte für {k} Schwerpunkte: {len(Z)}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        centroids, _ = kmeans2(Z, k, minit="++", seed=C.KMEANS_SEED, iter=C.KMEANS_ITER, missing="warn")
    bad = ~np.isfinite(centroids).all(axis=1)
    if bad.any():
        centroids[bad] = Z[np.random.default_rng(C.KMEANS_SEED).choice(len(Z), int(bad.sum()), replace=False)]
    return centroids


class Space:
    """Ein Verhaltensraum mit K Zellen: `assign(Merkmale)` gibt die Zellnummer je Fahrt zurück."""

    def __init__(self, encoder, mu, sd, centroids):
        self.encoder, self.mu, self.sd, self.centroids = encoder, mu, sd, centroids
        self.tree = cKDTree(centroids)

    def code(self, X):
        """Standardisierter Code der Merkmale."""
        return (self.encoder(X) - self.mu) / self.sd

    def assign(self, X):
        return self.tree.query(self.code(X))[1].astype(np.int64)

    @property
    def dim(self):
        return self.centroids.shape[1]

    def arrays(self):
        return {"mu": self.mu, "sd": self.sd, "centroids": self.centroids, **{f"enc_{k}": v for k, v in self.encoder.arrays().items()}}

    @classmethod
    def from_arrays(cls, method, arrays):
        enc_cls = {"pca": L.PCAEncoder, "ae": L.AutoEncoder}.get(method, L.IdentityEncoder)
        enc = enc_cls.from_arrays({k[4:]: v for k, v in arrays.items() if k.startswith("enc_")})
        return cls(enc, arrays["mu"], arrays["sd"], arrays["centroids"])


def fit_space(method, d, X, seed=0):
    """Lernt den Encoder (`method`: pca, ae oder identity) auf den Merkmalen `X` und legt K Zellen in den standardisierten Code: Mittel und Standardabweichung der Trainingscodes, k-Means-Schwerpunkte der standardisierten Codes."""
    if method == "pca":
        enc = L.PCAEncoder.fit(X, d)
    elif method == "ae":
        enc = L.AutoEncoder.fit(X, d, seed=seed)
    else:
        enc = L.IdentityEncoder()
    codes = enc(X)
    mu = codes.mean(axis=0)
    sd = np.where(codes.std(axis=0) > 1e-9, codes.std(axis=0), 1.0)
    return Space(enc, mu, sd, kmeans_centroids((codes - mu) / sd))


def variant_method(variant):
    """Encoder-Art einer Variante: `identity` für handgemachte Beschreibungen, sonst `pca` oder `ae`."""
    return "identity" if C.VARIANT_MODE[variant] == "hand" else C.VARIANT_METHOD[variant]


def train_space(variant, level, grid=None):
    """Der Raum einer gespeicherten Variante (hand_*, fixed_*), neu gerechnet: Merkmale von Zufallsfahrten (30 000 bei handgemachten, 10 000 bei einmal gelernten), Encoder lernen, Schwerpunkte legen."""
    grid = M.make_maze(level) if grid is None else grid
    hand = C.VARIANT_MODE[variant] == "hand"
    X = random_features(variant, grid, C.CENTROID_SAMPLES if hand else C.TRAIN_RIDES, C.CENTROID_SEED if hand else C.TRAIN_SEED)
    return fit_space(variant_method(variant), C.VARIANT_DIM[variant], X)
