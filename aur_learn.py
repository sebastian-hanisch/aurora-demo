"""Gelernte Verhaltensbeschreibungen: PCA und ein Autoencoder von Grund auf (NumPy, Backpropagation von Hand, Adam, Vollbatch).

Beide bilden die **Rohdaten** einer Fahrt (20 Positionen, 40 Werte) auf einen **Code** mit 2 oder 8 Werten ab; der Code ist die gelernte Verhaltensbeschreibung.

* **PCA:** Mittelwert abziehen, die ersten d Hauptachsen (Singulärwertzerlegung) als Projektion. Der beste lineare Code im Sinn des quadratischen Rekonstruktionsfehlers (Eckart-Young).
* **Autoencoder:** Eingabe (40, standardisiert) → 16 (tanh) → d (linearer Engpass) → 16 (tanh) → 40; Verlust = mittlerer quadratischer Rekonstruktionsfehler, Xavier-Start mit festem Zufall, Adam mit Lernrate 0.01, 400 Schritte über alle Daten.
  Der Code ist die Ausgabe des Engpasses. Ein Autoencoder ohne versteckte Schicht wäre die PCA (Baldi und Hornik 1989); hier ist er nichtlinear."""

import numpy as np

HIDDEN = 16
EPOCHS = 400
LEARNING_RATE = 0.01
BETA1, BETA2, ADAM_EPS = 0.9, 0.999, 1e-8


class IdentityEncoder:
    """Handgemachte Beschreibung: die Merkmale selbst sind der Code (die Beschreibung steckt in der Merkmalsfunktion)."""
    kind = "identity"

    def __call__(self, X):
        return np.asarray(X, float)

    def arrays(self):
        return {}

    @classmethod
    def from_arrays(cls, arrays):
        return cls()


class PCAEncoder:
    kind = "pca"

    def __init__(self, mean, axes):
        self.mean, self.axes = mean, axes

    @classmethod
    def fit(cls, X, d):
        X = np.asarray(X, float)
        mean = X.mean(axis=0)
        _, _, vt = np.linalg.svd(X - mean, full_matrices=False)
        return cls(mean, vt[:d])

    def __call__(self, X):
        return (np.asarray(X, float) - self.mean) @ self.axes.T

    def reconstruct(self, X):
        return self(X) @ self.axes + self.mean

    def arrays(self):
        return {"mean": self.mean, "axes": self.axes}

    @classmethod
    def from_arrays(cls, arrays):
        return cls(arrays["mean"], arrays["axes"])


def layer_sizes(n_features, d, width=HIDDEN):
    return [n_features, width, d, width, n_features]


def init_parameters(sizes, seed):
    """Xavier-Start: Gewichte gleichverteilt in ±sqrt(6 / (Eingang + Ausgang)), Biases 0."""
    rng = np.random.default_rng(seed)
    weights, biases = [], []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        weights.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)))
        biases.append(np.zeros(fan_out))
    return weights, biases


def forward(weights, biases, Z):
    """Alle Schichtausgaben (Index 0 = Eingabe, Index 2 = Code). tanh nach der ersten und dritten Schicht, sonst linear."""
    outs = [Z]
    for i, (w, b) in enumerate(zip(weights, biases)):
        z = outs[-1] @ w + b
        outs.append(np.tanh(z) if i in (0, 2) else z)
    return outs


def loss_and_gradients(weights, biases, Z):
    """Mittlerer quadratischer Rekonstruktionsfehler (je Element) und seine Gradienten (Backpropagation von Hand)."""
    n, d = Z.shape
    outs = forward(weights, biases, Z)
    diff = outs[-1] - Z
    loss = float((diff ** 2).mean())
    delta = 2.0 * diff / (n * d)
    gw, gb = [None] * 4, [None] * 4
    for i in reversed(range(4)):
        gw[i] = outs[i].T @ delta
        gb[i] = delta.sum(axis=0)
        if i > 0:
            delta = delta @ weights[i].T
            if i - 1 in (0, 2):
                delta = delta * (1.0 - outs[i] ** 2)
    return loss, gw, gb


class AutoEncoder:
    kind = "ae"

    def __init__(self, weights, biases, mean, scale):
        self.weights, self.biases, self.mean, self.scale = weights, biases, mean, scale

    @classmethod
    def fit(cls, X, d, seed=0, epochs=EPOCHS, lr=LEARNING_RATE, width=HIDDEN):
        X = np.asarray(X, float)
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale = np.where(scale > 1e-9, scale, 1.0)
        Z = (X - mean) / scale
        weights, biases = init_parameters(layer_sizes(Z.shape[1], d, width), seed)
        params = weights + biases
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        for epoch in range(1, epochs + 1):
            _, gw, gb = loss_and_gradients(weights, biases, Z)
            lr_t = lr * np.sqrt(1.0 - BETA2 ** epoch) / (1.0 - BETA1 ** epoch)
            for k, g in enumerate(gw + gb):
                m[k] = BETA1 * m[k] + (1 - BETA1) * g
                v[k] = BETA2 * v[k] + (1 - BETA2) * g ** 2
                params[k] -= lr_t * m[k] / (np.sqrt(v[k]) + ADAM_EPS)
        return cls(weights, biases, mean, scale)

    def __call__(self, X):
        return forward(self.weights, self.biases, (np.asarray(X, float) - self.mean) / self.scale)[2]

    def reconstruct(self, X):
        return forward(self.weights, self.biases, (np.asarray(X, float) - self.mean) / self.scale)[-1] * self.scale + self.mean

    def loss(self, X):
        """Mittlerer quadratischer Rekonstruktionsfehler in standardisierten Einheiten."""
        Z = (np.asarray(X, float) - self.mean) / self.scale
        return loss_and_gradients(self.weights, self.biases, Z)[0]

    def arrays(self):
        out = {"mean": self.mean, "scale": self.scale}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            out[f"w{i}"], out[f"b{i}"] = w, b
        return out

    @classmethod
    def from_arrays(cls, arrays):
        return cls([arrays[f"w{i}"] for i in range(4)], [arrays[f"b{i}"] for i in range(4)], arrays["mean"], arrays["scale"])


def reconstruction_error(encoder, X):
    """Mittlerer quadratischer Rekonstruktionsfehler je Element in den Einheiten der Rohdaten (nur PCA und Autoencoder)."""
    X = np.asarray(X, float)
    return float(((encoder.reconstruct(X) - X) ** 2).mean())
