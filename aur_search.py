"""MAP-Elites mit einer handgemachten oder gelernten Verhaltensbeschreibung, und die AURORA-Schleife.

**Archiv:** K = 256 Zellen eines Verhaltensraums (`aur_space.Space`); je Zelle die Fahrt mit der höchsten **Qualität** = Direktheit (Luftlinie Start–Ende geteilt durch die gefahrene Weglänge, 0 bis 1); ein Kind ersetzt den Elite seiner Zelle nur,
wenn es um mehr als `QUALITY_TIE` besser ist (oder die Zelle leer war). Auswahl der Eltern und Mutation wie in Stück 5 und 7: je Generation 100 Eltern gleichverteilt aus den belegten Zellen, Mutation Gauß 0.25 auf 15 % der Gene.

**Varianten:** `hand_*`: handgemachte Beschreibung, der Raum ist fest (vorgerechnet). `fixed_*`: auf 10 000 Zufallsfahrten gelernter Raum, fest (vorgerechnet). `aur_*`: die **AURORA-Schleife**: die ersten 5 Generationen sind Zufallsfahrten (500), aus ihren Rohdaten wird die erste
Beschreibung gelernt und alle 500 Fahrten werden eingetragen; in den Generationen 10, 30, 60, 100, 150 und 250 wird neu gelernt (Trainingsdaten: die 500 Zufallsfahrten, alle Elites und die Kinder dieser Generation), die Elites werden nach Qualität
absteigend neu eingetragen (kollidieren zwei in einer Zelle, bleibt die bessere), dann die Kinder.

**Kennzahlen** (alle nach jeder Generation, unabhängig von der Beschreibung): belegte Zellen, QD-Score (Summe der Qualität; bei gleicher Zellzahl K vergleichbar), **verschiedene 2 × 2-Endzellen** der Elites (von 144), **Wegzellen** (2 × 2-Zellen, die irgendeine Elite-Fahrt berührt, von 144),
Erfolg (Endposition im Zielkreis) und wahrer Restweg. Bis zur ersten gelernten Beschreibung (Generationen 1 bis 4 der AURORA-Läufe) gibt es kein Archiv: belegte Zellen, QD-Score, Endzellen und Wegzellen sind dort 0.

Zufall nur über `numpy.random.default_rng(seed)` (PCG64, plattformstabil); die gelernten Räume (Autoencoder, Eigen- bzw. Singulärzerlegung) sind es nicht ganz (BLAS, LAPACK je Plattform)."""

from dataclasses import dataclass

import numpy as np

import aur_constants as C
import aur_learn as L
import aur_maze as M
import aur_robot as R
import aur_space as SP

QUALITY_TIE = 1e-12                          # Toleranz beim Ersetzen eines Elites (Gleichstände bis auf Rundung ersetzen nicht, auf jeder Plattform gleich)
BLOCKS = C.N // C.PATH_CELL                  # 2 × 2-Zellen je Achse (12)


@dataclass
class SearchResult:
    coverage: list                           # belegte Zellen nach jeder Generation
    qd_score: list                           # Summe der Qualität über die belegten Zellen nach jeder Generation
    end_cells: list                          # verschiedene 2 × 2-Endzellen der Elites nach jeder Generation
    path_cells: list                         # 2 × 2-Zellen, die irgendeine Elite-Fahrt berührt, nach jeder Generation
    best_quality: list                       # beste Qualität nach jeder Generation
    best_progress: list                      # kleinster wahrer Restweg (Zellen) aller bisher besuchten Zellen nach jeder Generation
    first_solved: object                     # Zahl der Evaluationen bis zum ersten Individuum im Zielkreis (None: nie)
    space: object = None                     # der Verhaltensraum am Ende des Laufs
    elite_ends: object = None                # Endpositionen der Elites (E, 2), wenn gespeichert
    elite_quality: object = None             # Qualität der Elites (E,), wenn gespeichert
    elite_codes: object = None               # standardisierte Codes der Elites (E, d), wenn gespeichert
    retrains: int = 0                        # wie oft die Beschreibung neu gelernt wurde (ohne die erste)
    recon: object = None                     # mittlerer quadratischer Rekonstruktionsfehler des Encoders auf den Rohdaten der Elites (nur PCA und Autoencoder, sonst None)
    evaluations: int = 0


def directness(track):
    """Qualität jeder Fahrt: Luftlinie Start–Ende geteilt durch die gefahrene Weglänge (0, wenn der Roboter gar nicht gefahren ist). `track` hat die Form (P, Schritte + 1, 2)."""
    driven = np.linalg.norm(np.diff(track, axis=1), axis=2).sum(axis=1)
    straight = np.linalg.norm(track[:, -1, :] - np.array(C.START), axis=1)
    return np.where(driven > 1e-9, straight / np.maximum(driven, 1e-9), 0.0)


def block_index(points):
    """Nummer der 2 × 2-Zelle jedes Punktes (i · 12 + j, Form (...,)), Punkte außerhalb auf den Rand begrenzt."""
    ij = np.clip(np.floor(np.asarray(points) / C.PATH_CELL).astype(int), 0, BLOCKS - 1)
    return ij[..., 0] * BLOCKS + ij[..., 1]


def path_masks(track):
    """Welche der 144 Zellen berührt jede Fahrt: Wahrheitsfeld (P, 144) über alle Punkte der Fahrt."""
    flat = block_index(track)
    masks = np.zeros((len(track), BLOCKS * BLOCKS), bool)
    masks[np.arange(len(track))[:, None], flat] = True
    return masks


def reinsertion_plan(order, new_cells):
    """Neueintragen eines Archivs in einen neuen Raum: `order` sind die alten Zellen nach Qualität absteigend, `new_cells` ihre neuen Zellen. Gibt {neue Zelle: alte Zelle} zurück; kollidieren zwei, bleibt die frühere (bessere)."""
    plan = {}
    for old, new in zip(order, new_cells):
        plan.setdefault(int(new), int(old))
    return plan


class Archive:
    """Archiv mit K Zellen: Qualität (−1 = leer), Steuerfolge, Merkmale, Endposition und berührte Zellen je Zelle."""

    def __init__(self, steps, n_features, k=None):
        k = C.K if k is None else k
        self.k = k
        self.quality = np.full(k, -1.0)
        self.genomes = np.zeros((k, steps, 2))
        self.feats = np.zeros((k, n_features))
        self.ends = np.zeros((k, 2))
        self.masks = np.zeros((k, BLOCKS * BLOCKS), bool)

    @property
    def filled(self):
        return np.nonzero(self.quality >= 0)[0]

    def insert(self, cells, quality, genomes, feats, ends, masks):
        """Kinder der Reihe nach eintragen: leere Zelle oder um mehr als QUALITY_TIE besser."""
        for i in range(len(cells)):
            c = int(cells[i])
            if self.quality[c] < 0 or quality[i] > self.quality[c] + QUALITY_TIE:
                self.quality[c], self.genomes[c], self.feats[c], self.ends[c], self.masks[c] = quality[i], genomes[i], feats[i], ends[i], masks[i]

    def rebuilt(self, space):
        """Dasselbe Archiv in einem neuen Raum: Elites nach Qualität absteigend neu eintragen (Gleichstand: kleinere alte Zelle zuerst), bei Kollision gewinnt der bessere."""
        old = self.filled
        order = old[np.argsort(-self.quality[old], kind="stable")]
        plan = reinsertion_plan(order, space.assign(self.feats[order]))
        new = Archive(self.genomes.shape[1], self.feats.shape[1], self.k)
        for c_new, c_old in plan.items():
            new.quality[c_new], new.genomes[c_new], new.feats[c_new], new.ends[c_new], new.masks[c_new] = self.quality[c_old], self.genomes[c_old], self.feats[c_old], self.ends[c_old], self.masks[c_old]
        return new

    def metrics(self):
        f = self.filled
        if len(f) == 0:
            return 0, 0.0, 0, 0
        return len(f), float(self.quality[f].sum()), len(np.unique(block_index(self.ends[f]))), int(self.masks[f].any(axis=0).sum())


def run_search(grid, seed, variant, space=None, gens=C.STUDY_GENS, pop=C.POP, steps=C.STEPS, keep_archive=False):
    """Ein Lauf. `hand_*` und `fixed_*` brauchen den vorgerechneten `space`; `aur_*` lernt ihn im Lauf (`space` muss None sein)."""
    mode, method, d = C.VARIANT_MODE[variant], SP.variant_method(variant), C.VARIANT_DIM[variant]
    if (mode == "aur") != (space is None):
        raise ValueError("aur_* lernt den Raum im Lauf, hand_* und fixed_* brauchen ihn")
    rng = np.random.default_rng(seed)
    n = grid.shape[0]
    true_path = M.bfs_distances(grid, M.cell_of(C.GOAL))
    res = SearchResult(coverage=[], qd_score=[], end_cells=[], path_cells=[], best_quality=[], best_progress=[], first_solved=None)
    arch = None
    warm = []                                                    # AURORA: (Qualität, Steuerfolge, Merkmale, Endposition, Wege) der Aufwärmfahrten
    genomes = R.random_genomes(rng, pop, steps)
    best_q, best_p = -1.0, np.inf
    for gen in range(gens):
        end, track = R.simulate(genomes, grid, trajectories=True)
        dist = R.distance_to_goal(end)
        res.evaluations += pop
        if res.first_solved is None and dist.min() < C.SUCCESS_RADIUS:
            res.first_solved = gen * pop + int(np.argmax(dist < C.SUCCESS_RADIUS)) + 1
        idx = np.clip(end.astype(int), 0, n - 1)
        best_p = min(best_p, float(true_path[idx[:, 0], idx[:, 1]].min()))
        q = directness(track)
        best_q = max(best_q, float(q.max()))
        feats = SP.features(variant, genomes, track)
        masks = path_masks(track)
        if mode != "aur":
            if arch is None:
                arch = Archive(steps, feats.shape[1])
            arch.insert(space.assign(feats), q, genomes, feats, end, masks)
        elif gen < C.WARMUP_GENS - 1:
            warm.append((q, genomes, feats, end, masks))
        elif gen == C.WARMUP_GENS - 1:
            warm.append((q, genomes, feats, end, masks))
            wq, wg, wf, we, wm = (np.concatenate([w[i] for w in warm]) for i in range(5))
            space = SP.fit_space(method, d, wf)
            arch = Archive(steps, wf.shape[1])
            arch.insert(space.assign(wf), wq, wg, wf, we, wm)
            warm_feats = wf
        else:
            if gen in C.RETRAIN_GENS:
                f = arch.filled
                space = SP.fit_space(method, d, np.vstack([warm_feats, arch.feats[f], feats]))
                arch = arch.rebuilt(space)
                res.retrains += 1
            arch.insert(space.assign(feats), q, genomes, feats, end, masks)
        cov, qd, ends, paths = arch.metrics() if arch is not None else (0, 0.0, 0, 0)
        res.coverage.append(cov)
        res.qd_score.append(qd)
        res.end_cells.append(ends)
        res.path_cells.append(paths)
        res.best_quality.append(best_q)
        res.best_progress.append(best_p)
        if arch is None or len(arch.filled) == 0 or (mode == "aur" and gen < C.WARMUP_GENS - 1):
            genomes = R.random_genomes(rng, pop, steps)
            continue
        keys = arch.filled
        pick = rng.integers(0, len(keys), pop)
        parents = arch.genomes[keys[pick]]
        mask = rng.random(parents.shape) < C.MUT_RATE
        genomes = R.clip_genomes(parents + rng.normal(0.0, C.MUT_SIGMA, parents.shape) * mask)
    res.space = space
    if keep_archive and arch is not None:
        f = arch.filled
        res.elite_ends, res.elite_quality, res.elite_codes = arch.ends[f].copy(), arch.quality[f].copy(), space.code(arch.feats[f])
    if arch is not None and mode != "hand" and len(arch.filled):
        res.recon = L.reconstruction_error(space.encoder, arch.feats[arch.filled])
    return res
