"""Konstanten der AURORA-Demo: Labyrinth und Roboter wie in Stück 1 bis 7, Verhaltensbeschreibungen (handgemacht und gelernt), Studien-Achsen, Regler, Voreinstellungen.
Längen in Zellen (Zellraster N × N, Zelle (i, j) belegt den Bereich [i, i + 1) × [j, j + 1))."""


def fmt_int(n):
    """Ganzzahl mit Leerzeichen als Tausendertrenner (10000 -> 10 000), wie in der README."""
    return f"{n:,}".replace(",", " ")


def fmt_med(x):
    """Median der Evaluationen: ganze Zahl mit Leerzeichen als Tausendertrenner, ein halber Wert behält seine Stelle (988.5, 1 308.5); None = kein Lauf war erfolgreich."""
    if x is None:
        return "–"
    return fmt_int(int(x)) if x == int(x) else f"{x:,.1f}".replace(",", " ")


N = 24                                       # Kantenlänge des Labyrinths in Zellen (Rand ist Wand)
START = (3.5, 3.5)                           # Mitte der Startzelle (3, 3)
GOAL = (3.5, 20.5)                           # Mitte der Zielzelle (3, 20)
SUCCESS_RADIUS = 2.0                         # Erfolg: Endposition höchstens so weit vom Ziel
START_HEADING = 0.7853981633974483           # Anfangsrichtung π/4
STEPS = 140                                  # Länge der Steuerfolge
TURN_MAX = 0.6                               # Drehung je Schritt (rad), Gen ∈ [−1, 1]
SPEED_MAX = 0.7                              # Fahrstrecke je Schritt (Zellen), Gen ∈ [0, 1]

POP = 100                                    # Kinder je Generation
MUT_SIGMA = 0.25                             # Standardabweichung der Mutation je verändertem Gen
MUT_RATE = 0.15                              # Anteil der Gene, die je Kind mutieren

LEVELS = (0, 1, 2, 3, 4)
LEVEL_LABELS = {0: "0: freies Feld", 1: "1: Öffnung links", 2: "2: Öffnung in der Mitte", 3: "3: Öffnung rechts", 4: "4: Schlangenlinie"}
GAP = 3                                      # Breite einer Öffnung in der Wand (Zellen)
STUDY_LEVELS = (0, 3)                        # Stufen der Studie und des Live-Laufs: freies Feld und die Falle (Räume und Schwerpunkte sind je Stufe vorgerechnet)
STUDY_LEVEL_LABELS = {0: "freies Feld", 3: "Stufe 3 (Falle)"}

# Rohdaten einer Fahrt für die gelernten Beschreibungen: 20 Positionen (nach jedem 20. Teil der Fahrt), durch N geteilt: 40 Werte
RAW_POINTS = 20
RAW_DIM = 2 * RAW_POINTS

# Handgemachte Beschreibungen: Endposition (2 Werte), Positionen nach jedem Viertel (8 Werte), Fahrstil (Weglänge und Gesamtdrehung, 2 Werte: sagt nichts über den Ort)
HAND_KINDS = ("end", "way4", "style")
HAND_PARTS = {"end": 1, "way4": 4}
HAND_DIM = {"end": 2, "way4": 8, "style": 2}
STYLE_MAX_LENGTH = STEPS * SPEED_MAX         # größtmögliche Weglänge: Normierung der ersten Fahrstil-Größe auf [0, 1]

# Varianten: handgemacht, gelernt und einmal gelernt („fixed“: Training auf 10 000 Zufallsfahrten vor dem Lauf), gelernt und im Lauf neu trainiert („aur“, die AURORA-Schleife)
VARIANTS = ("hand_end", "hand_way4", "hand_style", "fixed_pca2", "fixed_pca8", "fixed_ae2", "fixed_ae8", "aur_pca2", "aur_pca8", "aur_ae2", "aur_ae8")
VARIANT_MODE = {v: v.split("_")[0] for v in VARIANTS}                                           # hand / fixed / aur
VARIANT_METHOD = {v: (v.split("_")[1] if VARIANT_MODE[v] == "hand" else v.split("_")[1][:-1]) for v in VARIANTS}      # end / way4 / style bzw. pca / ae
VARIANT_DIM = {v: HAND_DIM[v.split("_")[1]] if VARIANT_MODE[v] == "hand" else int(v[-1]) for v in VARIANTS}
VARIANT_LABELS = {
    "hand_end": "Handgemacht: Endposition (2 Werte)", "hand_way4": "Handgemacht: 4 Wegpunkte (8 Werte)", "hand_style": "Handgemacht: Fahrstil (2 Werte)",
    "fixed_pca2": "PCA, einmal gelernt (2 Werte)", "fixed_pca8": "PCA, einmal gelernt (8 Werte)", "fixed_ae2": "Autoencoder, einmal gelernt (2 Werte)", "fixed_ae8": "Autoencoder, einmal gelernt (8 Werte)",
    "aur_pca2": "AURORA mit PCA (2 Werte)", "aur_pca8": "AURORA mit PCA (8 Werte)", "aur_ae2": "AURORA mit Autoencoder (2 Werte)", "aur_ae8": "AURORA mit Autoencoder (8 Werte)",
}
GROUP_LABELS = {"hand": "handgemacht", "fixed": "einmal gelernt", "aur": "AURORA (im Lauf neu gelernt)"}
STORED_VARIANTS = tuple(v for v in VARIANTS if VARIANT_MODE[v] != "aur")      # Räume (Encoder und Schwerpunkte) in spaces.npz; die AURORA-Räume entstehen im Lauf

K = 256                                      # Zellen (Schwerpunkte) in jedem Raum: gleiche Zellzahl, damit der QD-Score vergleichbar ist
CENTROID_SAMPLES = 30000                     # Fahrten für die Schwerpunkte der handgemachten Beschreibungen (k-Means auf deren Verhalten)
CENTROID_SEED = 12345                        # Zufall dieser Fahrten
TRAIN_RIDES = 10000                          # Trainingsfahrten der einmal gelernten Beschreibungen
TRAIN_SEED = 777
KMEANS_SEED = 1                              # Zufall des k-Means (Start nach k-means++, 25 Iterationen)
KMEANS_ITER = 25
WARMUP_GENS = 5                              # AURORA: die ersten Generationen sind Zufallsfahrten (500), dann wird die erste Beschreibung gelernt
RETRAIN_GENS = (10, 30, 60, 100, 150, 250)   # AURORA: in diesen Generationen wird neu gelernt und das Archiv neu eingetragen
PATH_CELL = 2                                # Wegzellen: 2 × 2-Zellen des Labyrinths, die irgendeine Elite-Fahrt berührt (höchstens 144)
END_CELLS_TOTAL = 144

STUDY_SEEDS = 20                             # Läufe je Zelle der Studie
STUDY_GENS = 400                             # Generationen je Lauf (bei POP = 100: 40 000 Evaluationen)
STUDY_BUDGET = POP * STUDY_GENS

GENS_MIN, GENS_MAX, GENS_STEP, DEFAULT_GENS = 10, 200, 10, 100      # Regler: Generationen des Live-Laufs
DEFAULT_LEVEL = 0
SEED_MAX = 999999
DEFAULT_SEED = 35
DEFAULT_A = "hand_end"
DEFAULT_B = "aur_ae2"

PRESET_ORDER = ("Richtiges Vorwissen", "Falsches Vorwissen", "Neu lernen", "Falle")


def _preset(level=DEFAULT_LEVEL, gens=DEFAULT_GENS, a=DEFAULT_A, b=DEFAULT_B):
    return {"level": level, "gens": gens, "seed": DEFAULT_SEED, "a": a, "b": b}


PRESETS = {
    "Richtiges Vorwissen": _preset(a="hand_end", b="aur_ae2"),
    "Falsches Vorwissen": _preset(a="hand_style", b="aur_ae2"),
    "Neu lernen": _preset(a="fixed_ae8", b="aur_ae8"),
    "Falle": _preset(level=3, gens=200),
}
PRESET_HELP = {
    "Richtiges Vorwissen": "Handgemachte Endposition gegen AURORA mit Autoencoder (2 Werte): nach 400 Generationen enden die Elites im freien Feld im Mittel in 137.15 gegen 52.85 verschiedenen Endzellen (Studie).",
    "Falsches Vorwissen": "Fahrstil (Weglänge und Drehung, sagt nichts über den Ort) gegen AURORA mit Autoencoder (2 Werte): 33.80 gegen 52.85 Endzellen im freien Feld (Studie).",
    "Neu lernen": "Autoencoder (8 Werte) einmal gelernt gegen im Lauf neu gelernt (AURORA): 73.35 gegen 63.65 Endzellen im freien Feld (Studie).",
    "Falle": "Stufe 3 mit 200 Generationen: Endposition gegen AURORA mit Autoencoder (2 Werte); in der Studie (400 Generationen) erreichen das Ziel 20 beziehungsweise 10 von 20 Läufen.",
}                                                      # jede Zahl in tests/test_claims.py
