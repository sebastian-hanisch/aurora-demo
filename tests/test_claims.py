"""JEDE Zahl aus README und App-Texten wird hier nachgerechnet: Täuschungsmaße exakt aus dem Labyrinth, Studienzahlen aus der vorgerechneten Datei (je Zelle 20 Läufe, Seeds 0 bis 19, 40 000 Evaluationen, 256 Zellen).
Die Datei ist fest; ändern sich die Zahlen nach einer neuen Rechnung, müssen README und Hilfetexte nachgezogen werden."""

from pathlib import Path

import numpy as np

import aur_constants as C
import aur_evaluation as E

PRE = E.load_precomputed()
LEARNED = C.VARIANTS[3:]


def cell(level, variant):
    return E.study_cell(PRE, level, variant)


def mean(level, variant, key):
    return round(float(np.mean(cell(level, variant)[key])), 2)


def recon(level, variant):
    """Mittlerer Rekonstruktionsfehler (ungerundet)."""
    return float(np.mean(cell(level, variant)["recon"]))


def vals(level, variant, key):
    return np.array(cell(level, variant)[key], float)


def p(level, a, b, key="end_cells"):
    return E.mannwhitney_p(vals(level, a, key), vals(level, b, key))


def test_exact_deception_measures_and_study_design():
    """README: Stufe 3: kürzester Weg 49 Zellen, Umwegfaktor 2.88, Korrelation 0.69, 1 Falle; 11 Beschreibungen × 2 Stufen × 20 Läufe = 440 Läufe à 400 Generationen (40 000 Evaluationen); 256 Zellen; 500 Zufallsfahrten zum Aufwärmen."""
    m3 = E.level_metrics(3)
    assert (m3["path"], round(m3["detour"], 2), round(m3["fdc"], 2), m3["minima"]) == (49, 2.88, 0.69, 1)
    assert len(C.VARIANTS) * len(C.STUDY_LEVELS) * C.STUDY_SEEDS == 440 and C.STUDY_GENS == 400 and C.STUDY_BUDGET == 40000 and C.K == 256 and C.POP * C.WARMUP_GENS == 500 and C.RETRAIN_GENS == (10, 30, 60, 100, 150, 250)
    assert C.TRAIN_RIDES == 10000 and C.CENTROID_SAMPLES == 30000 and C.RAW_DIM == 40 and C.STYLE_MAX_LENGTH == 98.0


def test_the_right_prior_beats_everything_in_the_free_field():
    """README: verschiedene Endzellen im freien Feld: Endposition 137.15, 4 Wegpunkte 84.30, Fahrstil 33.80; gelernt: PCA einmal 77.00 / 65.70 (2 / 8 Werte), Autoencoder einmal 47.20 / 73.35, AURORA mit PCA 77.15 / 61.35, AURORA mit Autoencoder 52.85 / 63.65;
    alle gelernten und die anderen handgemachten liegen unter der Endposition (alle p < 0.001)."""
    assert [mean(0, v, "end_cells") for v in ("hand_end", "hand_way4", "hand_style")] == [137.15, 84.30, 33.80]
    assert [mean(0, v, "end_cells") for v in LEARNED] == [77.00, 65.70, 47.20, 73.35, 77.15, 61.35, 52.85, 63.65]
    assert all(p(0, v, "hand_end") < 0.001 for v in C.VARIANTS[1:])


def test_learned_eight_values_stay_below_the_hand_made_four_waypoints():
    """README: die gelernten 8-Werte-Beschreibungen (65.70 / 73.35 / 61.35 / 63.65) liegen im freien Feld unter den 4 Wegpunkten (84.30), alle p < 0.001; auf Stufe 3 (61.20 gegen 54.10 / 53.80 / 52.80 / 50.15) ebenso."""
    eight = ("fixed_pca8", "fixed_ae8", "aur_pca8", "aur_ae8")
    assert all(p(0, v, "hand_way4") < 0.001 and mean(0, v, "end_cells") < mean(0, "hand_way4", "end_cells") for v in eight)
    assert mean(3, "hand_way4", "end_cells") == 61.20 and [mean(3, v, "end_cells") for v in eight] == [54.10, 53.80, 52.80, 50.15]
    assert all(p(3, v, "hand_way4") < 0.001 for v in eight)


def test_the_wrong_prior_is_worse_than_every_learned_description_in_the_end_cells():
    """README: Fahrstil 33.80 (freies Feld) und 34.95 (Stufe 3) Endzellen; gelernt im freien Feld 47.20 bis 77.15, auf Stufe 3 42.05 bis 55.45 (alle p < 0.001)."""
    free, trap = [mean(0, v, "end_cells") for v in LEARNED], [mean(3, v, "end_cells") for v in LEARNED]
    assert mean(3, "hand_style", "end_cells") == 34.95 and (min(free), max(free)) == (47.20, 77.15) and (min(trap), max(trap)) == (42.05, 55.45)
    assert all(p(0, v, "hand_style") < 0.001 and p(3, v, "hand_style") < 0.001 for v in LEARNED)


def test_on_the_trap_the_path_cells_favour_pca_only_weakly_and_disfavour_the_autoencoder():
    """README: Wegzellen auf Stufe 3: Endposition 120.70, 4 Wegpunkte 90.45, Fahrstil 96.00; PCA 102.70 / 97.80 / 98.35 / 99.45, Autoencoder 91.95 / 90.90 / 84.35 / 91.85; gegen den Fahrstil nur die einmal gelernte PCA mit 2 Werten belegt (p = 0.00088),
    die drei anderen PCA p > 0.05; alle vier Autoencoder darunter mit p < 0.01. Im freien Feld sind die Wegzellen fast gesättigt: 144.00 bei der Endposition, Fahrstil 137.50, alle anderen mindestens 143.55."""
    assert [mean(3, v, "path_cells") for v in ("hand_end", "hand_way4", "hand_style")] == [120.70, 90.45, 96.00]
    pca, ae = ("fixed_pca2", "fixed_pca8", "aur_pca2", "aur_pca8"), ("fixed_ae2", "fixed_ae8", "aur_ae2", "aur_ae8")
    assert [mean(3, v, "path_cells") for v in pca] == [102.70, 97.80, 98.35, 99.45] and [mean(3, v, "path_cells") for v in ae] == [91.95, 90.90, 84.35, 91.85]
    assert round(p(3, "fixed_pca2", "hand_style", "path_cells"), 5) == 0.00088 and all(p(3, v, "hand_style", "path_cells") > 0.05 for v in pca[1:]) and all(p(3, v, "hand_style", "path_cells") < 0.01 for v in ae)
    assert mean(0, "hand_end", "path_cells") == 144.00 and mean(0, "hand_style", "path_cells") == 137.50 and min(mean(0, v, "path_cells") for v in C.VARIANTS if v != "hand_style") == 143.55


def test_relearning_is_not_better_than_learning_once():
    """README: Endzellen einmal gelernt gegen AURORA, freies Feld: PCA 2 Werte 77.00 / 77.15 (p = 0.93), PCA 8 Werte 65.70 / 61.35 (p = 0.0011), Autoencoder 2 Werte 47.20 / 52.85 (p < 0.001, der einzige belegte Gewinn), Autoencoder 8 Werte 73.35 / 63.65 (p < 0.001);
    Stufe 3: 55.45 / 52.35 (p < 0.001), 54.10 / 52.80 (p = 0.12), 42.40 / 42.05 (p = 0.54), 53.80 / 50.15 (p < 0.001); belegt schlechter in 4 von 8 Paaren."""
    pair = lambda lv, m: (mean(lv, f"fixed_{m}", "end_cells"), mean(lv, f"aur_{m}", "end_cells"), p(lv, f"aur_{m}", f"fixed_{m}"))
    free = [pair(0, m) for m in ("pca2", "pca8", "ae2", "ae8")]
    trap = [pair(3, m) for m in ("pca2", "pca8", "ae2", "ae8")]
    assert [(a, b) for a, b, _ in free] == [(77.00, 77.15), (65.70, 61.35), (47.20, 52.85), (73.35, 63.65)] and [(a, b) for a, b, _ in trap] == [(55.45, 52.35), (54.10, 52.80), (42.40, 42.05), (53.80, 50.15)]
    assert round(free[0][2], 2) == 0.93 and round(free[1][2], 4) == 0.0011 and free[2][2] < 0.001 and free[3][2] < 0.001
    assert trap[0][2] < 0.001 and round(trap[1][2], 2) == 0.12 and round(trap[2][2], 2) == 0.54 and trap[3][2] < 0.001
    assert sum(1 for a, b, pv in free + trap if b < a and pv < 0.05) == 4 and sum(1 for a, b, pv in free + trap if b > a and pv < 0.05) == 1 and all(b <= a for a, b, _ in trap)


def test_aurora_loses_cells_when_it_re_inserts():
    """README: belegte Zellen im freien Feld, einmal gelernt gegen AURORA: PCA 8 Werte 254.95 / 247.40, Autoencoder 8 Werte 255.00 / 243.95 (von 256)."""
    assert [mean(0, v, "coverage") for v in ("fixed_pca8", "aur_pca8", "fixed_ae8", "aur_ae8")] == [254.95, 247.40, 255.00, 243.95]


def test_pca_and_the_autoencoder_have_no_uniform_order():
    """README: einmal gelernt, Endzellen im freien Feld: 2 Werte PCA 77.00 gegen Autoencoder 47.20, 8 Werte PCA 65.70 gegen Autoencoder 73.35 (beide p < 0.001)."""
    assert p(0, "fixed_pca2", "fixed_ae2") < 0.001 and p(0, "fixed_pca8", "fixed_ae8") < 0.001
    assert mean(0, "fixed_pca2", "end_cells") > mean(0, "fixed_ae2", "end_cells") and mean(0, "fixed_pca8", "end_cells") < mean(0, "fixed_ae8", "end_cells")


def test_the_autoencoder_does_not_reconstruct_better_than_pca():
    """README: Rekonstruktionsfehler auf den Rohdaten der Elites, einmal gelernt, 2 / 8 Werte: PCA 0.0174 / 0.0008 gegen Autoencoder 0.0272 / 0.0013 im freien Feld, 0.0166 / 0.0008 gegen 0.0318 / 0.0012 auf Stufe 3."""
    r = lambda lv, v: round(recon(lv, v), 4)
    assert [r(0, v) for v in ("fixed_pca2", "fixed_pca8", "fixed_ae2", "fixed_ae8")] == [0.0174, 0.0008, 0.0272, 0.0013] and [r(3, v) for v in ("fixed_pca2", "fixed_pca8", "fixed_ae2", "fixed_ae8")] == [0.0166, 0.0008, 0.0318, 0.0012]
    assert all(recon(lv, f"fixed_ae{d}") > recon(lv, f"fixed_pca{d}") for lv in C.STUDY_LEVELS for d in (2, 8))


def test_a_small_reconstruction_error_does_not_mean_many_end_cells():
    """README: die PCA mit 8 Werten hat den kleinsten Rekonstruktionsfehler (0.0008), aber weniger Endzellen (65.70) als die PCA mit 2 Werten (77.00, Fehler 0.0174) im freien Feld."""
    assert recon(0, "fixed_pca8") < recon(0, "fixed_pca2") and mean(0, "fixed_pca8", "end_cells") < mean(0, "fixed_pca2", "end_cells")


def test_the_qd_score_is_not_a_measure_across_descriptions():
    """README: QD-Score im freien Feld: Endposition 213.60, gelernt mit 2 Werten 224.39 (PCA) bis 230.46 (AURORA mit Autoencoder), also höher bei weniger Endzellen; mittlere Direktheit 0.834 gegen 0.877 bis 0.903."""
    q2 = [mean(0, v, "qd") for v in ("fixed_pca2", "fixed_ae2", "aur_pca2", "aur_ae2")]
    assert mean(0, "hand_end", "qd") == 213.60 and q2 == [224.39, 228.04, 225.19, 230.46] and min(q2) > 213.60
    qual = lambda v: round(float(np.mean(np.array(cell(0, v)["qd"]) / np.array(cell(0, v)["coverage"]))), 3)
    assert qual("hand_end") == 0.834 and [qual(v) for v in ("fixed_pca2", "fixed_ae2", "aur_pca2", "aur_ae2")] == [0.877, 0.891, 0.880, 0.903]


def test_the_trap_results():
    """README: Stufe 3, Erfolg (von 20): Endposition 20, 4 Wegpunkte 15, Fahrstil 19; PCA einmal 12 / 18 (2 / 8 Werte), Autoencoder einmal 5 / 13, AURORA mit PCA 11 / 20, AURORA mit Autoencoder 10 / 16; Median der Evaluationen bis zum Erfolg bei der Endposition 3 526.5;
    im freien Feld erreichen alle 11 Beschreibungen das Ziel in 20 von 20 Läufen."""
    count = lambda lv, v: E.success_count(cell(lv, v)["runs"])
    assert [count(3, v) for v in C.VARIANTS] == [20, 15, 19, 12, 18, 5, 13, 11, 20, 10, 16]
    assert E.median_solved(cell(3, "hand_end")["runs"]) == 3526.5 and C.fmt_med(3526.5) == "3 526.5"
    assert all(count(0, v) == 20 for v in C.VARIANTS)


def test_the_preset_help_texts_quote_the_study():
    """PRESET_HELP: 137.15 / 52.85, 33.80 / 52.85, 73.35 / 63.65 Endzellen im freien Feld; 20 beziehungsweise 10 von 20 Läufen auf Stufe 3."""
    h = C.PRESET_HELP
    assert "137.15" in h["Richtiges Vorwissen"] and "52.85" in h["Richtiges Vorwissen"] and mean(0, "hand_end", "end_cells") == 137.15 and mean(0, "aur_ae2", "end_cells") == 52.85
    assert "33.80" in h["Falsches Vorwissen"] and "52.85" in h["Falsches Vorwissen"] and mean(0, "hand_style", "end_cells") == 33.80
    assert "73.35" in h["Neu lernen"] and "63.65" in h["Neu lernen"] and mean(0, "fixed_ae8", "end_cells") == 73.35 and mean(0, "aur_ae8", "end_cells") == 63.65
    assert "20 beziehungsweise 10 von 20" in h["Falle"] and E.success_count(cell(3, "hand_end")["runs"]) == 20 and E.success_count(cell(3, "aur_ae2")["runs"]) == 10


def test_every_row_of_the_readme_table_matches_the_stored_study():
    """Die große Tabelle der README (22 Zeilen) wird Zeile für Zeile aus der vorgerechneten Datei neu gebildet."""
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    levels = {0: "freies Feld", 3: "Stufe 3"}
    for lv in C.STUDY_LEVELS:
        for v in C.VARIANTS:
            c = cell(lv, v)
            qd, cov, en, pa = np.array(c["qd"]), np.array(c["coverage"], float), np.array(c["end_cells"], float), np.array(c["path_cells"], float)
            rc = [x for x in c["recon"] if x is not None]
            row = (f"| {levels[lv]} | {C.VARIANT_LABELS[v]} | {cov.mean():.2f} | {qd.mean():.2f} | {np.mean(qd / cov):.3f} | {en.mean():.2f} ± {en.std(ddof=1):.2f} | {pa.mean():.2f} ± {pa.std(ddof=1):.2f} | "
                   f"{E.success_count(c['runs'])} von 20 | {np.mean(c['progress']):.2f} | {f'{np.mean(rc):.4f}' if rc else '–'} |")
            assert row in readme, row
