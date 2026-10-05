"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Stufen, alle Beschreibungen als A und B, Randwerte, Studien-Regler, Würfel-Knopf, Permalink-Grenzen, Abschnitte, Footer."""

import random
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import aur_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
STUDY_CHARTS = 4              # Endzellen-Balken, Wegzellen-Balken, Endzellen über den Evaluationen, Rekonstruktionsfehler gegen Endzellen
LIVE_CHARTS = 4               # zwei Elite-Karten, zwei Kurven (dazu je eine Code-Karte bei Beschreibungen mit 2 Werten)


def _charts(a, b):
    return STUDY_CHARTS + LIVE_CHARTS + sum(1 for v in (a, b) if C.VARIANT_DIM[v] == 2)


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_default_run_has_no_exception_and_shows_the_reference_values():
    at = _run()
    _ok(at)
    assert _metric(at, "Kürzester Weg") == "17 Zellen" and _metric(at, "Fallen (lokale Minima)") == "0" and _metric(at, "Freie Zellen") == "484"
    assert _metric(at, "A: belegte Zellen").endswith("von 256") and _metric(at, "B: belegte Zellen").endswith("von 256") and _metric(at, "B: verschiedene Endzellen").endswith("von 144")
    assert len(at.get("plotly_chart")) == _charts(C.DEFAULT_A, C.DEFAULT_B)


def test_all_sections_are_present():
    at = _run()
    assert [s.value for s in at.subheader] == ["📐 Ein Lauf: zwei Beschreibungen nebeneinander", "🔬 Wie viel ist das Vorwissen wert? Die Studie über 20 Läufe", "🚧 Wo die Annahmen enden"]
    assert any(m.value.startswith("## 🔗 Wie täuschend ist diese Stufe?") for m in at.markdown)
    assert any("Diese Demo ist Teil des Portfolios" in c.value for c in at.caption)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert (at.session_state["level_select"], at.session_state["gens_slider"], at.session_state["a_select"], at.session_state["b_select"]) == (p["level"], p["gens"], p["a"], p["b"])
    assert at.metric


@pytest.mark.parametrize("level", list(C.STUDY_LEVELS))
def test_every_level_runs(level):
    at = _run(level_select=level, gens_slider=10)
    _ok(at)
    assert _metric(at, "A: belegte Zellen").endswith("von 256")


@pytest.mark.parametrize("variant", C.VARIANTS)
def test_every_variant_runs_as_a_and_as_b(variant):
    at = _run(a_select=variant, b_select="hand_end", gens_slider=10)
    _ok(at)
    assert len(at.get("plotly_chart")) == _charts(variant, "hand_end")
    at = _run(a_select="hand_end", b_select=variant, gens_slider=10)
    _ok(at)
    assert len(at.get("plotly_chart")) == _charts("hand_end", variant)


def test_the_same_description_twice_gives_the_same_run():
    at = _run(a_select="aur_ae2", b_select="aur_ae2", gens_slider=20)
    _ok(at)
    assert _metric(at, "A: QD-Score") == _metric(at, "B: QD-Score") and _metric(at, "A: Wegzellen") == _metric(at, "B: Wegzellen")


@pytest.mark.parametrize("kw", [dict(gens_slider=10), dict(gens_slider=200, a_select="aur_pca8", b_select="aur_ae8"), dict(level_select=3, gens_slider=40, a_select="hand_style", b_select="fixed_ae2"),
                                dict(seed_input=0, gens_slider=10), dict(seed_input=999999, gens_slider=10, level_select=3)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_the_live_info_names_the_cells_and_the_reconstruction_error():
    at = _run(gens_slider=30, a_select="hand_end", b_select="aur_ae2")
    text = next(i.value for i in at.info if i.value.startswith("Nach 30 Generationen belegt A"))
    assert "Wegzellen" in text and "Rekonstruktionsfehler des Encoders" in text and "B " in text and "Das ist ein Lauf" in text
    plain = next(i.value for i in _run(gens_slider=10, a_select="hand_end", b_select="hand_way4").info if i.value.startswith("Nach 10 Generationen belegt A"))
    assert "Rekonstruktionsfehler" not in plain


def test_only_two_value_descriptions_get_a_code_map():
    at = _run(gens_slider=10, a_select="hand_way4", b_select="aur_ae2")
    assert any("der Code hat 8 Werte" in c.value for c in at.caption) and any("Code mit 2 Werten" in c.value for c in at.caption)


def test_the_solvability_line_names_the_steps_used():
    assert any("Lösbar? Ja" in c.value and "Schritten benutzt" in c.value for c in _run(level_select=3, gens_slider=10).caption)


def test_the_study_table_has_one_row_per_variant_and_the_selector_switches_it():
    at = _run(study_level_select=3)
    _ok(at)
    table = next(m.value for m in at.markdown if m.value.startswith(f"**{C.STUDY_LEVEL_LABELS[3]}**"))
    assert table.count("\n| ") == len(C.VARIANTS) + 1 and all(C.VARIANT_LABELS[v] in table for v in C.VARIANTS)       # Kopfzeile + eine Zeile je Beschreibung (die Trennzeile beginnt mit |---)


def test_the_study_info_names_the_main_findings():
    at = _run()
    text = next(i.value for i in at.info if i.value.startswith("**Das richtige Vorwissen schlägt alles"))
    assert "137.15" in text and "33.80" in text and "Neu lernen (AURORA) ist nicht durchgehend besser" in text and "rekonstruiert hier nicht besser als die PCA" in text and "die Zahl der Endzellen sagt nicht, wer das Ziel erreicht" in text


def test_dice_button_changes_the_seed_and_the_run(monkeypatch):
    """Der Würfel zieht sonst einen unseeded Zufalls-Seed; deshalb ist der gewürfelte Seed im Test fest."""
    monkeypatch.setattr(random, "randint", lambda a, b: 508145)
    at = _run(gens_slider=20)
    old_seed = at.session_state["seed_input"]
    old = (_metric(at, "A: QD-Score"), _metric(at, "B: QD-Score"))
    next(b for b in at.button if b.label == "🎲 Neuen Lauf würfeln").click().run()
    _ok(at)
    assert at.session_state["seed_input"] == 508145 != old_seed
    assert (_metric(at, "A: QD-Score"), _metric(at, "B: QD-Score")) != old


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["level"] = "9"
    at.query_params["gens"] = "104"
    at.query_params["a"] = "hand_style"
    at.query_params["b"] = "fixed_pca8"
    at.run()
    _ok(at)
    assert at.session_state["level_select"] == 3 and at.session_state["gens_slider"] == 100 and at.session_state["a_select"] == "hand_style" and at.session_state["b_select"] == "fixed_pca8"


def test_permalink_ignores_garbage():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["level"] = "viele"
    at.query_params["seed"] = "nan"
    at.query_params["a"] = "hand_end2"
    at.query_params["b"] = "grid"
    at.run()
    _ok(at)
    assert at.session_state["level_select"] == C.DEFAULT_LEVEL and at.session_state["seed_input"] == C.DEFAULT_SEED
    assert at.session_state["a_select"] == C.DEFAULT_A and at.session_state["b_select"] == C.DEFAULT_B


def test_no_sentence_wide_comma_replacement_in_the_app_source():
    """Regressionsschutz: `.replace(",", ".")` auf einem ganzen (verketteten) Satz macht aus Kommas im Fließtext Punkte; Tausender nur über `fmt_int`."""
    assert '.replace(",", ".")' not in Path(APP).read_text(encoding="utf-8")
