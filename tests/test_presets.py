"""Presets: Vollständigkeit, gültige Werte, Permalink-Konstanten, Formatierer."""

import pytest

import aur_constants as C
import aur_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_ORDER) and len(C.PRESETS) == 4
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for preset in C.PRESETS.values():
        assert preset["level"] in C.STUDY_LEVELS and C.GENS_MIN <= preset["gens"] <= C.GENS_MAX and P.snap_gens(preset["gens"]) == preset["gens"]
        assert preset["a"] in C.VARIANTS and preset["b"] in C.VARIANTS
        for key, state_key in P.PRESET_KEYS.items():
            P.SETTING_SPECS[state_key].caster(preset[key])


def test_preset_names_state_the_values_they_set():
    assert C.PRESETS["Richtiges Vorwissen"]["a"] == "hand_end" and C.PRESETS["Falsches Vorwissen"]["a"] == "hand_style" and C.PRESETS["Falle"]["level"] == 3
    neu = C.PRESETS["Neu lernen"]
    assert neu["a"].startswith("fixed_") and neu["b"].startswith("aur_") and neu["a"][-3:] == neu["b"][-3:]


def test_the_default_preset_equals_the_default_settings():
    p = C.PRESETS["Richtiges Vorwissen"]
    assert (p["level"], p["gens"], p["seed"], p["a"], p["b"]) == (C.DEFAULT_LEVEL, C.DEFAULT_GENS, C.DEFAULT_SEED, C.DEFAULT_A, C.DEFAULT_B)


def test_bounds_and_url_params():
    assert P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("gens_slider") == (10, 200) and P.bounds("level_select") == (0, 3)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS) == 5


@pytest.mark.parametrize("value,expected", [(0, 10), (14, 10), (16, 20), (104, 100), (106, 110), (999, 200)])
def test_generations_snap_to_the_step_from_the_lower_bound(value, expected):
    assert P.snap_gens(value) == expected


@pytest.mark.parametrize("value,expected", [(0, 0), (1, 0), (2, 3), (3, 3)])
def test_levels_snap_to_the_study_levels(value, expected):
    assert P.snap_to_option("level_select", value) == expected


@pytest.mark.parametrize("value", C.VARIANTS)
@pytest.mark.parametrize("key", ["a_select", "b_select"])
def test_variant_casters_accept_every_variant(key, value):
    assert P.SETTING_SPECS[key].caster(value) == value


@pytest.mark.parametrize("value", ["hand", "grid", "aur_ae3", "", 3, None, ["hand_end"]])
def test_variant_casters_refuse_everything_else(value):
    with pytest.raises(ValueError):
        P.SETTING_SPECS["a_select"].caster(value)


def test_variant_tables_are_consistent():
    assert C.K == 256 and C.WARMUP_GENS < min(C.RETRAIN_GENS) and list(C.RETRAIN_GENS) == sorted(C.RETRAIN_GENS) and C.RAW_DIM == 40 and C.DEFAULT_A in C.VARIANTS and C.DEFAULT_B in C.VARIANTS
    assert C.STYLE_MAX_LENGTH == pytest.approx(98.0) and C.STUDY_BUDGET == 40000 and C.POP * C.WARMUP_GENS >= C.K


def test_formatters():
    assert C.fmt_int(40000) == "40 000" and C.fmt_med(None) == "–" and C.fmt_med(988.5) == "988.5" and C.fmt_med(1195.0) == "1 195"
