"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).
Alle Regler sind immer sichtbar und wirken immer: im Live-Lauf laufen zwei frei wählbare Beschreibungen (A und B) mit derselben Stufe und demselben Seed nebeneinander; es gibt keinen ausblendbaren Regler und kein KEPT-Muster."""

import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import aur_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _variant(value):
    """Permalink-Wert einer Beschreibung: nur die elf Namen aus `C.VARIANTS`, sonst ValueError (der Wert wird dann ignoriert)."""
    if not isinstance(value, str) or value not in C.VARIANTS:
        raise ValueError(f"unbekannte Beschreibung: {value!r}")
    return value


SETTING_SPECS = {
    "level_select": SettingSpec("level", int, C.DEFAULT_LEVEL, C.STUDY_LEVELS[0], C.STUDY_LEVELS[-1]),
    "gens_slider": SettingSpec("gens", int, C.DEFAULT_GENS, C.GENS_MIN, C.GENS_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "a_select": SettingSpec("a", _variant, C.DEFAULT_A),
    "b_select": SettingSpec("b", _variant, C.DEFAULT_B),
}
PRESET_KEYS = {"level": "level_select", "gens": "gens_slider", "seed": "seed_input", "a": "a_select", "b": "b_select"}
OPTIONS = {"level_select": C.STUDY_LEVELS}                    # Regler, die nur feste Werte kennen


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def snap_to_option(state_key, value):
    """Regler mit festen Werten: ein Permalink-Wert dazwischen rastet auf den nächsten Wert ein (bei Gleichstand auf den kleineren)."""
    return min(OPTIONS[state_key], key=lambda o: (abs(o - value), o))


def snap_gens(value):
    """Die Generationen rasten auf das nächste Vielfache der Schrittweite (10, von 10 an) innerhalb der Grenzen ein."""
    snapped = C.GENS_MIN + round((value - C.GENS_MIN) / C.GENS_STEP) * C.GENS_STEP
    return int(min(C.GENS_MAX, max(C.GENS_MIN, snapped)))


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if value != value:                     # NaN
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                if state_key in OPTIONS:
                    value = snap_to_option(state_key, value)
                if state_key == "gens_slider":
                    value = snap_gens(value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
