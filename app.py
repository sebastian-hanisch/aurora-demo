"""AURORA – wie viel ist Vorwissen über die Verhaltensbeschreibung wert? - Stück 8 der Konzepte-Linie "Novelty Search und Quality-Diversity"
Sebastian Hanisch - Operations Research und Machine Learning

Derselbe Hof-Roboter, dieselbe Qualität (Direktheit) und dasselbe Archiv aus 256 Zellen wie in Stück 7 - aber die Beschreibung des Verhaltens ist nicht mehr von Hand gewählt, sondern aus den Rohdaten der Fahrt gelernt (PCA oder Autoencoder),
einmal vorab oder im Lauf immer wieder (AURORA). Gemessen wird, ob das ein richtig gewähltes Vorwissen einholt und ein falsch gewähltes schlägt. Siehe README für die Einordnung in die Linie.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import aur_constants as C
import aur_evaluation as E
import aur_maze as M
from aur_presets import (apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params)
from aur_visualization import (build_elite_map, build_end_cells_chart, build_end_cells_curve, build_latent_map, build_path_cells_chart, build_path_cells_curve, build_recon_scatter, build_study_curve)

st.set_page_config(page_title="AURORA – Sebastian Hanisch", layout="wide")

PRE = E.load_precomputed()
N = C.STUDY_SEEDS


@st.cache_data(show_spinner=False)
def _pair(level, gens, seed, a, b):
    return E.live_pair(level, gens, seed, a, b)


@st.cache_data(show_spinner=False)
def _solvability(level):
    return E.solvability(level)


def _cm(level, variant):
    return E.study_cell(PRE, level, variant)


def _vals(level, variant, key):
    return np.array(_cm(level, variant)[key], float)


def _mean(level, variant, key):
    return float(_vals(level, variant, key).mean())


def _mean_quality(level, variant):
    c = _cm(level, variant)
    return float(np.mean([q / n for q, n in zip(c["qd"], c["coverage"])]))


def _pstr(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _range(level, variants, key):
    vals = [_mean(level, v, key) for v in variants]
    return min(vals), max(vals)


st.title("🧠 AURORA – wie viel ist Vorwissen über die Verhaltensbeschreibung wert?")
st.markdown(
    """
Bisher wählte ich die Beschreibung des Verhaltens **von Hand**: die Endposition der Fahrt oder ihre Position nach jedem Viertel. Das ist **Vorwissen** darüber, worauf es ankommt. **AURORA** (Grillotti und Cully) braucht es nicht: ein **Autoencoder** verdichtet die **Rohdaten** einer Fahrt
(hier 20 Positionen, 40 Werte) zu einem kurzen **Code** (2 oder 8 Werte), der Code ist die Beschreibung, und während der Suche wird er **immer wieder neu gelernt**; das Archiv wird dabei neu eingetragen. Dieses Stück misst, **wie viel das Vorwissen wert ist**:
Holt AURORA ein richtig gewähltes Vorwissen ein, und schlägt es ein falsch gewähltes?
"""
)
st.caption(
    "Stück 8 der Linie „Novelty Search und Quality-Diversity“; Labyrinth, Roboter, Qualität (Direktheit), Auswahl der Eltern und die 256 CVT-Zellen wie in [Stück 7](https://sebastianhanisch-cvt-map-elites-demo.streamlit.app/). "
    "Die AURORA-Schleife ist eine eigene, vereinfachte Umsetzung (PCA oder ein kleiner Autoencoder in NumPy, Zellen als CVT statt der Archivstruktur der Quelle), keine Referenzbibliothek."
)

with st.expander("So lernen die Beschreibungen ihre Zellen", expanded=True):
    st.markdown(
        """
- **Handgemacht:** *Endposition* (2 Werte), *4 Wegpunkte* (die Position nach jedem Viertel der Fahrt, 8 Werte) oder *Fahrstil* (Weglänge und mittleres Drehgen, 2 Werte: **sagt nichts über den Ort** und ist das bewusst falsche Vorwissen).
- **Gelernt:** Rohdaten = die Position nach jedem 20. Teil der Fahrt (40 Werte). **PCA** projiziert auf die ersten 2 oder 8 Hauptachsen; der **Autoencoder** (40 → 16 → Code → 16 → 40, tanh, 400 Adam-Schritte) nimmt den Engpass als Code.
- **Zellen:** immer **256**, damit der QD-Score vergleichbar bleibt: ein k-Means legt 256 Schwerpunkte in den (standardisierten) Code, die Zelle ist der nächste Schwerpunkt. Bei den handgemachten und den **einmal gelernten** Beschreibungen ist der Raum fest (auf Zufallsfahrten gelernt).
- **AURORA:** die ersten 5 Generationen sind 500 Zufallsfahrten, aus ihnen wird die erste Beschreibung gelernt. In den Generationen 10, 30, 60, 100, 150 und 250 wird **neu gelernt** (Trainingsdaten: die 500 Fahrten, alle Elites und die neuen Kinder), die Elites werden nach Qualität neu eingetragen
  (kollidieren zwei in einer Zelle, bleibt die bessere). Bis zur ersten gelernten Beschreibung gibt es kein Archiv.
- **Vergleichsmaße, nicht an der Beschreibung hängend:** **Endzellen** (die Zahl der verschiedenen 2 × 2-Zellen des Labyrinths, in denen die Elites enden, von 144) und **Wegzellen** (die 2 × 2-Zellen, die irgendeine Elite-Fahrt berührt, von 144). Beide bevorzugen eine Beschreibung, die den Ort enthält; das
  ist Teil der Frage, siehe die Grenzen.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESET_ORDER))
for i, name in enumerate(C.PRESET_ORDER):
    with preset_cols[i]:
        st.button(name, key=f"preset_{name}", width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name] or None)

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    level = st.selectbox("Labyrinth (Stufe)", options=C.STUDY_LEVELS, key="level_select", format_func=lambda v: C.LEVEL_LABELS[v], help="Das freie Feld und die Falle (Stufe 3); für beide sind die Räume vorgerechnet.")
    a = st.selectbox("Beschreibung A", options=C.VARIANTS, key="a_select", format_func=lambda v: C.VARIANT_LABELS[v], help="Läuft neben B mit gleicher Stufe und gleichem Seed.")
    b = st.selectbox("Beschreibung B", options=C.VARIANTS, key="b_select", format_func=lambda v: C.VARIANT_LABELS[v], help="Läuft neben A mit gleicher Stufe und gleichem Seed. Gleiche Wahl wie bei A: zwei gleiche Läufe.")
    gens = st.slider("Generationen im Live-Lauf", *bounds("gens_slider"), step=C.GENS_STEP, key="gens_slider", help=f"Je Generation {C.POP} Fahrten, also {C.POP} Evaluationen.")
    seed = st.number_input("Zufalls-Seed", min_value=bounds("seed_input")[0], max_value=bounds("seed_input")[1], step=1, key="seed_input", help="Bestimmt die Anfangspopulation und alle Zufallsentscheidungen; beide Beschreibungen starten gleich.")
    st.button("🎲 Neuen Lauf würfeln", on_click=randomize_seed)

level, a, b, gens, seed = int(level), str(a), str(b), int(gens), int(seed)
sync_query_params({"level_select": level, "a_select": a, "b_select": b, "gens_slider": gens, "seed_input": seed})

grid = M.make_maze(level)
lm = E.level_metrics(level)

st.markdown("---")
st.markdown("## 🔗 Wie täuschend ist diese Stufe?")
st.caption(f"{C.LEVEL_LABELS[level]}. Die Maße stammen aus Stück 1. Hier geht es um das Archiv, nicht um das Ziel: keine Beschreibung benutzt das Ziel.")
m1 = st.columns(4)
m1[0].metric("Kürzester Weg", f"{lm['path']} Zellen", delta=f"{lm['detour']:.2f}-facher Umweg", delta_color="off")
m1[1].metric("Fitness-Distanz-Korrelation", f"{lm['fdc']:.2f}", help="1 = die Luftlinie führt zum Ziel; je kleiner, desto täuschender.")
m1[2].metric("Fallen (lokale Minima)", f"{lm['minima']}", help="Zellen, in denen die Luftlinie zum Ziel kleiner ist als in allen freien Nachbarzellen.")
m1[3].metric("Freie Zellen", f"{lm['free']}", help="So viele Zellen kann eine Endposition belegen.")
dist_ok, steps_used = _solvability(level)
st.caption(f"Lösbar? Ja: eine aus dem kürzesten Pfad abgeleitete Steuerfolge fährt ins Ziel (Abstand am Ende {dist_ok:.2f} Zellen, {steps_used} von {C.STEPS} Schritten benutzt).")

st.markdown("---")
st.subheader("📐 Ein Lauf: zwei Beschreibungen nebeneinander")
with st.spinner(f"Lasse beide Beschreibungen {gens} Generationen laufen (gelernte lernen dabei) …"):
    ra, rb = _pair(level, gens, seed, a, b)
st.caption(f"Beide mit {C.K} Zellen, {C.fmt_int(gens * C.POP)} Evaluationen. A: {C.VARIANT_LABELS[a]}. B: {C.VARIANT_LABELS[b]}.")
cols = st.columns(2)
for col, tag, variant, res in ((cols[0], "A", a, ra), (cols[1], "B", b, rb)):
    with col:
        st.markdown(f"**{tag}: {C.VARIANT_LABELS[variant]}**")
        r1 = st.columns(3)
        r1[0].metric(f"{tag}: belegte Zellen", f"{res.coverage[-1]} von {C.K}")
        r1[1].metric(f"{tag}: verschiedene Endzellen", f"{res.end_cells[-1]} von {C.END_CELLS_TOTAL}")
        r1[2].metric(f"{tag}: Wegzellen", f"{res.path_cells[-1]} von {C.END_CELLS_TOTAL}")
        r2 = st.columns(3)
        r2[0].metric(f"{tag}: QD-Score", f"{res.qd_score[-1]:.2f}")
        r2[1].metric(f"{tag}: beste Direktheit", f"{res.best_quality[-1]:.3f}")
        r2[2].metric(f"{tag}: Evaluationen bis zum Erfolg", C.fmt_int(res.first_solved) if res.first_solved is not None else "–")
        st.plotly_chart(build_elite_map(grid, res.elite_ends, res.elite_quality), width="stretch", key=f"map_{tag}_{level}_{variant}_{gens}_{seed}")
st.caption("Je ein Punkt je belegter Zelle: die Endposition des Elites, nach Direktheit gefärbt (hell = direkter). Handgemacht und gelernt unterscheiden sich darin, **wohin** die Zellen die Elites verteilen.")
col_c, col_d = st.columns(2)
with col_c:
    st.markdown("**Verschiedene Endzellen über den Generationen**")
    st.plotly_chart(build_end_cells_curve(ra.end_cells, rb.end_cells, a, b), width="stretch", key=f"ends_{level}_{a}_{b}_{gens}_{seed}")
with col_d:
    st.markdown("**Wegzellen über den Generationen**")
    st.plotly_chart(build_path_cells_curve(ra.path_cells, rb.path_cells, a, b), width="stretch", key=f"paths_{level}_{a}_{b}_{gens}_{seed}")
st.caption("Bei den gelernten Beschreibungen der AURORA-Läufe stehen die Kurven bis zur ersten gelernten Beschreibung (Generation 5) bei 0, danach springen sie bei jedem Neulernen (10, 30, 60, 100, 150).")
st.markdown("**Was steckt im Code?** (nur bei Beschreibungen mit 2 Werten)")
lat = st.columns(2)
for col, tag, variant, res in ((lat[0], "A", a, ra), (lat[1], "B", b, rb)):
    with col:
        if res.elite_codes.shape[1] == 2:
            st.plotly_chart(build_latent_map(res.elite_codes, res.space.centroids, res.elite_ends[:, 0]), width="stretch", key=f"latent_{tag}_{level}_{variant}_{gens}_{seed}")
            st.caption(f"{tag}: Code mit 2 Werten; die Farbe ist das x der Endposition.")
        else:
            st.caption(f"{tag}: der Code hat {res.elite_codes.shape[1]} Werte, eine Karte gibt es nur bei 2.")
recon = [(tag, res.recon) for tag, res in (("A", ra), ("B", rb)) if res.recon is not None]
st.info(
    f"Nach {gens} Generationen belegt A {ra.coverage[-1]} und B {rb.coverage[-1]} von {C.K} Zellen; die Elites enden in {ra.end_cells[-1]} beziehungsweise {rb.end_cells[-1]} verschiedenen 2 × 2-Zellen (von {C.END_CELLS_TOTAL}) und berühren "
    f"{ra.path_cells[-1]} beziehungsweise {rb.path_cells[-1]} Wegzellen. "
    + ("Rekonstruktionsfehler des Encoders auf den Rohdaten der Elites: " + ", ".join(f"{t} {r:.4f}" for t, r in recon) + ". " if recon else "")
    + f"Das ist ein Lauf; die Studie unten vergleicht {N} Läufe."
)

st.markdown("---")
st.subheader("🔬 Wie viel ist das Vorwissen wert? Die Studie über 20 Läufe")
st.markdown(
    f"Je Stufe und Beschreibung {N} Läufe mit unterschiedlichem Seed, jeweils {C.fmt_int(PRE['budget'])} Evaluationen ({C.STUDY_GENS} Generationen), immer {C.K} Zellen. Verglichen werden **Endzellen** und **Wegzellen**; "
    "der QD-Score ist bei gleicher Zellzahl vergleichbar, hängt aber an der Beschreibung (zwei Werte sind leichter zu füllen als acht)."
)
study_level = st.selectbox("Stufe für die Studie", options=C.STUDY_LEVELS, key="study_level_select", format_func=lambda v: C.STUDY_LEVEL_LABELS[v])
study_level = int(study_level)
col_g, col_h = st.columns(2)
with col_g:
    st.markdown("**Verschiedene Endzellen**")
    st.plotly_chart(build_end_cells_chart(PRE, study_level), width="stretch", key=f"ends_chart_{study_level}")
with col_h:
    st.markdown("**Wegzellen**")
    st.plotly_chart(build_path_cells_chart(PRE, study_level), width="stretch", key=f"paths_chart_{study_level}")
rows = []
for v in C.VARIANTS:
    c = _cm(study_level, v)
    ends, paths, qd = _vals(study_level, v, "end_cells"), _vals(study_level, v, "path_cells"), _vals(study_level, v, "qd")
    rc = [x for x in c["recon"] if x is not None]
    rows.append(f"| {C.VARIANT_LABELS[v]} | {np.mean(c['coverage']):.2f} | {qd.mean():.2f} | {_mean_quality(study_level, v):.3f} | {ends.mean():.2f} ± {ends.std(ddof=1):.2f} | {paths.mean():.2f} ± {paths.std(ddof=1):.2f} | "
                f"{E.success_count(c['runs'])} von {N} | {np.mean(c['progress']):.2f} | {f'{np.mean(rc):.4f}' if rc else '–'} |")
st.markdown(f"**{C.STUDY_LEVEL_LABELS[study_level]}**\n\n| Beschreibung | belegte Zellen | QD-Score | mittlere Direktheit je Elite | verschiedene Endzellen (Mittel ± Standardabweichung) | Wegzellen (Mittel ± Standardabweichung) | Erfolg | Restweg | Rekonstruktionsfehler |\n"
            "|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows))
col_e, col_f = st.columns(2)
with col_e:
    st.markdown("**Verschiedene Endzellen über den Evaluationen**")
    st.plotly_chart(build_study_curve(PRE, study_level), width="stretch", key=f"study_curve_{study_level}")
with col_f:
    st.markdown("**Kleiner Rekonstruktionsfehler, viele Endzellen?**")
    st.plotly_chart(build_recon_scatter(PRE, study_level), width="stretch", key=f"recon_{study_level}")

LEARNED = C.VARIANTS[3:]
LEARNED_2 = ("fixed_pca2", "fixed_ae2", "aur_pca2", "aur_ae2")
LEARNED_8 = ("fixed_pca8", "fixed_ae8", "aur_pca8", "aur_ae8")
e_end, e_style, e_way4 = (_mean(0, v, "end_cells") for v in ("hand_end", "hand_style", "hand_way4"))
l2, l8, lall = _range(0, LEARNED_2, "end_cells"), _range(0, LEARNED_8, "end_cells"), _range(0, LEARNED, "end_cells")
t_end, t_style, t_way4 = (_mean(3, v, "end_cells") for v in ("hand_end", "hand_style", "hand_way4"))
t_all = _range(3, LEARNED, "end_cells")
t_pca, t_ae = _range(3, ("fixed_pca2", "fixed_pca8", "aur_pca2", "aur_pca8"), "path_cells"), _range(3, ("fixed_ae2", "fixed_ae8", "aur_ae2", "aur_ae8"), "path_cells")
t_style_paths = _mean(3, "hand_style", "path_cells")
retrain = {m: (_mean(0, f"fixed_{m}", "end_cells"), _mean(0, f"aur_{m}", "end_cells")) for m in ("pca2", "pca8", "ae2", "ae8")}
retrain3 = {m: (_mean(3, f"fixed_{m}", "end_cells"), _mean(3, f"aur_{m}", "end_cells")) for m in ("pca2", "pca8", "ae2", "ae8")}
recon_fixed = {m: float(np.mean([x for x in _cm(0, f"fixed_{m}")["recon"]])) for m in ("pca2", "ae2", "pca8", "ae8")}
trap_runs = {v: E.success_count(_cm(3, v)["runs"]) for v in C.VARIANTS}
p_pca2_paths = E.mannwhitney_p(_vals(3, "fixed_pca2", "path_cells"), _vals(3, "hand_style", "path_cells"))
n_worse = sum(1 for lv in C.STUDY_LEVELS for m in ("pca2", "pca8", "ae2", "ae8") if _mean(lv, f"aur_{m}", "end_cells") < _mean(lv, f"fixed_{m}", "end_cells") and E.mannwhitney_p(_vals(lv, f"aur_{m}", "end_cells"), _vals(lv, f"fixed_{m}", "end_cells")) < 0.05)
st.info(
    f"**Das richtige Vorwissen schlägt alles (freies Feld):** bei der Endposition enden die Elites in {e_end:.2f} verschiedenen Endzellen, bei 4 Wegpunkten in {e_way4:.2f}, bei den gelernten Beschreibungen in {lall[0]:.2f} bis {lall[1]:.2f} "
    f"(2 Werte {l2[0]:.2f} bis {l2[1]:.2f}, 8 Werte {l8[0]:.2f} bis {l8[1]:.2f}); alle Unterschiede zur Endposition haben p < 0.001. Auf Stufe 3 sind es {t_end:.2f} gegen {t_way4:.2f} (4 Wegpunkte) und {t_all[0]:.2f} bis {t_all[1]:.2f} (gelernt). "
    f"**Das falsche Vorwissen (Fahrstil) ist schlechter als jede gelernte Beschreibung bei den Endzellen:** {e_style:.2f} im freien Feld und {t_style:.2f} auf Stufe 3 (alle Unterschiede p < 0.001). Bei den Wegzellen auf Stufe 3 ({t_style_paths:.2f}) liegt die PCA mit {t_pca[0]:.2f} bis {t_pca[1]:.2f} darüber "
    f"(belegt nur bei der einmal gelernten PCA mit 2 Werten, {_pstr(p_pca2_paths)}; bei den anderen drei p > 0.05), der Autoencoder mit {t_ae[0]:.2f} bis {t_ae[1]:.2f} darunter (alle vier p < 0.01). "
    f"**Neu lernen (AURORA) ist nicht durchgehend besser als einmal Lernen:** Endzellen im freien Feld, einmal gelernt gegen AURORA: PCA 2 Werte {retrain['pca2'][0]:.2f} / {retrain['pca2'][1]:.2f}, PCA 8 Werte {retrain['pca8'][0]:.2f} / {retrain['pca8'][1]:.2f}, "
    f"Autoencoder 2 Werte {retrain['ae2'][0]:.2f} / {retrain['ae2'][1]:.2f}, Autoencoder 8 Werte {retrain['ae8'][0]:.2f} / {retrain['ae8'][1]:.2f}; auf Stufe 3 liegt AURORA bei {'allen vier' if all(b <= a for a, b in retrain3.values()) else 'nicht allen'} Paaren gleich oder darunter. Von den acht Paaren ist AURORA nur im freien Feld mit dem Autoencoder (2 Werte) belegt besser, bei {n_worse} belegt schlechter (p < 0.05). "
    f"**Der Autoencoder rekonstruiert hier nicht besser als die PCA** (einmal gelernt, Fehler auf den Rohdaten der Elites im freien Feld: bei 2 Werten {recon_fixed['ae2']:.4f} gegen {recon_fixed['pca2']:.4f}, bei 8 Werten {recon_fixed['ae8']:.4f} gegen {recon_fixed['pca8']:.4f}); die Ursache ist nicht untersucht, "
    f"vermutlich ist er mit 400 Schritten wenig trainiert. Ein kleiner Rekonstruktionsfehler heißt nicht viele Endzellen (siehe Abbildung). "
    f"**In der Falle (Stufe 3)** erreichen das Ziel {trap_runs['hand_end']} von {N} Läufen mit der Endposition, {trap_runs['hand_style']} mit dem Fahrstil und {min(trap_runs[v] for v in LEARNED)} bis {max(trap_runs[v] for v in LEARNED)} mit den gelernten Beschreibungen: "
    "die Zahl der Endzellen sagt nicht, wer das Ziel erreicht."
)

st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Rohdaten sind gut gewählt** | Die gelernten Beschreibungen sehen 20 Positionen der Fahrt, also schon den Ort; AURORA sollte aus **Sensordaten** lernen (Bilder, Entfernungsmesser), die hier fehlen. Ob es aus ihnen etwas findet, ist nicht gemessen. | kein Folgestück |
| **Die Endzellen messen Vielfalt** | Sie bevorzugen Beschreibungen, die den Ort enthalten (Endposition); eine Beschreibung des Fahrstils kann bei anderen Maßen vorn liegen. Die Wegzellen sind im freien Feld gesättigt (fast alle 144). | kein Folgestück |
| **Ein kleiner Autoencoder genügt** | 40 → 16 → Code → 16 → 40, 400 Schritte, ein Zufallsstart: er rekonstruiert hier nicht besser als die PCA. Tiefere Netze, längeres Training oder mehrere Starts wurden nicht gemessen. | kein Folgestück |
| **Neu lernen lohnt** | Die Neulern-Zeitpunkte (10, 30, 60, 100, 150, 250) und die Wahl der Trainingsdaten sind eigene Festlegungen; die Referenz verwaltet das Archiv anders (Abstandsschwelle statt CVT-Zellen). | kein Folgestück |
| **Ein besseres Archiv löst das Ziel** | Stufe 4 hängt am Suchraum (Stück 3): auch eine gelernte Beschreibung löst das Ziel nicht. | **Go-Explore** und **POET** (Folgestücke) |
"""
)
st.caption(
    "Verwandt im Portfolio: [deception-maze-demo](https://sebastianhanisch-deception-maze-demo.streamlit.app/) (Stück 1), [novelty-search-demo](https://sebastianhanisch-novelty-search-demo.streamlit.app/) (Stück 2), "
    "[behavior-space-demo](https://sebastianhanisch-behavior-space-demo.streamlit.app/) (Stück 3), [novelty-objective-demo](https://sebastianhanisch-novelty-objective-demo.streamlit.app/) (Stück 4), "
    "[map-elites-demo](https://sebastianhanisch-map-elites-demo.streamlit.app/) (Stück 5), [cma-me-demo](https://sebastianhanisch-cma-me-demo.streamlit.app/) (Stück 6), "
    "[cvt-map-elites-demo](https://sebastianhanisch-cvt-map-elites-demo.streamlit.app/) (Stück 7), [autoencoder-demo](https://sebastianhanisch-autoencoder-demo.streamlit.app/) (der Autoencoder allein)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Rohdaten:** $r(g) = \big(x_{t_1}, y_{t_1}, \dots, x_{t_{20}}, y_{t_{20}}\big)/N \in [0,1]^{40}$, $t_i = \lfloor T\, i/20\rfloor$, $T = 140$, $N = 24$.

**PCA:** $\bar r$ Mittelwert der Trainingsdaten, $V_d$ die ersten $d$ rechten Singulärvektoren von $R - \bar r$: Code $c = V_d (r - \bar r)$. **Autoencoder:** $c = W_2\tanh(W_1 z + b_1) + b_2$ mit $z = (r - \mu)/\sigma$ und Decoder $\hat z = W_4 \tanh(W_3 c + b_3) + b_4$;
Verlust $\tfrac{1}{40 n}\lVert \hat Z - Z\rVert_F^2$, Adam, 400 Schritte, Lernrate 0.01.

**Zellen:** der Code wird mit Mittel und Standardabweichung der Trainingscodes standardisiert, $\tilde c$; $K = 256$ Schwerpunkte $m_j$ aus einem k-Means auf den standardisierten Trainingscodes; Zelle $z = \arg\min_j \lVert \tilde c - m_j\rVert$.

**AURORA:** Trainingsdaten $\mathcal{D}_t$ = 500 Zufallsfahrten, Rohdaten aller Elites, Kinder der Generation $t$; nach dem Neulernen werden die Elites nach Qualität absteigend in die neuen Zellen eingetragen (bei Kollision bleibt die bessere).

**Qualität und Maße:** Direktheit $q(g) = \lVert p_T - p_0\rVert / \sum_t \lVert p_t - p_{t-1}\rVert$; QD-Score $=\sum_z q(g^\ast_z)$; Endzellen $\lvert\{(\lfloor x_T/2\rfloor, \lfloor y_T/2\rfloor)\}\rvert$ über die Elites; Wegzellen $\bigl\lvert\bigcup_z \{(\lfloor x_t/2\rfloor, \lfloor y_t/2\rfloor)\}_t\bigr\rvert$; beide höchstens $12^2 = 144$.

Implementiert in `aur_learn.py` (PCA, Autoencoder), `aur_space.py` (Merkmale, Zellen) und `aur_search.py` (Archiv, AURORA-Schleife, Kennzahlen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html))."
)
