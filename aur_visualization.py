"""Plotly-Abbildungen der AURORA-Demo: Endpositionen der Elites (nach Direktheit gefärbt), die Karte des gelernten Codes (nur bei 2 Werten), Verläufe von Endzellen und Wegzellen, Endzellen und Wegzellen je Variante (Studie),
Endzellen über den Evaluationen, Rekonstruktionsfehler gegen Endzellen. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen. Das Raster `grid[i, j]` wird mit i als x-Achse gezeichnet."""

import numpy as np
import plotly.graph_objects as go

import aur_constants as C
import aur_evaluation as E

WALL_COLOR = "#3a3f47"
START_COLOR, GOAL_COLOR = "#54a24b", "#e45756"
GROUP_COLORS = {"hand": "#f16913", "fixed": "#9ecae1", "aur": "#08519c"}
VARIANT_COLORS = {"hand_end": "#a63603", "hand_way4": "#f16913", "hand_style": "#fdae6b", "fixed_pca2": "#c6dbef", "fixed_pca8": "#9ecae1", "fixed_ae2": "#6baed6", "fixed_ae8": "#4292c6",
                  "aur_pca2": "#2171b5", "aur_pca8": "#08519c", "aur_ae2": "#08306b", "aur_ae8": "#41b6c4"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.25, top=10):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=top, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _arena_axes(fig):
    """Beide Achsen 0 bis N mit gleichem Maßstab (quadratische Zellen), Zeichenfläche passt sich der kleineren Seite an (constrain=domain)."""
    fig.update_xaxes(range=[0, C.N], showgrid=False, zeroline=False, title_text="x (Zellen)", constrain="domain")
    fig.update_yaxes(range=[0, C.N], showgrid=False, zeroline=False, title_text="y (Zellen)", scaleanchor="x", scaleratio=1, constrain="domain")


def _walls(fig, grid):
    z = np.where(grid, 1.0, np.nan)
    fig.add_trace(go.Heatmap(z=z.T, x0=0.5, dx=1, y0=0.5, dy=1, zmin=0, zmax=1, colorscale=[[0, WALL_COLOR], [1, WALL_COLOR]], showscale=False, hoverinfo="skip", xgap=0, ygap=0))


def _markers(fig):
    fig.add_trace(go.Scatter(x=[C.START[0]], y=[C.START[1]], mode="markers", marker=dict(color=START_COLOR, size=12, symbol="square", line=dict(color="white", width=1.5)), name="Start"))
    fig.add_trace(go.Scatter(x=[C.GOAL[0]], y=[C.GOAL[1]], mode="markers", marker=dict(color=GOAL_COLOR, size=15, symbol="star", line=dict(color="white", width=1.5)), name="Ziel"))
    fig.add_shape(type="circle", xref="x", yref="y", x0=C.GOAL[0] - C.SUCCESS_RADIUS, x1=C.GOAL[0] + C.SUCCESS_RADIUS, y0=C.GOAL[1] - C.SUCCESS_RADIUS, y1=C.GOAL[1] + C.SUCCESS_RADIUS,
                  line=dict(color=GOAL_COLOR, dash="dot", width=1.5))


def build_elite_map(grid, ends, quality):
    """Endpositionen aller Elites (ein Punkt je belegter Zelle) im Labyrinth, nach Direktheit gefärbt (hell = direkter)."""
    fig = go.Figure()
    _walls(fig, grid)
    fig.add_trace(go.Scatter(x=ends[:, 0], y=ends[:, 1], mode="markers", name="Endpunkte der Elites", hoverinfo="skip",
                             marker=dict(color=quality, colorscale="Viridis", cmin=0, cmax=1, size=6, line=dict(color="white", width=0.4),
                                         colorbar=dict(title=dict(text="Direktheit des Elites", side="top"), orientation="h", thickness=10, len=0.9, y=-0.2, yanchor="top"))))
    _markers(fig)
    _arena_axes(fig)
    return _base(fig, 470, legend_y=-0.55, top=0)


def build_latent_map(codes, centroids, end_x):
    """Karte des Codes bei 2 Werten: die Schwerpunkte (grau) und die Elites, nach dem x der Endposition gefärbt (hell = weiter rechts im Labyrinth). Zeigt, ob der Code den Ort enthält."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=centroids[:, 0], y=centroids[:, 1], mode="markers", name="Schwerpunkte", hoverinfo="skip", marker=dict(color="#bbbbbb", size=4)))
    fig.add_trace(go.Scatter(x=codes[:, 0], y=codes[:, 1], mode="markers", name="Elites", hoverinfo="skip",
                             marker=dict(color=end_x, colorscale="Viridis", cmin=0, cmax=C.N, size=6, line=dict(color="white", width=0.4),
                                         colorbar=dict(title=dict(text="x der Endposition (Zellen)", side="top"), orientation="h", thickness=10, len=0.9, y=-0.25, yanchor="top"))))
    fig.update_xaxes(title_text="Code 1 (standardisiert)", zeroline=False)
    fig.update_yaxes(title_text="Code 2 (standardisiert)", zeroline=False)
    return _base(fig, 400, legend_y=-0.6, top=0)


def _pair_curve(a_y, b_y, a, b, y_title, y_range=None):
    gens = list(range(1, len(a_y) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=a_y, mode="lines", line=dict(color=VARIANT_COLORS[a], width=2.5), name=C.VARIANT_LABELS[a]))
    fig.add_trace(go.Scatter(x=gens, y=b_y, mode="lines", line=dict(color=VARIANT_COLORS[b], width=2.5, dash="dot" if a == b else "solid"), name=C.VARIANT_LABELS[b]))
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text=y_title, rangemode="tozero", range=y_range)
    return _base(fig, 320, legend_y=-0.45)


def build_end_cells_curve(a_y, b_y, a, b):
    """Verschiedene 2 × 2-Endzellen der Elites (von 144) über den Generationen, zwei Beschreibungen."""
    return _pair_curve(a_y, b_y, a, b, "verschiedene Endzellen (von 144)", y_range=[0, C.END_CELLS_TOTAL * 1.03])


def build_path_cells_curve(a_y, b_y, a, b):
    """Wegzellen (2 × 2-Zellen, die irgendeine Elite-Fahrt berührt, von 144) über den Generationen, zwei Beschreibungen."""
    return _pair_curve(a_y, b_y, a, b, "Wegzellen (von 144)", y_range=[0, C.END_CELLS_TOTAL * 1.03])


def _bars(pre, level, key, y_title):
    fig = go.Figure()
    for v in C.VARIANTS:
        vals = np.array(E.study_cell(pre, level, v)[key], float)
        fig.add_trace(go.Bar(x=[C.VARIANT_LABELS[v]], y=[float(vals.mean())], name=C.VARIANT_LABELS[v], marker_color=VARIANT_COLORS[v], error_y=dict(type="data", array=[float(vals.std(ddof=1))], color="#555", thickness=1), showlegend=False))
    fig.update_yaxes(title_text=y_title, rangemode="tozero", range=[0, C.END_CELLS_TOTAL])
    fig.update_xaxes(tickangle=-35)
    return _base(fig, 470)


def build_end_cells_chart(pre, level):
    """Mittlere Zahl verschiedener Endzellen (von 144) nach 400 Generationen je Beschreibung; Fehlerbalken = Standardabweichung über die 20 Läufe."""
    return _bars(pre, level, "end_cells", "verschiedene Endzellen nach 400 Generationen (Mittel ± Standardabweichung)")


def build_path_cells_chart(pre, level):
    """Mittlere Zahl der Wegzellen (von 144) nach 400 Generationen je Beschreibung; Fehlerbalken = Standardabweichung über die 20 Läufe."""
    return _bars(pre, level, "path_cells", "Wegzellen nach 400 Generationen (Mittel ± Standardabweichung)")


def build_study_curve(pre, level):
    """Mittlere Zahl verschiedener Endzellen über den Evaluationen, ein Linienzug je Beschreibung."""
    fig = go.Figure()
    for v in C.VARIANTS:
        curve = E.study_cell(pre, level, v)["mean_end_cells"]
        fig.add_trace(go.Scatter(x=[(i + 1) * C.POP for i in range(len(curve))], y=curve, mode="lines", line=dict(color=VARIANT_COLORS[v], width=2.5), name=C.VARIANT_LABELS[v]))
    fig.update_xaxes(title_text="Evaluationen")
    fig.update_yaxes(title_text="verschiedene Endzellen (Mittel, von 144)", rangemode="tozero", range=[0, C.END_CELLS_TOTAL * 1.03])
    return _base(fig, 440, legend_y=-0.9)


def build_recon_scatter(pre, level):
    """Rekonstruktionsfehler des Encoders auf den Rohdaten (logarithmisch) gegen die Zahl der Endzellen, je gelernter Variante: ein kleiner Fehler heißt nicht viele Endzellen."""
    fig = go.Figure()
    for v in C.VARIANTS:
        if C.VARIANT_MODE[v] == "hand":
            continue
        c = E.study_cell(pre, level, v)
        fig.add_trace(go.Scatter(x=[float(np.mean(c["recon"]))], y=[float(np.mean(c["end_cells"]))], mode="markers+text", text=[v.replace("_", " ")], textposition="top center", name=C.VARIANT_LABELS[v],
                                 marker=dict(color=VARIANT_COLORS[v], size=12), showlegend=False))
    fig.update_xaxes(title_text="Rekonstruktionsfehler (mittlerer quadratischer Fehler, Rohdaten)", type="log")
    fig.update_yaxes(title_text="verschiedene Endzellen (Mittel)", rangemode="tozero")
    return _base(fig, 400)
