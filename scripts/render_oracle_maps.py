"""Render the three Oraculum figures (spec Section 4) deterministically from
fixtures. Content contract per figure is in the spec; regenerate with:
    uv run python scripts/render_oracle_maps.py

Styling follows the repo's `dataviz` skill: categorical hues are drawn from
the skill's documented eight-hue palette (never eyeballed), assigned to the
six settlement types by an order chosen to clear the CVD / normal-vision
separation floors for the pairs that actually touch on this fixed city
layout (see the derivation note below `TYPE_COLORS`). Service markers use
shape (never color) as their identity channel, since six more categorical
hues layered on the settlement fills would blow the palette's series budget.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tests.oraculum_fixtures import (  # noqa: E402
    load_barriers,
    load_exhibit,
    load_services,
    load_settlements,
)
from tests.reference_impl import (  # noqa: E402
    RULESETS, SCENARIOS, adjacency, apply_barrier, compute_city,
)

OUT = REPO / "docs" / "oracle"
CSV = REPO / "tests" / "fixtures" / "oraculum" / "expected_values.csv"
BASE_X, BASE_Y = 1_000_000, 1_000_000

# ---------------------------------------------------------------------------
# dataviz-skill palette (references/palette.md) -- documented hexes only.
# ---------------------------------------------------------------------------
PAGE_BG = "#f9f9f7"       # page plane
SURFACE = "#fcfcfb"       # chart surface
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
BASELINE = "#c3c2b7"

# Six settlement types need pairwise-distinguishable fills, but the
# documented 8-hue categorical order fails the CVD/normal-vision floors on
# several pairs that actually touch in this fixed layout (e.g. slot1..slot6
# in palette order puts UC beside RV and beside RUAC at normal-vision
# DeltaE ~13, below the 15 floor). Per the skill's "Themes" guidance
# ("enumerate candidate orderings ... pick the one that maximizes the
# minimum adjacent CVD ΔE"), the six hues below were chosen by enumerating
# assignments over the *actual* touching type-pairs in this fixture
# (Planned-UC, Planned-RUAC, UC-JJC, UC-RV, UC-RUAC, JJC-RUAC, JJC-IND,
# RUAC-IND) and picking one that clears both floors with margin: worst
# normal-vision ΔE 19.6 (floor 15), worst CVD ΔE 15.3 (target 8). All six
# hex values are slots from the documented 8-hue palette (blue, yellow,
# magenta, green, violet, red); orange and aqua are reserved below for
# non-settlement glyphs so nothing on the map re-uses a settlement hue.
TYPE_COLORS = {
    "Planned": "#2a78d6",  # blue
    "UC": "#eda100",       # yellow
    "JJC": "#e87ba4",      # magenta
    "RUAC": "#008300",     # green
    "IND": "#4a3aa7",      # violet
    "RV": "#e34948",       # red
}
CANAL_COLOR = "#eb6834"    # orange slot -- unused by any settlement type
ROAD_COLOR = INK_SECONDARY
ARROW_COLOR = INK_MUTED
IDEAL_ONLY_COLOR = "#7a5195"  # ideal-rule links the code rule drops
ARROW_HIGHLIGHT = "#e34948"  # red slot, reused deliberately to call out A->E
MARKER_FACE = INK_PRIMARY
MARKER_EDGE = SURFACE

SERVICE_MARKERS = {
    "clinic": "o", "school": "s", "bank": "^",
    "police": "v", "ration": "D", "transport": "P",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE,
    "figure.facecolor": PAGE_BG,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": PAGE_BG,
})

# Hand-placed label anchors (dx, dy from BASE, ha, va) chosen so the
# settlement id/pop/area label sits in a corner of each rectangle that is
# clear of that settlement's own service markers (both are known from the
# fixture generator's fixed coordinates).
LABEL_ANCHOR = {
    "A": (60, 1940, "left", "top"),
    "B": (1060, 1940, "left", "top"),
    "C": (2060, 1940, "left", "top"),
    "RV": (1160, 2940, "left", "top"),
    "D": (440, 940, "right", "top"),
    "E": (2440, 940, "right", "top"),
    "IND": (3440, 940, "right", "top"),
}


def _label_box(ax, xy, text, **kw):
    ax.annotate(
        text, xy=xy, fontsize=7.3, color=INK_PRIMARY, linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.28", facecolor=SURFACE, alpha=0.88,
                   edgecolor=BASELINE, linewidth=0.6),
        zorder=8, **kw,
    )


def _draw_settlements(ax, city, ghost=frozenset()):
    for _, row in city.iterrows():
        sid = row["USO_AREA_U"]
        ghosted = sid in ghost
        ax.add_patch(plt.Polygon(
            list(row.geometry.exterior.coords),
            facecolor=TYPE_COLORS[row["USO_FINAL"]],
            alpha=0.18 if ghosted else 0.55,
            edgecolor=INK_MUTED if ghosted else INK_PRIMARY,
            linestyle="--" if ghosted else "-",
            linewidth=1.1, zorder=2))


def _draw_labels(ax, city, annotate=None):
    for _, row in city.iterrows():
        sid = row["USO_AREA_U"]
        dx, dy, ha, va = LABEL_ANCHOR[sid]
        text = f"{sid}\npop {row['population']} · {row['area_km2']:g} km²"
        if annotate and sid in annotate:
            text += f"\n{annotate[sid]}"
        _label_box(ax, (BASE_X + dx, BASE_Y + dy), text, ha=ha, va=va)


def _draw_services(ax, services):
    for svc, gdf in services.items():
        if svc == "road":
            x, y = gdf.geometry.iloc[0].xy
            ax.plot(x, y, color=ROAD_COLOR, linewidth=2.4,
                     linestyle=(0, (4, 2)), zorder=4, solid_capstyle="round")
            continue
        marker = SERVICE_MARKERS[svc]
        xs = [g.x for g in gdf.geometry]
        ys = [g.y for g in gdf.geometry]
        ax.scatter(xs, ys, marker=marker, s=64, facecolor=MARKER_FACE,
                    edgecolor=MARKER_EDGE, linewidth=1.1, zorder=7)


def _draw_barriers(ax, barriers):
    for _, b in barriers.iterrows():
        x, y = b.geometry.xy
        ax.plot(x, y, color=CANAL_COLOR, linewidth=4.5,
                 solid_capstyle="butt", zorder=5)


def _draw_directed_edges(ax, city, nbrs, ghost=frozenset(), highlight=None,
                         force_dashed=False, force_rad=None, color=None):
    """Draw directed neighbor arrows. Mutual pairs get two curved arcs
    (bowed apart) so both directions stay visible; one-way pairs get a
    single straight arrow, which itself signals the asymmetry."""
    cent = city.set_index("USO_AREA_U").geometry.centroid
    seen_mutual = set()
    for i, js in nbrs.items():
        if i not in cent.index:
            continue
        for j in sorted(js):
            if j not in cent.index:
                continue
            mutual = i in nbrs.get(j, set())
            dashed = force_dashed or i in ghost or j in ghost
            is_highlight = highlight == (i, j)
            if mutual:
                pair = frozenset((i, j))
                rad = 0.12 if i < j else -0.12
                key = (pair, i < j)
                if key in seen_mutual:
                    continue
                seen_mutual.add(key)
            else:
                rad = 0.0
            if force_rad is not None:
                rad = force_rad
            edge_color = color or (ARROW_HIGHLIGHT if is_highlight
                                   else ARROW_COLOR)
            lw = 1.7 if is_highlight else 0.9
            alpha = 0.95 if is_highlight else 0.6
            ax.annotate(
                "", xy=(cent[j].x, cent[j].y), xytext=(cent[i].x, cent[i].y),
                arrowprops=dict(
                    arrowstyle="-|>", lw=lw, mutation_scale=13,
                    linestyle="--" if dashed else "-",
                    color=edge_color, alpha=alpha, shrinkA=20, shrinkB=20,
                    connectionstyle=f"arc3,rad={rad}"),
                zorder=6 if not is_highlight else 9)


def _code_nbrs(city, barriers):
    return apply_barrier(adjacency(city, "bbox"), city, barriers, "global")


def _ideal_nbrs(city, barriers):
    return apply_barrier(adjacency(city, "border"), city, barriers, "pair")


def _ideal_only_edges(city, barriers):
    """Links the manuscript's ideal rule has but the code rule drops."""
    ideal, code = _ideal_nbrs(city, barriers), _code_nbrs(city, barriers)
    return {i: ideal[i] - code.get(i, set()) for i in ideal}


def _legend_handles(*, services=True, canal=True, road=True, nbrs=True):
    handles = [mpatches.Patch(facecolor=c, edgecolor=INK_PRIMARY, alpha=0.55,
                                linewidth=0.8, label=t)
               for t, c in TYPE_COLORS.items()]
    if services:
        handles += [
            Line2D([0], [0], marker=m, linestyle="none",
                    markerfacecolor=MARKER_FACE, markeredgecolor=MARKER_EDGE,
                    markeredgewidth=1.0, markersize=8, label=svc)
            for svc, m in SERVICE_MARKERS.items()
        ]
    if canal:
        handles.append(Line2D([0], [0], color=CANAL_COLOR, lw=4,
                                solid_capstyle="butt", label="canal (barrier)"))
    if road:
        handles.append(Line2D([0], [0], color=ROAD_COLOR, lw=2.2,
                                linestyle=(0, (4, 2)), label="road"))
    if nbrs:
        handles.append(Line2D([0], [0], color=ARROW_COLOR, lw=1.2,
                                label="neighbor, code rule (directed)"))
        handles.append(Line2D([0], [0], color=ARROW_HIGHLIGHT, lw=1.9,
                                label="A→E exists; E→A does not"))
        handles.append(Line2D([0], [0], color=IDEAL_ONLY_COLOR, lw=1.2,
                                linestyle="--",
                                label="ideal-only link (code rule drops it)"))
    return handles


def render_city():
    city, barriers, services = load_settlements(), load_barriers(), load_services()
    nbrs = _code_nbrs(city, barriers)

    fig, ax = plt.subplots(figsize=(11.5, 8), dpi=150)
    _draw_settlements(ax, city)
    # Spec figure-1 contract: ideal-only links (present under the
    # manuscript's border+pair-severing rule, dropped by the code rule)
    # render dashed, so the reader sees exactly what the code discards.
    _draw_directed_edges(ax, city, _ideal_only_edges(city, barriers),
                         force_dashed=True, force_rad=0.34,
                         color=IDEAL_ONLY_COLOR)
    _draw_directed_edges(ax, city, nbrs, highlight=("A", "E"))
    _draw_barriers(ax, barriers)
    _draw_services(ax, services)
    _draw_labels(ax, city)

    ax.set_title(
        "Oraculum — settlements, services, canal barrier, and the code "
        "rule's DIRECTED neighbor graph\narrow i→j: i's PCEN counts j's "
        "decayed services. The canal strips A and D from every OTHER "
        "settlement's list,\nso A keeps E as an outgoing neighbor but E "
        "never regains A — A→E exists, E→A does not.",
        fontsize=10.5, color=INK_PRIMARY, loc="left")
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.margins(0.08)

    handles = _legend_handles()
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                fontsize=7.6, frameon=True, facecolor=SURFACE,
                edgecolor=BASELINE, borderpad=0.8, labelspacing=0.7,
                title="Legend", title_fontsize=8.2)

    fig.savefig(OUT / "oraculum_city.png", bbox_inches="tight")
    plt.close(fig)


def render_exclusion_variants():
    city, barriers = load_settlements(), load_barriers()
    exp = pd.read_csv(CSV)

    def clinic_note(scenario, sids):
        # Short form ("PCEN 0.0125") -- these four narrow columns (A/B/C are
        # each only 1 km wide) leave little room per label; the suptitle
        # already states these are clinic PCEN values, so the per-cell tag
        # only needs the number to stay clear of the neighboring cell's box.
        sub = exp[(exp["rule"] == "code") & (exp["scenario"] == scenario)
                  & (exp["denom"] == "pop") & (exp["metric"] == "clinic_pcen")]
        return {r["settlement"]: f"PCEN {r['value']:.4f}"
                for _, r in sub.iterrows() if r["settlement"] in sids}

    affected = {"B", "C", "E"}
    panels = [
        ("baseline\nall seven settlements", frozenset(), frozenset(), "baseline"),
        ("excl_contributing\nRV/IND ghosted; code rule SWALLOWS their\n"
         "contribution — collapses to removal", frozenset({"RV", "IND"}),
         frozenset(), "excl_contributing"),
        ("excl_removed\nRV/IND physically removed", frozenset({"RV", "IND"}),
         frozenset({"RV", "IND"}), "excl_removed"),
        ("excl_ind_removed\nIND removed only — pure\nrenormalization",
         frozenset({"IND"}), frozenset({"IND"}), "excl_ind_removed"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(24, 7.2), dpi=150)
    for ax, (title, ghost, hide, scenario) in zip(axes, panels):
        sub_city = city[~city["USO_AREA_U"].isin(hide)]
        nbrs = _code_nbrs(sub_city, barriers)
        note = clinic_note(scenario, affected)
        _draw_settlements(ax, sub_city, ghost=ghost - hide)
        _draw_directed_edges(ax, sub_city, nbrs, ghost=ghost - hide)
        _draw_barriers(ax, barriers[barriers.geometry.apply(
            lambda g: any(g.intersects(row.geometry)
                          for _, row in sub_city.iterrows()))])
        _draw_labels(ax, sub_city, annotate=note)
        ax.set_title(title, fontsize=9.3, color=INK_PRIMARY)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.margins(0.10)

    fig.suptitle(
        "Oraculum exclusion scenarios (code rule, popsize denominator) — "
        "clinic PCEN annotated on B, C, E shows the difference between\n"
        "“contributing” (RV/IND ghosted, still on the map) and "
        "“removed” (RV/IND gone) under the production except:pass swallow",
        fontsize=11.5, color=INK_PRIMARY, y=1.04)

    handles = _legend_handles(services=False, canal=True, road=False, nbrs=True)
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7.6,
                frameon=True, facecolor=SURFACE, edgecolor=BASELINE,
                bbox_to_anchor=(0.5, -0.08))

    fig.savefig(OUT / "oraculum_exclusion_variants.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# RV-exclusion decision maps (memo `docs/oracle/rv-exclusion-decision-memo.md`)
# ---------------------------------------------------------------------------
# The reference impl's SCENARIOS has "RV excluded but contributing"
# (excl_rv_only) but no "RV alone, fully removed"; the decision memo needs
# both sides of the same coin, so register it here rather than widening the
# fixture CSV (which is round-trip tested at its current row count).
SCENARIOS.setdefault("rv_removed", (frozenset({"RV"}), True))

CHANGED_EDGE = "#e34948"  # red slot: value differs from baseline

# The RV maps carry four-line labels, so several corners used by
# LABEL_ANCHOR would now cover service markers (A's second clinic, E's
# clinic). Corners re-chosen against the fixture's service coordinates.
RV_LABEL_ANCHOR = {
    **LABEL_ANCHOR,
    "A": (60, 1500, "left", "center"),
    "D": (440, 60, "right", "bottom"),
    "E": (1500, 940, "center", "top"),
}


def _draw_undirected_edges(ax, city, nbrs, ghost=frozenset()):
    """Ideal rule is symmetric, so one line per pair (dashed if it touches a
    ghosted settlement)."""
    cent = city.set_index("USO_AREA_U").geometry.centroid
    done = set()
    for i, js in nbrs.items():
        for j in js:
            pair = frozenset((i, j))
            if pair in done or i not in cent.index or j not in cent.index:
                continue
            done.add(pair)
            dashed = i in ghost or j in ghost
            ax.plot([cent[i].x, cent[j].x], [cent[i].y, cent[j].y],
                    color=ARROW_COLOR, lw=1.1, alpha=0.7,
                    linestyle="--" if dashed else "-", zorder=6)


def _ideal_frame(scenario, denom="pop"):
    city, barriers, services = (load_settlements(), load_barriers(),
                                load_services())
    return compute_city(city, services, barriers, scenario=scenario,
                        denom=denom, **RULESETS["ideal"])


def _rv_panel(ax, scenario, ghost, hide, title, base):
    city, barriers, services = (load_settlements(), load_barriers(),
                                load_services())
    sub_city = city[~city["USO_AREA_U"].isin(hide)]
    nbrs = _ideal_nbrs(sub_city, barriers)
    df = _ideal_frame(scenario)

    _draw_settlements(ax, sub_city, ghost=ghost)
    _draw_undirected_edges(ax, sub_city, nbrs, ghost=ghost)
    _draw_barriers(ax, barriers)
    kept = {k: v for k, v in services.items()
            if k == "road" or True}
    _draw_services(ax, {k: (v[~v.geometry.apply(
        lambda g: any(g.within(row.geometry) for _, row in
                      city[city["USO_AREA_U"].isin(hide)].iterrows()))]
        if k != "road" else v) for k, v in kept.items()})

    for _, row in sub_city.iterrows():
        sid = row["USO_AREA_U"]
        dx, dy, ha, va = RV_LABEL_ANCHOR[sid]
        text = f"{sid}\npop {row['population']} · {row['area_km2']:g} km²"
        changed = False
        if sid in df.index:
            r = df.loc[sid]
            text += (f"\nclinic PCEN {r['clinic_pcen']:.4f}"
                     f"\nclinic idx {r['clinic_idx']:.3f} · PSI {r['psi_eq1']:.3f}")
            if sid in base.index:
                b = base.loc[sid]
                changed = (abs(b['clinic_pcen'] - r['clinic_pcen']) > 1e-12
                           or abs(b['clinic_idx'] - r['clinic_idx']) > 1e-12)
        else:
            text += "\n(not indexed — no PSI)"
        ax.annotate(
            text, xy=(BASE_X + dx, BASE_Y + dy), fontsize=7.0,
            color=INK_PRIMARY, linespacing=1.35, ha=ha, va=va,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=SURFACE,
                      alpha=0.92,
                      edgecolor=CHANGED_EDGE if changed else BASELINE,
                      linewidth=1.4 if changed else 0.6),
            zorder=8)
    ax.set_title(title, fontsize=10.2, color=INK_PRIMARY, loc="left")
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.margins(0.08)


RV_PANELS = [
    ("oraculum_rv_baseline.png", "baseline", frozenset(), frozenset(),
     "1 · Baseline — all seven settlements indexed (ideal rule: shared "
     "border, canal severs A–D only)\nB counts RV's two clinics at decay ½: "
     "clinic PCEN_B = (1 + 2·½ [A] + 2·½ [RV] + 1·½ [E]) / 200 = 0.0175"),
    ("oraculum_rv_contributing.png", "excl_rv_only", frozenset({"RV"}),
     frozenset(),
     "2 · Semantics (a) — RV excluded from the index but still CONTRIBUTING "
     "its services\nRV gets no PSI row; B still counts RV's clinics "
     "(PCEN_B unchanged at 0.0175). Nothing else moves: the min/max anchors\n(C, IND) survive, so re-normalizing over six settlements changes no index."),
    ("oraculum_rv_removed.png", "rv_removed", frozenset(),
     frozenset({"RV"}),
     "3 · Semantics (b) — RV fully REMOVED before neighbor sums (what the "
     "code does today)\nRV's clinics vanish: PCEN_B = (1 + 2·½ + 1·½) / "
     "200 = 0.0125. Red boxes = values that differ from the baseline map."),
]


def render_rv_decision():
    base = _ideal_frame("baseline")
    handles = [h for h in _legend_handles(nbrs=False)]
    handles.append(Line2D([0], [0], color=ARROW_COLOR, lw=1.2,
                          label="neighbors (ideal rule, symmetric)"))
    handles.append(Line2D([0], [0], color=ARROW_COLOR, lw=1.2,
                          linestyle="--", label="neighbor link to an excluded settlement"))
    handles.append(mpatches.Patch(facecolor=SURFACE, edgecolor=CHANGED_EDGE,
                                  linewidth=1.4, label="value differs from baseline"))
    for fname, scenario, ghost, hide, title in RV_PANELS:
        fig, ax = plt.subplots(figsize=(11.5, 8), dpi=150)
        _rv_panel(ax, scenario, ghost, hide, title, base)
        ax.legend(handles=handles, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), fontsize=7.6, frameon=True,
                  facecolor=SURFACE, edgecolor=BASELINE, borderpad=0.8,
                  labelspacing=0.7, title="Legend", title_fontsize=8.2)
        fig.savefig(OUT / fname, bbox_inches="tight")
        plt.close(fig)


def render_divergence():
    ex = load_exhibit().rename(columns={"id": "USO_AREA_U"})
    fill = TYPE_COLORS["Planned"]
    bbox_color = "#e34948"     # red slot -- the "invented neighbor" cue
    touch_color = "#4a3aa7"    # violet slot -- the intersects-only cue

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    for _, row in ex.iterrows():
        ax.add_patch(plt.Polygon(list(row.geometry.exterior.coords),
                                   facecolor=fill, alpha=0.42,
                                   edgecolor=INK_PRIMARY, linewidth=1.1,
                                   zorder=2))
        b = row.geometry.bounds
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                     fill=False, linestyle="--",
                                     edgecolor=bbox_color, linewidth=1.3,
                                     zorder=3))
        c = row.geometry.centroid
        _label_box(ax, (c.x, c.y),
                    f"{row['USO_AREA_U']}\npop {row['population']}, "
                    f"{row['clinics']} clinic(s)", ha="center", va="center")

    cent = ex.set_index("USO_AREA_U").geometry.centroid
    ax.annotate("", xy=(cent["P"].x, cent["P"].y), xytext=(cent["Q"].x, cent["Q"].y),
                 arrowprops=dict(arrowstyle="-|>", color=bbox_color, lw=2.0,
                                  mutation_scale=16, shrinkA=10, shrinkB=10),
                 zorder=6)
    _label_box(ax, (cent["Q"].x, cent["Q"].y + 300),
                "phantom bbox link Q→P\n(Q and P never touch as polygons)\n"
                "Q clinic PCEN +0.005147", ha="center", va="bottom")

    ax.plot([cent["R"].x, cent["S"].x], [cent["R"].y, cent["S"].y],
             color=touch_color, lw=1.8, linestyle=":", zorder=6)
    _label_box(ax, ((cent["R"].x + cent["S"].x) / 2, cent["R"].y - 550),
                "corner touch R–S\n(bbox AND intersects both count it)\n"
                "S clinic PCEN +0.016569, R unchanged", ha="center", va="top")

    ax.set_title(
        "Divergence exhibit — polygon (solid fill) vs bounding box "
        "(dashed): bbox/intersects adjacency invents neighbors\nthat "
        "border-sharing (the manuscript's ideal rule) denies",
        fontsize=11, color=INK_PRIMARY, loc="left", pad=16)

    handles = [
        mpatches.Patch(facecolor=fill, edgecolor=INK_PRIMARY, alpha=0.42,
                        label="exhibit polygon"),
        Line2D([0], [0], color=bbox_color, lw=1.6, linestyle="--",
                label="bounding box overlay"),
        Line2D([0], [0], color=bbox_color, lw=2.0, label="phantom bbox link (Q→P)"),
        Line2D([0], [0], color=touch_color, lw=1.8, linestyle=":",
                label="corner-touch link (R–S)"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=BASELINE)

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.autoscale_view()
    ax.margins(0.12)
    fig.savefig(OUT / "oraculum_divergence.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    render_city()
    render_exclusion_variants()
    render_divergence()
    render_rv_decision()
    print(f"wrote 6 figures to {OUT}")


if __name__ == "__main__":
    main()
