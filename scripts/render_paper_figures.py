"""Paper figures from the DEL-32 run (DEL-33; spec
docs/superpowers/specs/2026-10-27-del-33-paper-figures-design.md).

Render:
    uv run python scripts/render_paper_figures.py --run-dir ~/delhi_data/phase4_ratified

Five figures: the headline planned-relative pair (bars + full-Delhi map),
the JJC-vs-Planned juxtaposition, and the two min-max appendix versions.
Styling copies the documented palette and rc block from
`scripts/render_oracle_maps.py`; only the seven reported types ever appear
as a category, and settlements without a PSI row are drawn grey.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

REPO = Path(__file__).resolve().parent.parent

ID_COL = "USO_AREA_U"
PROFILE = "ratified-2026"
REPORTED_TYPES = ("Planned", "UAC", "JJC", "RUAC", "UV", "SDA", "JJR")
DENOMINATORS = ("pop", "popdensity")
HEADLINE_DENOM = "popdensity"
RUN_GLOB = "delhi_psi_*_{denom}_2020.csv"

EXPECTED_FIGURES = (
    "fig4_psi_by_type.png",
    "fig5_psi_map.png",
    "fig6_jjc_vs_planned.png",
    "figA1_psi_by_type_minmax.png",
    "figA2_psi_map_minmax.png",
)

# --- documented palette (scripts/render_oracle_maps.py) ------------------
PAGE_BG = "#f9f9f7"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
BASELINE = "#c3c2b7"

# Seven reported types need seven hues: the oracle's six settlement slots
# (blue, yellow, magenta, green, violet, red) plus the documented orange
# slot, which the oracle reserved for non-settlement glyphs (this script
# draws no canal/road glyph, so it is free).
TYPE_COLORS = {
    "Planned": "#2a78d6",   # blue
    "UAC": "#eda100",       # yellow
    "JJC": "#e87ba4",       # magenta
    "RUAC": "#008300",      # green
    "UV": "#4a3aa7",        # violet
    "SDA": "#e34948",       # red
    "JJR": "#eb6834",       # orange
}
REL_CMAP = LinearSegmentedColormap.from_list(
    "psi_relative", ["#2a78d6", "#f2f1ec", "#e34948"])
NORM_CMAP = LinearSegmentedColormap.from_list(
    "psi_minmax", ["#f2f1ec", "#2a78d6"])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE,
    "figure.facecolor": PAGE_BG,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": PAGE_BG,
})

# Same suppression as the oracle renderer: no matplot version tag, so a
# re-render on one machine is byte-stable (the integration test's property).
PNG_METADATA = {"Software": None}

SCALE_COLUMNS = {"rel": "rel_psi", "norm": "norm_psi"}
SCALE_TITLES = {
    "rel": "PSI relative to the planned average (1 = average planned colony)",
    "norm": "PSI on the min-max scale (appendix)",
}
SERVICE_COLUMNS = (
    ("bank_pcen", "Bank"),
    ("health_pcen", "Health"),
    ("police_pcen", "Police"),
    ("ration_pcen", "Ration"),
    ("school_pcen", "School"),
    ("transport_pcen", "Transport"),
    ("road_pcen", "Road"),
)


class FigureError(Exception):
    """A one-line, user-facing input problem (never a traceback)."""


# --- pure data helpers ---------------------------------------------------
def load_run(run_dir):
    """Both denominator CSVs from a DEL-32 run directory."""
    run_dir = Path(run_dir).expanduser()
    run = {}
    for denom in DENOMINATORS:
        pattern = RUN_GLOB.format(denom=denom)
        matches = sorted(run_dir.glob(pattern))
        if len(matches) != 1:
            wanted = " / ".join(
                RUN_GLOB.format(denom=d) for d in DENOMINATORS)
            raise FigureError(
                f"{run_dir}: expected exactly one {pattern!r} "
                f"(both denominators: {wanted}); found {len(matches)}")
        run[denom] = pd.read_csv(matches[0])
    return run


def type_means(frame):
    """Per-type means over the seven reported types, descending by mean_rel.

    Types outside REPORTED_TYPES are dropped, so an excluded category can
    never leak into a bar chart even if a future run reports it.
    """
    kept = frame[frame["category"].isin(REPORTED_TYPES)]
    means = kept.groupby("category").agg(
        mean_rel=("rel_psi", "mean"),
        mean_norm=("norm_psi", "mean"),
        n=("rel_psi", "size"),
    )
    order = sorted(means.index, key=lambda c: (-means.loc[c, "mean_rel"], c))
    return means.loc[order]


def map_frame(settlements, frame):
    """Every settlement row, joined to the run; unscored rows get NaN."""
    got = settlements.merge(
        frame[[ID_COL, "category", "rel_psi", "norm_psi"]],
        on=ID_COL, how="left")
    got["scored"] = got["rel_psi"].notna()
    return got


def ramp_bounds(values):
    """The 1st/99th percentile pair used for a colour ramp's ends."""
    clean = pd.Series(values).dropna().astype(float)
    return float(clean.quantile(0.01)), float(clean.quantile(0.99))


# --- renderers -----------------------------------------------------------
def _save(fig, out_dir, name):
    path = Path(out_dir) / name
    fig.savefig(path, bbox_inches="tight", metadata=PNG_METADATA)
    plt.close(fig)
    return path


def render_psi_by_type(run, out_dir, *, scale="rel"):
    """Figure 4: seven types × two denominators, headlines then appendix."""
    column = SCALE_COLUMNS[scale]
    means_column = "mean_rel" if scale == "rel" else "mean_norm"
    means = {denom: type_means(run[denom]) for denom in DENOMINATORS}
    order = list(means[HEADLINE_DENOM].index)
    for denom in DENOMINATORS:                     # union, stable order
        order += [c for c in means[denom].index if c not in order]

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)
    y = np.arange(len(order))
    height = 0.38
    number_format = "{:.2f}" if scale == "rel" else "{:.3f}"
    for offset, denom, alpha in ((-height / 2, "pop", 0.55),
                                 (height / 2, HEADLINE_DENOM, 1.0)):
        values = [float(means[denom].loc[c, means_column])
                  if c in means[denom].index else np.nan for c in order]
        bars = ax.barh(y + offset, values, height=height,
                       color=[TYPE_COLORS[c] for c in order], alpha=alpha,
                       edgecolor=SURFACE, linewidth=0.6,
                       label=f"{denom} denominator")
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                ax.annotate(number_format.format(value),
                            (value, bar.get_y() + bar.get_height() / 2),
                            xytext=(3, 0), textcoords="offset points",
                            va="center", ha="left", fontsize=8,
                            color=INK_SECONDARY)
    if scale == "rel":
        ax.axvline(1.0, color=INK_MUTED, linewidth=1, linestyle="--")
        ax.text(1.0, -0.9, " average planned colony",
                color=INK_SECONDARY, fontsize=8, va="bottom")
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlabel("rel_psi" if scale == "rel" else "norm_psi")
    ax.set_title(SCALE_TITLES[scale], loc="left", fontsize=11)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(axis="x", color=BASELINE, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    name = ("fig4_psi_by_type.png" if scale == "rel"
            else "figA1_psi_by_type_minmax.png")
    return _save(fig, out_dir, name)


def render_psi_map(run, settlements, out_dir, *, scale="rel"):
    """Figure 5: full-Delhi map, grey where no PSI row exists."""
    column = SCALE_COLUMNS[scale]
    got = map_frame(settlements, run[HEADLINE_DENOM])
    scored, unscored = got[got["scored"]], got[~got["scored"]]

    fig, ax = plt.subplots(figsize=(10.5, 8), dpi=150)
    if not unscored.empty:
        unscored.plot(ax=ax, color=BASELINE, edgecolor=INK_MUTED,
                      linewidth=0.2, zorder=1)
    if scale == "rel":
        lo, hi = ramp_bounds(scored[column])
        norm = TwoSlopeNorm(vcenter=1.0, vmin=min(lo, 0.999),
                            vmax=max(hi, 1.001))
        cmap = REL_CMAP
    else:
        lo, hi = ramp_bounds(scored[column])
        norm, cmap = None, NORM_CMAP
    handle = scored.plot(ax=ax, column=column, cmap=cmap, norm=norm,
                         edgecolor=INK_MUTED, linewidth=0.2, zorder=2)
    ax.set_axis_off()
    ax.set_title(
        f"PSI by settlement — {HEADLINE_DENOM} denominator\n"
        f"{'relative scale (1 = average planned colony)' if scale == 'rel' else 'min-max scale (appendix)'}"
        f" — grey: no PSI ({len(unscored)} settlements)",
        loc="left", fontsize=11)
    bar = fig.colorbar(handle.collections[-1], ax=ax, shrink=0.55,
                       pad=0.01)
    bar.set_label("rel_psi" if scale == "rel" else "norm_psi")
    if scale == "rel":
        bar.ax.axhline(1.0, color=INK_PRIMARY, linewidth=1)
    name = "fig5_psi_map.png" if scale == "rel" else "figA2_psi_map_minmax.png"
    return _save(fig, out_dir, name)


def render_jjc_vs_planned(run, out_dir):
    """Figure 6: the juxtaposition — distributions and the PCEN mechanism."""
    frame = run[HEADLINE_DENOM]
    planned = frame.loc[frame["category"] == "Planned", "rel_psi"].dropna()
    jjc = frame.loc[frame["category"] == "JJC", "rel_psi"].dropna()
    share = float((jjc > planned.median()).mean())

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12, 5.6), dpi=150,
        gridspec_kw={"width_ratios": (1, 1.35)})

    # The distributions have a long tail (a handful of tiny colonies), so
    # the panel is clipped at the pooled 99th percentile and says how many
    # settlements fall above the clip — a log axis would hide the zeros.
    pooled = np.concatenate([planned.to_numpy(), jjc.to_numpy()])
    cap = float(max(np.nanquantile(pooled, 0.99), 1.0))
    hidden = 0
    for x, (label, values) in enumerate((("Planned", planned), ("JJC", jjc))):
        ordered = np.sort(values.to_numpy())
        shown = ordered[ordered <= cap]
        hidden += len(ordered) - len(shown)
        offsets = np.linspace(-0.18, 0.18, len(shown)) if len(shown) else []
        left.scatter(offsets + x, shown, s=16,
                     color=TYPE_COLORS[label], alpha=0.65,
                     edgecolor=SURFACE, linewidth=0.4)
        median = float(np.median(ordered))
        left.hlines(median, x - 0.28, x + 0.28,
                    color=INK_PRIMARY, linewidth=1.6)
        left.text(x + 0.32, median, f"median {median:.2f}",
                  va="center", fontsize=8, color=INK_SECONDARY)
    left.set_xlim(-0.5, 1.75)
    left.set_ylim(0, cap * 1.02)
    left.set_xticks([0, 1], ["Planned", "JJC"])
    left.set_ylabel("rel_psi")
    left.set_title(f"Distributions — {share:.0%} of JJC above the "
                   f"Planned median", loc="left", fontsize=10)
    left.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle="--")
    left.text(0.02, 0.02, f"{hidden} settlements above {cap:.1f} not shown",
              transform=left.transAxes, fontsize=7, color=INK_MUTED,
              va="bottom")
    for side in ("top", "right"):
        left.spines[side].set_visible(False)

    # The mechanism panel: each service's mean PCEN as a ratio to the
    # Planned mean, so 1.0 means "as much as a planned colony" and the
    # tiny per-capita PCEN values stay readable.
    labels = [label for _, label in SERVICE_COLUMNS]
    y = np.arange(len(SERVICE_COLUMNS))
    planned_means = {column: frame.loc[frame["category"] == "Planned",
                                       column].mean()
                     for column, _ in SERVICE_COLUMNS}
    for offset, label, color in ((-0.16, "Planned", TYPE_COLORS["Planned"]),
                                 (0.16, "JJC", TYPE_COLORS["JJC"])):
        ratios = [frame.loc[frame["category"] == label, column].mean()
                  / planned_means[column] for column, _ in SERVICE_COLUMNS]
        right.scatter(ratios, y + offset, s=42, color=color, label=label,
                      edgecolor=SURFACE, linewidth=0.6, zorder=3)
    right.axvline(1.0, color=INK_MUTED, linewidth=1, linestyle="--")
    right.set_yticks(y, labels)
    right.invert_yaxis()
    right.set_xlabel("mean PCEN relative to Planned (1.0 = as much)")
    right.set_title("The mechanism — service availability", loc="left",
                    fontsize=10)
    right.legend(frameon=False, fontsize=8, loc="lower right")
    right.grid(axis="x", color=BASELINE, linewidth=0.5, alpha=0.6)
    right.set_axisbelow(True)
    for side in ("top", "right"):
        right.spines[side].set_visible(False)

    return _save(fig, out_dir, "fig6_jjc_vs_planned.png")


# --- manifest and CLI ----------------------------------------------------
def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
            capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_manifest(run, settlements, out_dir):
    popdensity = map_frame(settlements, run[HEADLINE_DENOM])
    lines = [
        f"profile: {PROFILE}",
        f"reported: {len(run[HEADLINE_DENOM])}",
        f"unscored: {int((~popdensity['scored']).sum())}",
        f"settlements: {len(settlements)}",
        f"commit: {_git_commit()}",
        "",
    ]
    for denom in DENOMINATORS:
        means = type_means(run[denom])
        lines.append(f"[{denom}]")
        for category in means.index:
            row = means.loc[category]
            lines.append(
                f"  {category:9s} mean_rel={row['mean_rel']:.6g} "
                f"mean_norm={row['mean_norm']:.6g} n={int(row['n'])}")
        lo, hi = ramp_bounds(run[denom]["rel_psi"])
        lines.append(f"  ramp_clip=({lo:.6g}, {hi:.6g})")
        lines.append("")
    path = Path(out_dir) / "manifest.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def load_settlements(data_dir):
    from delhi_psi.config import load_config

    data_dir = Path(data_dir).expanduser()
    cfg = load_config(PROFILE, data_dir=str(data_dir))
    layer = cfg.layers.settlements
    path = data_dir / layer.path
    if not path.exists():
        raise FigureError(f"settlement layer not found: {path}")
    gdf = gpd.read_file(path)
    gdf = gdf.set_crs(epsg=cfg.crs.epsg, allow_override=True)
    if layer.id_col != ID_COL:
        gdf = gdf.rename(columns={layer.id_col: ID_COL})
    return gdf


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render the DEL-33 paper figures from a DEL-32 run.")
    parser.add_argument("--run-dir", required=True,
                        help="DEL-32 output directory with the two CSVs")
    parser.add_argument("--data-dir", default="~/delhi_data",
                        help="input data root (settlement layer)")
    parser.add_argument("--out-dir", default="docs/figures/ratified-2026",
                        help="where the PNGs and manifest land")
    args = parser.parse_args(argv)

    try:
        run = load_run(args.run_dir)
        settlements = load_settlements(args.data_dir)
        out_dir = Path(args.out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        render_psi_by_type(run, out_dir, scale="rel")
        render_psi_map(run, settlements, out_dir, scale="rel")
        render_jjc_vs_planned(run, out_dir)
        render_psi_by_type(run, out_dir, scale="norm")
        render_psi_map(run, settlements, out_dir, scale="norm")
        write_manifest(run, settlements, out_dir)
    except FigureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {len(EXPECTED_FIGURES)} figures + manifest to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
