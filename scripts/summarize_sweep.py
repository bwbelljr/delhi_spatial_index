"""The Phase 6 sweep statistics core (DEL-55 spec § 6, task 5).

Rendering the fenced `docs/data/` blocks is Task 6's job
(`scripts/summarize_sweep.py`'s CLI and `main()`, added there); this module
holds only the arithmetic, so it can be unit-tested against hand-computed
answers with no real output file in sight.

THE GOVERNING PRINCIPLE (spec § 6.1): PSI levels are NOT comparable across
sweep points. Eq. 2's min-max runs per run, and widening the neighbourhood
inflates every settlement at once. So every function here is either a
diagnostic on a single run's own numbers, or a RANK-based / within-run
STANDARDISED comparison across runs. Nothing here computes a level
difference or a level ratio between two points — that is exactly what
§ 6.9 forbids, and it stays forbidden by never being implemented, not by
being implemented and unused.

NO SCIPY. Not a direct or transitive dependency of this project (see
`pyproject.toml` / `uv.lock`), so `spearman_rho` and `kendall_tau_b` are
implemented on pandas + numpy: Spearman as Pearson correlation of
average-tied ranks, Kendall as the textbook tau-b formula computed by
chunked broadcasting.
"""

import logging
import math

import numpy as np
import pandas as pd

from delhi_psi import index

log = logging.getLogger(__name__)

# The six point services plus the one line service (road). "health_count"
# etc. are the point-count amount columns; "road_length" is the line-length
# amount column. These are the REAL production layer names
# (`delhi_psi/profiles/code-2025.yaml`'s `layers.point`/`layers.line`), which
# is what every sweep output CSV actually carries. This deliberately does
# NOT mirror `scripts/generate_production_fixtures.py`'s
# `POINT_SERVICES = ("clinic", ...)` — that tuple names the FIXTURE cities'
# (Oraculum/messy) own layer, which happens to be called "clinic" there;
# the real Delhi layer is "health" (`pipeline.compute_frames`'s docstring
# says so explicitly: "the oracle fixture's `clinic` maps to config
# `health`"). A real sweep-point CSV has `health_count`/`health_pcen`/
# `health_idx` columns, not `clinic_*` — confirmed against
# `~/psi_sweep/delhi_psi_adj-touch_popdensity_2020.csv`'s header. Using
# "clinic" here silently produced a KeyError against every real CSV; this
# was caught by Task 6's end-to-end smoke run against the real partial
# sweep, not by any hand-built test frame (which never spelled the service
# name out).
SERVICES = ("health", "school", "bank", "police", "ration", "transport",
            "road")

AMOUNT_COLUMNS = {
    "health": "health_count",
    "school": "school_count",
    "bank": "bank_count",
    "police": "police_count",
    "ration": "ration_count",
    "transport": "transport_count",
    "road": "road_length",
}

# Every column a per-point output CSV read must pass through `usecols`
# (spec § 6, preamble): the id, the type, the composition columns, the seven
# amount/pcen/idx columns, and the two PSI columns. Never the geometry, the
# centroid, or either neighbour-list column — at `band-10km`'s 4.37 M links
# those are most of the file (spec § 4.1) and nothing below reads them.
OUTPUT_USECOLS = (
    "USO_AREA_U", "USO_FINAL", "category", "population", "area_km2",
    *(AMOUNT_COLUMNS[svc] for svc in SERVICES),
    *(f"{svc}_pcen" for svc in SERVICES),
    *(f"{svc}_idx" for svc in SERVICES),
    "unnorm_psi", "norm_psi",
)

# flags() non-triggering defaults (spec § 6.6) for the flags that read a
# single row in isolation. A row that omits a key is read as "this condition
# cannot fire" rather than KeyError — callers pass only the fields relevant
# to the flags they care about. `n_isolates` is NOT here: the `isolates`
# flag is baseline-relative (see `flags`'s docstring) and has its own
# None-means-skip handling, not a default that could silently fire.
_FLAG_DEFAULTS = {
    "own_share_p50": 1.0,
    "n_at_psi1": 0,
    "p99_psi": 1.0,
    "rho_vs_own": 1.0,
    "n_fragile_pairs": 0,
}


def denominator_values(frame, denominator):
    """Eq. 3's denominator, exactly as `delhi_psi.index.pcen` computes it
    (`delhi_psi/index.py:300-305`): `popdensity = population / area_km2`,
    `pop = population`, `one = 1` for every row."""
    if denominator == "popdensity":
        return frame["population"] / frame["area_km2"]
    if denominator == "pop":
        return frame["population"]
    if denominator == "one":
        return pd.Series(1.0, index=frame.index)
    raise ValueError(
        f"unknown denominator {denominator!r}; allowed values: "
        f"{list(index.DENOMINATORS)}")


def own_share(frame, denominator, *, columns=None):
    """Eq. 3's own/(own+neighbour) share, pooled over `columns` (spec § 6.2).

    `columns` is a sequence of (amount_col, pcen_col) pairs; it defaults to
    the seven amount columns (six point services plus `road_length`) paired
    with their own `*_pcen` columns. Pooling sums own_pcen and sums pcen
    ACROSS services, then divides once — not one share per service averaged
    afterwards, which would weight a settlement's rarest service equally
    with its commonest.

    Returns NaN where the pooled `pcen` is exactly zero: a settlement that
    owns nothing and receives nothing from a neighbour has no share to
    report (0/0), not a share of zero. (pandas' Series division already
    returns NaN for 0/0 without raising under `-W error` — verified; only a
    bare numpy array division warns.) Excluding these rows from
    `own_share_p50` and counting them as `n_own_share_undef` is the
    renderer's job (Task 6); this function only has to not lie about them.

    Guard: `own_share <= 1 + 1e-9` on every non-NaN row, because the
    neighbour term Eq. 3 adds can never be negative. This is a real check
    against a mis-shaped reconstruction, and it is worth being explicit
    about what it does NOT prove: the denominator cancels out of the ratio
    (own/denom) / (pcen), so a wrong denominator formula passes this guard
    undetected. What actually pins the denominator is
    `test_own_share_of_one_when_there_is_no_neighbour_term` — an empty
    neighbour term is own_share == 1 exactly, and that is only true if
    `own_pcen` was computed with the SAME denominator `pcen` was.
    """
    if columns is None:
        columns = [(AMOUNT_COLUMNS[svc], f"{svc}_pcen") for svc in SERVICES]

    denom = denominator_values(frame, denominator)
    own_sum = pd.Series(0.0, index=frame.index)
    pcen_sum = pd.Series(0.0, index=frame.index)
    for amount_col, pcen_col in columns:
        own_sum = own_sum + frame[amount_col] / denom
        pcen_sum = pcen_sum + frame[pcen_col]

    share = own_sum / pcen_sum  # pandas: 0/0 -> NaN, no -W error warning

    bad = share > 1 + 1e-9  # NaN compares False, so undefined rows pass
    if bad.any():
        worst = share[bad].idxmax()
        raise ValueError(
            f"own_share: row {worst!r} has own_share={share[worst]!r} > 1, "
            "which Eq. 3's non-negative neighbour term makes impossible — "
            "the denominator reconstruction is mis-shaped (own_pcen and "
            "pcen were not built from the same per-row denominator).")

    return share


def own_only_psi(frame, denominator, *, second_normalization=True):
    """The own-only anchor (spec § 6.2): every neighbour term set to zero,
    so each settlement is scored on its own services alone.

    own-only PCEN is `own_count / denom` for each service present in
    `frame` (detected from `AMOUNT_COLUMNS`, so a test frame need not carry
    all seven), min-max normalised per service (Eq. 2), averaged (Eq. 1),
    and — when `second_normalization` — min-max'd again into the returned
    Series. Reuses `delhi_psi.index.minmax` rather than re-deriving Eq. 2,
    so a constant service column raises the identical `ValueError` the real
    pipeline raises on the identical fault, with the identical explanation.
    """
    denom = denominator_values(frame, denominator)
    present = [svc for svc in SERVICES if AMOUNT_COLUMNS[svc] in frame.columns]
    if not present:
        raise ValueError(
            "own_only_psi: frame has none of the expected amount columns "
            f"({sorted(AMOUNT_COLUMNS.values())})")

    work = pd.DataFrame(index=frame.index)
    idx_columns = []
    for svc in present:
        pcen_col = f"{svc}_pcen"
        idx_col = f"{svc}_idx"
        work[pcen_col] = frame[AMOUNT_COLUMNS[svc]] / denom
        work = index.minmax(work, source_col=pcen_col, target_col=idx_col)
        idx_columns.append(idx_col)

    work["unnorm_psi"] = work[idx_columns].mean(axis=1)
    if not second_normalization:
        return work["unnorm_psi"]

    work = index.minmax(work, source_col="unnorm_psi", target_col="norm_psi")
    return work["norm_psi"]


def percentile_rank(series):
    """Each value's percentile, 0-100, average rank for ties.

    `(rank - 1) / (n - 1) * 100` on `Series.rank(method="average")` — the
    minimum lands on 0, the maximum on 100, and a tied group lands on the
    mean of the positions it spans.
    """
    n = len(series)
    if n < 2:
        raise ValueError(
            "percentile_rank is undefined for fewer than two observations")
    ranks = series.rank(method="average")
    return (ranks - 1) / (n - 1) * 100


def category_order(frame, psi_col, *, category_col="category"):
    """The categories in `frame`, sorted by mean percentile rank of
    `psi_col` within this run, descending (spec § 6.3's `cat_order`)."""
    pct = percentile_rank(frame[psi_col])
    means = pct.groupby(frame[category_col]).mean()
    return list(means.sort_values(ascending=False).index)


def kendall_tau_order(a, b):
    """Kendall tau between two orderings of the same category set.

    `a` and `b` are lists of category labels, most-to-least. If the sets
    differ (a category absent from one run), the comparison is restricted
    to the intersection and the drop is logged — never silently widened or
    narrowed without a record of it.
    """
    common = [c for c in a if c in set(b)]
    dropped = (set(a) | set(b)) - set(common)
    if dropped:
        log.info(
            "kendall_tau_order: comparing %d common categories, dropping "
            "%d absent from one ordering: %s",
            len(common), len(dropped), sorted(dropped))
    if len(common) < 2:
        raise ValueError(
            "kendall_tau_order needs at least two categories in common")
    xs = pd.Series([a.index(c) for c in common])
    ys = pd.Series([b.index(c) for c in common])
    return kendall_tau_b(xs, ys)


def cliffs_delta(x, y):
    """Cliff's delta: P(a random x outranks a random y) - P(the reverse).

    Computed via the Mann-Whitney U on the COMBINED, average-tied ranking
    of x and y (not an O(|x|*|y|) pairwise count): with `n_x`, `n_y` the
    group sizes and `U_x` the rank-sum statistic for x,
        delta = 2 * U_x / (n_x * n_y) - 1.
    This is the standard U-statistic identity
    `U_x = n_x * n_y * (delta + 1) / 2` (`(delta+1)/2` is the probability of
    superiority), and average-tied ranking gives a tied pair exactly half
    credit in each direction, matching `cliffs_delta`'s own definition of a
    tie.
    """
    x = pd.Series(x, dtype=float).reset_index(drop=True)
    y = pd.Series(y, dtype=float).reset_index(drop=True)
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        raise ValueError("cliffs_delta needs at least one value on each side")
    combined = pd.concat([x, y], ignore_index=True)
    ranks = combined.rank(method="average")
    rank_x_sum = ranks.iloc[:n_x].sum()
    u_x = rank_x_sum - n_x * (n_x + 1) / 2
    return 2 * u_x / (n_x * n_y) - 1


def cohens_d(x, y):
    """Cohen's d with the pooled sample standard deviation (ddof=1)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_x, n_y = len(x), len(y)
    if n_x < 2 or n_y < 2:
        raise ValueError("cohens_d needs at least two values on each side")
    var_x = x.var(ddof=1)
    var_y = y.var(ddof=1)
    pooled = ((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2)
    if pooled == 0:
        raise ValueError(
            "cohens_d is undefined: both groups have zero variance")
    return (x.mean() - y.mean()) / math.sqrt(pooled)


def spearman_rho(a, b):
    """Spearman rho with no scipy: Pearson correlation of average-tied
    ranks (`Series.rank()` then `numpy.corrcoef`), per spec § 6's "no
    scipy" rule."""
    ra = pd.Series(a).rank(method="average")
    rb = pd.Series(b).rank(method="average")
    return float(np.corrcoef(ra, rb)[0, 1])


def kendall_tau_b(a, b):
    """Kendall's tau-b — NOT tau-a — computed with no scipy.

    tau_b = (C - D) / sqrt((n0 - n1) * (n0 - n2)), where n0 = n(n-1)/2,
    n1 counts pairs tied in `a` (whether or not also tied in `b`), and n2
    the mirror for `b`. Computed by broadcasting `a`'s and `b'`s pairwise
    differences in chunks of ~1,000 rows (a 4,131-row frame is a 17M-pair
    comparison), reducing each chunk's sign product to `int8` before
    counting so the intermediate stays cheap.

    tau-a (`(C - D) / n0`, ignoring ties in the denominator) silently
    passes every tie-free test and only diverges from tau-b when there are
    ties — see `test_kendall_tau_b_differs_from_tau_a_when_there_are_ties`.
    """
    a = np.asarray(pd.Series(a), dtype=float)
    b = np.asarray(pd.Series(b), dtype=float)
    n = len(a)
    if n != len(b):
        raise ValueError("kendall_tau_b: a and b must be the same length")
    if n < 2:
        raise ValueError("kendall_tau_b needs at least two observations")

    concordant = 0
    discordant = 0
    tied_x = 0
    tied_y = 0
    chunk = 1000
    cols = np.arange(n)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        rows = np.arange(start, end)
        da = a[rows][:, None] - a[None, :]
        db = b[rows][:, None] - b[None, :]
        sign_a = np.sign(da).astype(np.int8)
        sign_b = np.sign(db).astype(np.int8)
        prod = sign_a * sign_b
        self_mask = rows[:, None] == cols[None, :]
        concordant += int(np.count_nonzero((prod == 1) & ~self_mask))
        discordant += int(np.count_nonzero((prod == -1) & ~self_mask))
        tied_x += int(np.count_nonzero((sign_a == 0) & ~self_mask))
        tied_y += int(np.count_nonzero((sign_b == 0) & ~self_mask))

    # Every unordered pair {i, j} was counted twice (once as (i, j), once
    # as (j, i)), symmetrically.
    concordant //= 2
    discordant //= 2
    tied_x //= 2
    tied_y //= 2

    n0 = n * (n - 1) // 2
    denom = math.sqrt((n0 - tied_x) * (n0 - tied_y))
    if denom == 0:
        raise ValueError(
            "kendall_tau_b is undefined: every pair is tied in a or in b")
    return (concordant - discordant) / denom


def decile_set(series, *, top, fraction=0.10):
    """The tie-inclusive decile set of `series` (spec § 6.3): every row
    whose value ties the k-th order statistic, where
    `k = max(1, int(fraction * n))`.

    Returns `(members, gated)`. `members` is a set of `series`' index
    labels — a property of the VALUES, never of pandas' sort order: PSI has
    a mass point at exactly zero larger than a decile (452 of 4,131 rows at
    the baseline, 1,834 under the own-only anchor, against a decile of
    413), and `nsmallest(k)`/`nlargest(k)` alone would pick k of those rows
    arbitrarily. Taking the k-th order statistic's VALUE via
    `nsmallest(k).max()` / `nlargest(k).min()`, then including every row
    tied with it, sidesteps that: the boundary VALUE is well-defined
    regardless of which rows `nsmallest`/`nlargest` happened to return.

    `gated` is True when `members` exceeds `1.5 * k` — a "decile" that
    large is dominated by one tie block, not a decile, and every statistic
    built on it (`decile_jaccard`, § 6.5's `*_decile_share_*`) should render
    `-` rather than a number that measures where the tie block happened to
    fall.
    """
    s = pd.Series(series).dropna()
    n = len(s)
    if n == 0:
        return set(), False
    k = max(1, int(fraction * n))
    if top:
        threshold = s.nlargest(k).min()
        mask = s >= threshold
    else:
        threshold = s.nsmallest(k).max()
        mask = s <= threshold
    members = set(s.index[mask])
    gated = len(members) > 1.5 * k
    return members, gated


def decile_is_gated(series, *, top, fraction=0.10):
    """Whether `decile_set(series, ...)` would be gated, without needing
    the caller to unpack the set it does not want."""
    return decile_set(series, top=top, fraction=fraction)[1]


def decile_jaccard(a, b, *, top=True):
    """Jaccard overlap of `a` and `b`'s decile SETS (top or bottom).

    `None` when either side is gated (spec § 6.3) — the renderer turns
    that into `-` rather than printing a number that measures which rows
    happened to fall inside an oversized tie block.
    """
    set_a, gated_a = decile_set(a, top=top)
    set_b, gated_b = decile_set(b, top=top)
    if gated_a or gated_b:
        return None
    union = set_a | set_b
    if not union:
        return None
    return len(set_a & set_b) / len(union)


def _tied_random_rank(values, rng):
    """Ascending rank (1..n_cols) of each row of a 2D array, ties broken by
    an INDEPENDENT RANDOM permutation of column order per row, not by
    column position.

    `values` is shaped so that a fixed meaning attaches to each column
    across every row (e.g. one column per resampled settlement position, or
    one column per category) — so a plain `numpy.argsort(..., kind="stable")`
    would resolve every tie the same way, every row, because a stable sort
    keeps tied elements in their original column order. When the columns
    are concatenated in a fixed order upstream (as `bootstrap_rank_intervals`
    does, in point-estimate order), that turns "which side wins a tie" into
    a deterministic function of column position rather than a per-draw coin
    flip — see `test_the_bootstrap_breaks_ties_randomly_not_by_category_order`,
    which reproduces the resulting ~100%/0% split on a fully-tied
    two-category frame under a plain stable argsort.

    The fix: permute each row's column order with `rng` BEFORE the stable
    sort, then map the sorted positions back through that same permutation
    to recover the original column indices. Non-tied values still sort
    correctly (the permutation only changes which of several EQUAL values a
    stable sort sees first); tied values now land in a genuinely random
    order, independently per row.
    """
    n_rows, n_cols = values.shape
    col_order = np.tile(np.arange(n_cols), (n_rows, 1))
    shuffled_cols = rng.permuted(col_order, axis=1)
    shuffled_values = np.take_along_axis(values, shuffled_cols, axis=1)
    sorted_pos = np.argsort(shuffled_values, axis=1, kind="stable")
    orig_col_by_ascending_rank = np.take_along_axis(shuffled_cols, sorted_pos,
                                                      axis=1)
    ranks = np.empty((n_rows, n_cols), dtype=int)
    row_idx = np.arange(n_rows)[:, None]
    ranks[row_idx, orig_col_by_ascending_rank] = np.arange(n_cols)[None, :] + 1
    return ranks


def bootstrap_rank_intervals(frame, *, psi_col="norm_psi",
                             category_col="category", seed=0, n=1000):
    """A 95% bootstrap rank interval per category (spec § 6.4).

    1,000 (by default) resamples, EACH stratified by category — every
    category resampled with replacement to its OWN size, so every draw is
    the same total size as `frame` — vectorised across draws via
    `numpy.random.default_rng(seed).integers`. Within a draw, rows are
    ranked by `psi_col` using a plain (not tie-averaged) ordinal rank via
    `_tied_random_rank`, which breaks ties with a genuinely random per-draw
    permutation (see that function's docstring for why a plain stable
    argsort is NOT an acceptable substitute here): the point estimate
    (`category_order`, which DOES average-tie) only needs computing once,
    and averaging ties inside every one of 1,000 resamples would smooth
    over exactly the instability a tie mass point should produce.
    Composition-driven rank flips ARE the signal a wide interval is
    reporting (spec § 6.4: "honesty about ties is the interval itself"), so
    a random per-draw tie-break is what makes that signal honest.

    Returns a dict:
      - "categories": the category labels, in the point-estimate order.
      - "order": alias of "categories" (spec's `cat_order` shape).
      - "point_rank": {category: 1-indexed rank in the real (unresampled)
        ordering}.
      - "ci": {category: (low, high)} — the 2.5th/97.5th percentile of that
        category's rank across the `n` draws, as (int, int).
      - "draws": ndarray, shape (n, len(categories)) — every draw's rank
        per category, column-aligned with "categories". Task 6 uses this to
        derive `n_fragile_pairs` (adjacent-category flip probability).
      - "seed", "n": as given, for the caption.
    """
    categories = category_order(frame, psi_col, category_col=category_col)
    n_cat = len(categories)
    rng = np.random.default_rng(seed)

    psi = frame[psi_col].to_numpy()
    labels = frame[category_col].to_numpy()
    pools = {c: np.flatnonzero(labels == c) for c in categories}
    sizes = {c: len(pools[c]) for c in categories}
    total = sum(sizes.values())
    if total == 0:
        raise ValueError("bootstrap_rank_intervals: frame is empty")

    blocks = []
    boundaries = []
    start = 0
    for c in categories:
        size_c = sizes[c]
        boundaries.append((start, start + size_c))
        if size_c == 0:
            continue
        picks = rng.integers(0, size_c, size=(n, size_c))
        blocks.append(pools[c][picks])
        start += size_c
    draw_positions = np.concatenate(blocks, axis=1)
    draw_psi = psi[draw_positions]

    ranks = _tied_random_rank(draw_psi, rng)

    mean_ranks = np.full((n, n_cat), np.nan)
    for j, (lo, hi) in enumerate(boundaries):
        if hi > lo:
            mean_ranks[:, j] = ranks[:, lo:hi].mean(axis=1)

    # Rank the categories themselves by mean rank, descending (highest
    # mean rank = most PSI = category rank 1). An exact tie here is rare
    # once row-level ties are resolved randomly (it needs entire
    # categories' mean ranks to coincide), but is broken the same random
    # way for the same reason: `_tied_random_rank` on the negated
    # mean_ranks gives an ascending rank of `-mean_ranks`, i.e. a
    # descending rank of `mean_ranks`.
    draws = _tied_random_rank(-mean_ranks, rng)

    point_rank = {c: i + 1 for i, c in enumerate(categories)}
    ci = {}
    for j, c in enumerate(categories):
        lo_pct, hi_pct = np.percentile(draws[:, j], [2.5, 97.5])
        ci[c] = (int(round(lo_pct)), int(round(hi_pct)))

    return {
        "categories": categories,
        "order": categories,
        "point_rank": point_rank,
        "ci": ci,
        "draws": draws,
        "seed": seed,
        "n": n,
    }


def bootstrap_p_greater(frame, a, b, *, psi_col="norm_psi",
                        category_col="category", seed=0, n=1000):
    """Bootstrap probability that category `a`'s mean `psi_col` exceeds
    category `b`'s (spec § 6.5's `p_planned_gt_jjc`).

    `n` resamples of each group's own values (with replacement, to its own
    size), vectorised: the fraction of draws where a's resampled mean beats
    b's.
    """
    rng = np.random.default_rng(seed)
    psi_a = frame.loc[frame[category_col] == a, psi_col].to_numpy()
    psi_b = frame.loc[frame[category_col] == b, psi_col].to_numpy()
    n_a, n_b = len(psi_a), len(psi_b)
    if n_a == 0 or n_b == 0:
        raise ValueError(
            f"bootstrap_p_greater: category {a!r} or {b!r} has no rows")
    draws_a = psi_a[rng.integers(0, n_a, size=(n, n_a))].mean(axis=1)
    draws_b = psi_b[rng.integers(0, n_b, size=(n, n_b))].mean(axis=1)
    return float(np.mean(draws_a > draws_b))


def bootstrap_cliffs_delta_ci(frame, a, b, *, psi_col="norm_psi",
                              category_col="category", seed=0, n=1000):
    """95% bootstrap CI of `cliffs_delta` between categories `a` and `b`
    (spec § 6.5's primary gap statistic needs a CI; not in Task 5's
    interface list, added here in Task 6 alongside the renderer that
    actually needs it — same resampling shape as `bootstrap_p_greater`:
    each group resampled with replacement to its own size, `n` draws,
    seeded).

    Unlike `bootstrap_p_greater` (a single vectorised mean comparison),
    `cliffs_delta` itself needs a combined rank per draw, so this loops
    over `n` draws rather than vectorising — at n=1000 draws of a few
    thousand rows each this is milliseconds, not worth the complexity of a
    vectorised combined-rank-per-draw implementation.
    """
    rng = np.random.default_rng(seed)
    psi_a = frame.loc[frame[category_col] == a, psi_col].to_numpy()
    psi_b = frame.loc[frame[category_col] == b, psi_col].to_numpy()
    n_a, n_b = len(psi_a), len(psi_b)
    if n_a == 0 or n_b == 0:
        raise ValueError(
            f"bootstrap_cliffs_delta_ci: category {a!r} or {b!r} has no rows")
    deltas = np.empty(n)
    for i in range(n):
        draw_a = psi_a[rng.integers(0, n_a, n_a)]
        draw_b = psi_b[rng.integers(0, n_b, n_b)]
        deltas[i] = cliffs_delta(draw_a, draw_b)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


def decile_share(series, categories, label, *, top, fraction=0.10):
    """The fraction of the tie-gated top/bottom decile SET (spec § 6.3/6.5)
    that belongs to `label`. `series` and `categories` must share an index.
    `None` when the decile is gated (same 1.5x k rule as `decile_jaccard`)
    or empty — the renderer turns that into `—`."""
    members, gated = decile_set(series, top=top, fraction=fraction)
    if gated or not members:
        return None
    idx = pd.Index(sorted(members, key=str))
    return float((categories.reindex(idx) == label).mean())


def flags(row, *, baseline_isolates=None):
    """Degenerate-run flags (spec § 6.6), in documented order. `row` is a
    mapping; a missing key for one of the single-row flags reads as the
    value that cannot trigger it (see `_FLAG_DEFAULTS`), so a caller need
    only pass the fields relevant to the flags it wants checked.

    `isolates` is baseline-RELATIVE, not `n_isolates > 0` (corrected
    2026-09-06, spec commit 67e3f24, against the measured real-data run):
    the bbox baseline itself has 360 isolated settlements out of 4,131
    reported — an artifact of `code-2025`'s `global_asymmetric` barrier
    rule, which severs every link INTO a flagged settlement, and has
    nothing to do with any swept factor. `n_isolates > 0` would therefore
    fire on every point including both anchors, which makes it a constant
    rather than a flag. The two adjacency points measured so far genuinely
    have MORE (band-0km 697, adj-touch 715 — a narrower neighbourhood
    strands more settlements), and that comparison to the baseline is the
    real signal this flag is for: fires on `row["n_isolates"] >
    baseline_isolates`, strictly.

    `baseline_isolates=None` means the flag is not EVALUATED at all — not
    fired as False — because the comparison has no basis: the own-only
    anchor has no neighbourhood to begin with, and a point whose own
    artifact was unreadable has `n_isolates: None`
    (`run_sweep.py` records `degree_from: "artifact unreadable: ..."` in
    that case). It is the caller's job, per point, to decide whether a
    baseline comparison is meaningful and pass `None` when it is not; this
    function never raises over it, whether it is `baseline_isolates` or the
    row's own `n_isolates` that is missing or None.
    """
    def get(key):
        return row.get(key, _FLAG_DEFAULTS[key])

    result = []
    if baseline_isolates is not None:
        n_isolates = row.get("n_isolates")
        if n_isolates is not None and n_isolates > baseline_isolates:
            result.append("isolates")
    if get("own_share_p50") < 0.10:
        result.append("smoothed")
    if get("n_at_psi1") == 1 and get("p99_psi") < 0.5:
        result.append("pinned")
    if get("rho_vs_own") < 0.5 and get("n_fragile_pairs") >= 3:
        result.append("reshuffled")
    return tuple(result)


# =========================================================================
# Task 6: rendering the `docs/data/phase6_sweep.md` blocks, and `main()`.
#
# `docs/data/phase6_sweep.md` itself (prose, provenance, captions) is a
# hand-authored document (spec § 6, Task 6 step 4) that carries this
# script's stdout verbatim inside its fenced blocks — exactly the shape
# `docs/data/rule_effects.md` and `scripts/measure_rule_effects.py` already
# use. `main()` below only ever prints/writes the fenced blocks; it does
# not touch the document's prose.
# =========================================================================

import argparse
import json
from pathlib import Path

from delhi_psi import io as psi_io
from delhi_psi.config import load_config
from scripts._measure_common import FENCE, parse_block, render  # noqa: F401
from scripts.run_sweep import degree_report

BLOCKS = ("points", "ordering", "gap", "denominator_check")

DASH = "—"  # em dash. Every gated cell (`decile_jaccard`/`decile_share`
                # returning `None`) and every statistic a point genuinely
                # lacks (the baseline's timings, the own-only anchor's whole
                # neighbourhood) renders this — never a number, never a
                # blank cell, per this task's resolution #2.

ID_COL = "USO_AREA_U"
PSI_COL = "norm_psi"

FORMAL = ("Planned", "SDA")
INFORMAL = ("JJC", "JJR")

# Spec § 3's eleven real sweep points, cheapest/most-legible first — the two
# anchors (`baseline`, `own-only`) are handled separately, since neither has
# a manifest (§ 6.3). A point renders only if its manifest actually exists
# under `--work-dir`: the real run is still in flight (8 of 11 present as of
# this writing, spec § 7), and discovering what IS there rather than
# hard-failing on what is not is what lets this same code produce today's
# dry-run-scale document and, completely unmodified, Task 7's full one.
SWEEP_POINT_ORDER = (
    "adj-touch", "band-0km", "band-1km", "band-5km", "band-10km",
    "decay-none", "decay-power05", "decay-power2", "decay-exp2km",
    "decay-exp5km", "decay-boundary",
)


def _fmt(value, ndigits=None):
    """One block cell. `None` or NaN renders the em dash (resolution #2); a
    number with `ndigits` given renders fixed-point (`f"{x:.{n}f}"`), which
    is what keeps a value like `1.000` from losing its trailing zeros the
    way a bare `round()` would; anything else renders as `str()`."""
    if value is None:
        return DASH
    if isinstance(value, float) and math.isnan(value):
        return DASH
    if ndigits is not None:
        return f"{float(value):.{ndigits}f}"
    return str(value)


def _fmt_int(value):
    return DASH if value is None else str(int(value))


def _identity_fields(cfg):
    """`adjacency`, `radius_km`, `decay_form`, `decay_param`,
    `decay_distance` (spec § 6.3's identity columns), read straight off the
    loaded profile config. NOT read off the manifest's `stamp`: that only
    ever covers adjacency + barrier (spec § 4.2 — decay is applied
    downstream in `compute` and never enters the methodology stamp), and
    the baseline point (no manifest at all, § 6.3) needs the identical
    fields anyway, so one code path serves both."""
    m = cfg.methodology
    rule = str(m.adjacency.rule)
    radius_km = (_fmt(m.adjacency.max_distance_km, 1)
                if rule == "within_distance" else DASH)
    decay_form = str(m.decay.form)
    if decay_form == "inverse_power":
        decay_param = _fmt(m.decay.exponent, 2)
    elif decay_form == "exponential":
        decay_param = _fmt(m.decay.scale_km, 1)
    else:
        decay_param = DASH
    return {
        "adjacency": rule,
        "radius_km": radius_km,
        "decay_form": decay_form,
        "decay_param": decay_param,
        "decay_distance": str(m.decay.distance),
    }


# The own-only anchor has no neighbourhood at all (spec § 6.2): none of the
# identity columns describe anything real for it.
_OWN_ONLY_IDENTITY = {"adjacency": DASH, "radius_km": DASH,
                     "decay_form": DASH, "decay_param": DASH,
                     "decay_distance": DASH}


def _aligned(a, b):
    """Two Series reindexed onto their common index labels — the shape
    every cross-point comparison in § 6.3 needs, since a point's reported
    universe is not guaranteed to be identical, row for row, to another's
    (a settlement excluded post-neighbours under one adjacency rule need
    not be excluded under another)."""
    common = a.index.intersection(b.index)
    return a.loc[common], b.loc[common]


def load_output_frame(csv_path):
    """One sweep point's output CSV, restricted to `OUTPUT_USECOLS` (spec
    § 6 preamble — the neighbour-list and geometry columns are most of the
    file at `band-10km`'s link count and nothing here reads them) and
    indexed by settlement id."""
    frame = pd.read_csv(csv_path, usecols=list(OUTPUT_USECOLS))
    return frame.set_index(ID_COL)


def own_only_frame(bbox_frame):
    """The own-only anchor (spec § 6.2), derived arithmetically from the
    bbox baseline's own-count columns — no pipeline run, no profile.
    `population`, `area_km2`, `category` and the seven amount columns are
    properties of the settlement itself, not of the adjacency rule, so the
    bbox baseline's CSV is the one true source for them regardless of which
    other point is being summarised."""
    cols = ["population", "area_km2", "category",
            *(AMOUNT_COLUMNS[svc] for svc in SERVICES)]
    frame = bbox_frame[cols].copy()
    frame[PSI_COL] = own_only_psi(frame, "popdensity")
    return frame


def _n_fragile_pairs(bootstrap, *, threshold=0.05):
    """The number of ADJACENT pairs in `bootstrap`'s point-estimate
    `categories` whose bootstrap flip probability exceeds `threshold` (spec
    § 6.4). `draws` is column-aligned to `categories`: ascending rank,
    1 = best (highest mean PSI). A "flip" at position i is a draw where the
    category one place lower in the point estimate (`categories[i + 1]`)
    actually beats the one directly above it (`categories[i]`)."""
    categories = bootstrap["categories"]
    draws = bootstrap["draws"]
    count = 0
    for i in range(len(categories) - 1):
        flip = float(np.mean(draws[:, i + 1] < draws[:, i]))
        if flip > threshold:
            count += 1
    return count


def point_stats(frame, *, seed=0, n=1000):
    """Everything about ONE point's own frame that blocks `points` and
    `ordering` share, computed once: its category order, its bootstrap rank
    intervals, and the fragile-pair count derived from them. Kept separate
    from the render functions so a 1,000-draw bootstrap over a 4,131-row
    frame is never paid twice for the same point."""
    cat_order = category_order(frame, PSI_COL)
    bootstrap = bootstrap_rank_intervals(frame, psi_col=PSI_COL, seed=seed,
                                         n=n)
    return {"cat_order": cat_order, "bootstrap": bootstrap,
           "n_fragile_pairs": _n_fragile_pairs(bootstrap)}


# --- block `points` (spec § 6.3) -----------------------------------------
def render_points_row(point, *, profile, identity, structure, frame,
                      own_frame, bbox_frame, own_stats, bbox_stats,
                      baseline_isolates, status="OK"):
    """One row of block `points`: identity, structure/cost, composition,
    and the rank-based outcome columns, pre-formatted to the documented
    precision. `frame` is `None` for a point whose manifest is not `OK`
    with a readable output file — that row is reported as `FAILED` rather
    than omitted (spec § 5 item 4), carrying only its identity."""
    row = {"point": point, "profile": profile, **identity, **structure}
    if status != "OK" or frame is None:
        row["own_share_p50"] = DASH
        row["n_own_share_undef"] = DASH
        row["cat_order"] = DASH
        for key in ("tau_vs_own", "tau_vs_bbox", "rho_vs_own", "rho_vs_bbox",
                   "taub_vs_own", "taub_vs_bbox", "jaccard_top10",
                   "jaccard_bottom10", "planned_gt_jjc"):
            row[key] = DASH
        row["flag"] = "FAILED"
        return row

    this_stats = point_stats(frame)
    cat_order = this_stats["cat_order"]

    if point == "own-only":
        # own_share is IDENTICALLY 1 wherever pcen is nonzero (the
        # neighbour term is always zero here by construction), so
        # `own_share_p50` is exactly `1.000` regardless of how many rows
        # own nothing — but "own nothing" (0/0, undefined) is not itself
        # zero, and IS worth counting: it is the "1,834 of 4,131" figure
        # spec § 6.2 cites by name for this exact anchor.
        amount_cols = [AMOUNT_COLUMNS[svc] for svc in SERVICES]
        n_own_share_undef = int((frame[amount_cols].sum(axis=1) == 0).sum())
        own_share_p50 = 1.0
    else:
        share = own_share(frame, "popdensity")
        own_share_p50 = share.median(skipna=True)
        n_own_share_undef = int(share.isna().sum())
    row["own_share_p50"] = _fmt(own_share_p50, 3)
    row["n_own_share_undef"] = n_own_share_undef
    row["cat_order"] = ">".join(cat_order)

    this_psi, own_psi = frame[PSI_COL], own_frame[PSI_COL]
    bbox_psi = bbox_frame[PSI_COL]

    if point == "own-only":
        row["tau_vs_own"] = _fmt(1.0, 2)
        row["rho_vs_own"] = _fmt(1.0, 3)
        row["taub_vs_own"] = _fmt(1.0, 3)
        rho_vs_own_raw = 1.0
    else:
        row["tau_vs_own"] = _fmt(
            kendall_tau_order(cat_order, own_stats["cat_order"]), 2)
        a, b = _aligned(this_psi, own_psi)
        rho_vs_own_raw = spearman_rho(a, b)
        row["rho_vs_own"] = _fmt(rho_vs_own_raw, 3)
        row["taub_vs_own"] = _fmt(kendall_tau_b(a, b), 3)

    if point == "baseline":
        row["tau_vs_bbox"] = _fmt(1.0, 2)
        row["rho_vs_bbox"] = _fmt(1.0, 3)
        row["taub_vs_bbox"] = _fmt(1.0, 3)
        row["jaccard_top10"] = _fmt(1.0, 3)
        row["jaccard_bottom10"] = _fmt(1.0, 3)
    else:
        row["tau_vs_bbox"] = _fmt(
            kendall_tau_order(cat_order, bbox_stats["cat_order"]), 2)
        a, b = _aligned(this_psi, bbox_psi)
        row["rho_vs_bbox"] = _fmt(spearman_rho(a, b), 3)
        row["taub_vs_bbox"] = _fmt(kendall_tau_b(a, b), 3)
        row["jaccard_top10"] = _fmt(decile_jaccard(a, b, top=True), 3)
        row["jaccard_bottom10"] = _fmt(decile_jaccard(a, b, top=False), 3)

    row["planned_gt_jjc"] = (
        str(cat_order.index("Planned") < cat_order.index("JJC"))
        if "Planned" in cat_order and "JJC" in cat_order else DASH)

    flag_row = {
        "own_share_p50": (own_share_p50 if not math.isnan(own_share_p50)
                          else 1.0),
        "n_at_psi1": int((frame[PSI_COL] == 1).sum()),
        "p99_psi": float(frame[PSI_COL].quantile(0.99)),
        "rho_vs_own": rho_vs_own_raw,
        "n_fragile_pairs": this_stats["n_fragile_pairs"],
    }
    if structure.get("_n_isolates_raw") is not None:
        flag_row["n_isolates"] = structure["_n_isolates_raw"]
    computed = flags(flag_row, baseline_isolates=baseline_isolates)
    row.pop("_n_isolates_raw", None)
    row["flag"] = ",".join(computed) if computed else "-"
    return row


def _structure_fields(manifest, *, is_baseline=False, is_own_only=False,
                      baseline_artifact_report=None):
    """`n_reported`, `n_isolates`, `n_links`, `deg_mean`, `deg_p50`,
    `deg_max`, `preprocess_s`, `compute_s` (spec § 6.3), formatted. The
    baseline reads its structure off the proven joblib artifact (no
    manifest exists for it — it predates this runner); the own-only anchor
    has no neighbourhood at all and every structure column but `n_reported`
    is the em dash (this task's brief, step 3). `_n_isolates_raw` is carried
    alongside for `flags()`'s baseline-relative comparison and popped
    before the row is emitted."""
    if is_own_only:
        return {"n_reported": manifest["n_reported"], "n_isolates": DASH,
               "n_links": DASH, "deg_mean": DASH, "deg_p50": DASH,
               "deg_max": DASH, "preprocess_s": DASH, "compute_s": DASH,
               "_n_isolates_raw": None}
    if is_baseline:
        degree = baseline_artifact_report
        return {
            "n_reported": manifest["n_reported"],
            "n_isolates": _fmt_int(degree["n_isolates"]),
            "n_links": _fmt_int(degree["n_links"]),
            "deg_mean": _fmt(degree["deg_mean"], 1),
            "deg_p50": _fmt_int(degree["deg_p50"]),
            "deg_max": _fmt_int(degree["deg_max"]),
            "preprocess_s": DASH,
            "compute_s": DASH,
            "_n_isolates_raw": degree["n_isolates"],
        }
    return {
        "n_reported": manifest.get("n_reported"),
        "n_isolates": _fmt_int(manifest.get("n_isolates")),
        "n_links": _fmt_int(manifest.get("n_links")),
        "deg_mean": _fmt(manifest.get("deg_mean"), 1),
        "deg_p50": _fmt_int(manifest.get("deg_p50")),
        "deg_max": _fmt_int(manifest.get("deg_max")),
        "preprocess_s": _fmt(manifest.get("preprocess_s"), 3),
        "compute_s": _fmt(manifest.get("compute_s"), 3),
        "_n_isolates_raw": manifest.get("n_isolates"),
    }


def gather_points(work_dir, baseline_dir):
    """Every point block `points`/`ordering`/`gap` iterate over, in order:
    `("baseline", "own-only", *the profiles with a manifest present)`.
    Returns `(order, frames, stats, structures, manifests)`, all keyed by
    point name, so the three render functions below share one read of every
    CSV and one bootstrap per point."""
    work_dir = Path(work_dir).expanduser()
    baseline_dir = Path(baseline_dir).expanduser()

    bbox_csv = baseline_dir / "delhi_psi_code-2025_popdensity_2020.csv"
    bbox_frame = load_output_frame(bbox_csv)
    own_frame_ = own_only_frame(bbox_frame)

    baseline_artifact = psi_io.read_neighbors(
        baseline_dir / "colonies_neighbors.joblib")
    baseline_degree = degree_report(baseline_artifact, ID_COL)
    baseline_manifest = {"n_reported": len(bbox_frame)}

    frames = {"baseline": bbox_frame, "own-only": own_frame_}
    structures = {
        "baseline": _structure_fields(baseline_manifest, is_baseline=True,
                                      baseline_artifact_report=baseline_degree),
        "own-only": _structure_fields(
            {"n_reported": len(own_frame_)}, is_own_only=True),
    }
    manifests = {"baseline": {"status": "OK", "profile": "code-2025"},
                "own-only": {"status": "OK", "profile": None}}
    identities = {"baseline": _identity_fields(load_config("code-2025")),
                 "own-only": dict(_OWN_ONLY_IDENTITY)}

    order = ["baseline", "own-only"]
    for profile in SWEEP_POINT_ORDER:
        manifest_path = work_dir / "manifest" / f"{profile}.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        order.append(profile)
        manifests[profile] = manifest
        identities[profile] = _identity_fields(load_config(profile))
        structures[profile] = _structure_fields(manifest)
        outputs = manifest.get("outputs") or []
        if manifest.get("status") == "OK" and outputs:
            frames[profile] = load_output_frame(outputs[0])
        else:
            frames[profile] = None

    stats = {name: (point_stats(frame) if frame is not None else None)
            for name, frame in frames.items()}
    return order, frames, stats, structures, manifests, identities, \
        bbox_frame, own_frame_


def render_points_block(work_dir, baseline_dir):
    (order, frames, stats, structures, manifests, identities, bbox_frame,
     own_frame_) = gather_points(work_dir, baseline_dir)
    baseline_isolates = structures["baseline"]["_n_isolates_raw"]

    blocks = []
    for point in order:
        profile = manifests[point].get("profile") or point
        row = render_points_row(
            point, profile=profile, identity=identities[point],
            structure=structures[point], frame=frames[point],
            own_frame=own_frame_, bbox_frame=bbox_frame,
            own_stats=stats["own-only"], bbox_stats=stats["baseline"],
            baseline_isolates=baseline_isolates,
            status=manifests[point].get("status", "OK"))
        blocks.append(render(row, name="points"))
    return "\n".join(blocks)


# --- block `ordering` (spec § 6.4) ---------------------------------------
def render_ordering_block(work_dir, baseline_dir):
    (order, frames, stats, *_rest) = gather_points(work_dir, baseline_dir)
    blocks = []
    for point in order:
        if frames[point] is None:
            blocks.append(render({"point": point, "status": "FAILED"},
                                 name="ordering"))
            continue
        s = stats[point]
        bootstrap = s["bootstrap"]
        row = {"point": point}
        for cat in bootstrap["categories"]:
            lo, hi = bootstrap["ci"][cat]
            row[cat] = f"{bootstrap['point_rank'][cat]} [{lo}-{hi}]"
        row["n_fragile_pairs"] = s["n_fragile_pairs"]
        row["seed"] = bootstrap["seed"]
        row["n"] = bootstrap["n"]
        blocks.append(render(row, name="ordering"))
    return "\n".join(blocks)


# --- block `gap` (spec § 6.5) ---------------------------------------------
def _gap_row(point, frame, *, group, a_label, b_label, a_members, b_members):
    """One row of block `gap`, for either the per-category `Planned` vs
    `JJC` pair or the pooled `formal = {Planned, SDA}` vs
    `informal = {JJC, JJR}` grouping (spec § 6.5) — the same shape either
    way, since the pooled row is computed over a frame whose `category`
    column has been remapped to `"formal"`/`"informal"` first."""
    psi_a = frame.loc[frame["category"].isin(a_members), PSI_COL]
    psi_b = frame.loc[frame["category"].isin(b_members), PSI_COL]
    if len(psi_a) == 0 or len(psi_b) == 0:
        return {"point": point, "group": group, "a": a_label, "b": b_label,
               "cliffs_delta": DASH, "cliffs_delta_ci_lo": DASH,
               "cliffs_delta_ci_hi": DASH, "cohens_d": DASH,
               "pct_gap": DASH, "p_a_gt_b": DASH,
               "top_decile_share_a": DASH, "top_decile_share_b": DASH,
               "bottom_decile_share_b": DASH}

    pooled = frame.copy()
    pooled["_group"] = pooled["category"].where(
        pooled["category"].isin(a_members), other=None)
    pooled.loc[pooled["category"].isin(a_members), "_group"] = a_label
    pooled.loc[pooled["category"].isin(b_members), "_group"] = b_label

    delta = cliffs_delta(psi_a, psi_b)
    ci_lo, ci_hi = bootstrap_cliffs_delta_ci(pooled, a_label, b_label,
                                             psi_col=PSI_COL,
                                             category_col="_group")
    d = cohens_d(psi_a, psi_b)

    pct = percentile_rank(frame[PSI_COL])
    means = pct.groupby(frame["category"]).mean()
    pct_gap = float(means.loc[list(a_members)].mean()
                    - means.loc[list(b_members)].mean())

    p_gt = bootstrap_p_greater(pooled, a_label, b_label, psi_col=PSI_COL,
                               category_col="_group")

    top_share_a = decile_share(frame[PSI_COL], frame["category"], a_label,
                               top=True) if len(a_members) == 1 else \
        decile_share(pooled[PSI_COL], pooled["_group"], a_label, top=True)
    top_share_b = decile_share(pooled[PSI_COL], pooled["_group"], b_label,
                               top=True)
    bottom_share_b = decile_share(pooled[PSI_COL], pooled["_group"], b_label,
                                  top=False)

    return {
        "point": point, "group": group, "a": a_label, "b": b_label,
        "cliffs_delta": _fmt(delta, 2),
        "cliffs_delta_ci_lo": _fmt(ci_lo, 2),
        "cliffs_delta_ci_hi": _fmt(ci_hi, 2),
        "cohens_d": _fmt(d, 2),
        "pct_gap": _fmt(pct_gap, 1),
        "p_a_gt_b": _fmt(p_gt, 3),
        "top_decile_share_a": _fmt(top_share_a, 3),
        "top_decile_share_b": _fmt(top_share_b, 3),
        "bottom_decile_share_b": _fmt(bottom_share_b, 3),
    }


def _finalize_gap_rows(rows):
    """spec § 6.5: `p_a_gt_b` constant across every row carries no
    information ("If it comes out 1.000 at every point ... the doc states
    that in one sentence and drops the column"), so it is dropped and
    replaced by ONE generated note — GENERATED, not typed by hand into the
    document (fix round 1, item 4): a hand-edited drop breaks the
    guarantee that the document is exactly what this script prints, and
    silently defeats the drift test. If `p_a_gt_b` is missing from every
    row (nothing to check) or takes more than one value across the rows
    that carry it, that would itself be a real signal — spec § 6.5 asks
    for the column to be reported, not summarised away, so `rows` is
    returned completely unchanged in that case."""
    p_values = {r["p_a_gt_b"] for r in rows if "p_a_gt_b" in r}
    if len(p_values) != 1:
        return rows
    (value,) = p_values
    dropped = [{k: v for k, v in r.items() if k != "p_a_gt_b"} for r in rows]
    note = {"note": f"p_a_gt_b is constant at {value} across every row "
                    "above (spec § 6.5) and is dropped rather than "
                    "printed unchanged"}
    return dropped + [note]


def render_gap_block(work_dir, baseline_dir):
    (order, frames, *_rest) = gather_points(work_dir, baseline_dir)
    rows = []
    for point in order:
        frame = frames[point]
        if frame is None:
            rows.append({"point": point, "status": "FAILED"})
            continue
        if not {"Planned", "JJC"} <= set(frame["category"].unique()):
            continue  # own-only/anchors always carry both; a real point
                     # could in principle drop a whole category (n=0 flag
                     # in `own_only_psi`), and a gap needs both sides.
        rows.append(_gap_row(point, frame, group="Planned_vs_JJC",
                             a_label="Planned", b_label="JJC",
                             a_members={"Planned"}, b_members={"JJC"}))
        rows.append(_gap_row(point, frame, group="formal_vs_informal",
                             a_label="formal", b_label="informal",
                             a_members=set(FORMAL), b_members=set(INFORMAL)))
    rows = _finalize_gap_rows(rows)
    return "\n".join(render(row, name="gap") for row in rows)


# --- block `denominator_check` (spec § 4.3) -------------------------------
def render_denominator_check_block(baseline_dir):
    """The one-off check spec § 4.3 asks for: the baseline's category order
    and Planned-vs-JJC effect size under `pop` beside the same under
    `popdensity` — the only two points sharing the same denominator that
    was ever run. Only the baseline can answer this (spec § 4.3): every
    sweep point reports `popdensity` alone."""
    baseline_dir = Path(baseline_dir).expanduser()
    pop_frame = load_output_frame(
        baseline_dir / "delhi_psi_code-2025_pop_2020.csv")
    density_frame = load_output_frame(
        baseline_dir / "delhi_psi_code-2025_popdensity_2020.csv")

    pop_order = category_order(pop_frame, PSI_COL)
    density_order = category_order(density_frame, PSI_COL)
    tau = kendall_tau_order(pop_order, density_order)

    delta_pop = cliffs_delta(
        pop_frame.loc[pop_frame["category"] == "Planned", PSI_COL],
        pop_frame.loc[pop_frame["category"] == "JJC", PSI_COL])
    delta_density = cliffs_delta(
        density_frame.loc[density_frame["category"] == "Planned", PSI_COL],
        density_frame.loc[density_frame["category"] == "JJC", PSI_COL])

    row = {
        "point": "baseline",
        "cat_order_pop": ">".join(pop_order),
        "cat_order_popdensity": ">".join(density_order),
        "tau_pop_vs_popdensity": _fmt(tau, 2),
        "cliffs_delta_planned_jjc_pop": _fmt(delta_pop, 2),
        "cliffs_delta_planned_jjc_popdensity": _fmt(delta_density, 2),
        "agreement": "AGREE" if pop_order == density_order else "DISAGREE",
    }
    return render(row, name="denominator_check")


# --- CLI -------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        prog="summarize_sweep",
        description="Phase 6 sweep summariser: reads the sweep points' "
                    "manifests and output CSVs and the proven bbox "
                    "baseline, and prints the fenced docs/data/ blocks "
                    "(DEL-55, spec § 6). DRY RUN: code-2025 rules only.")
    parser.add_argument("--work-dir", default="~/psi_sweep",
                        help="where run_sweep.py wrote its manifests and "
                             "output CSVs (default: ~/psi_sweep)")
    parser.add_argument("--baseline-dir", default="~/delhi_data/phase3_verify",
                        help="the proven code-2025 run, read-only "
                             "(default: ~/delhi_data/phase3_verify)")
    parser.add_argument("--out", default=None,
                        help="write the blocks here instead of stdout")
    parser.add_argument("--block", choices=BLOCKS, default=None,
                        help="render only this block")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    work_dir = Path(args.work_dir).expanduser()
    baseline_dir = Path(args.baseline_dir).expanduser()
    wanted = BLOCKS if args.block is None else (args.block,)

    renderers = {
        "points": lambda: render_points_block(work_dir, baseline_dir),
        "ordering": lambda: render_ordering_block(work_dir, baseline_dir),
        "gap": lambda: render_gap_block(work_dir, baseline_dir),
        "denominator_check": lambda: render_denominator_check_block(baseline_dir),
    }
    text = "\n".join(renderers[name]() for name in wanted)

    if args.out:
        Path(args.out).write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
