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

# The six point services plus the one line service (road). "clinic_count"
# etc. are the point-count amount columns; "road_length" is the line-length
# amount column. Mirrors `scripts/generate_production_fixtures.py`'s
# `POINT_SERVICES + ("road",)`.
SERVICES = ("clinic", "school", "bank", "police", "ration", "transport",
            "road")

AMOUNT_COLUMNS = {
    "clinic": "clinic_count",
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

# flags() non-triggering defaults (spec § 6.6). A row that omits a key is
# read as "this condition cannot fire" rather than KeyError — callers pass
# only the fields relevant to the flags they care about.
_FLAG_DEFAULTS = {
    "n_isolates": 0,
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


def flags(row):
    """Degenerate-run flags (spec § 6.6), in documented order. `row` is a
    mapping; a missing key reads as the value that cannot trigger its flag
    (see `_FLAG_DEFAULTS`), so a caller need only pass the fields relevant
    to the flags it wants checked."""
    def get(key):
        return row.get(key, _FLAG_DEFAULTS[key])

    result = []
    if get("n_isolates") > 0:
        result.append("isolates")
    if get("own_share_p50") < 0.10:
        result.append("smoothed")
    if get("n_at_psi1") == 1 and get("p99_psi") < 0.5:
        result.append("pinned")
    if get("rho_vs_own") < 0.5 and get("n_fragile_pairs") >= 3:
        result.append("reshuffled")
    return tuple(result)
