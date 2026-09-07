# Archive: original master branch (2020-2021)

This folder is a snapshot of the `master` branch tip (commit 071e25f,
"feat: Complete PSI with 2019/20 pop estimates", Aug 2021).

The current pipeline at the repository root supersedes this code:

- `spatial_index_utils.py` (root) is this version plus small fixes
  (deprecated `cascaded_union` import removed, `set_geometry` fix).
- `Colonies Dataset Pre-Processing (2025).ipynb` supersedes the
  pre-processing notebooks here.
- `Colonies Public Services Index Calculations Updated (no RV) 2025.ipynb`
  supersedes the PSI calculation notebooks here.

Kept for reference (analyses with no current driver notebook):

- `Services Index for Wards.ipynb` - ward-level index driver
- `Colonies PSI with Buffer.ipynb` - buffer-based PSI variant driver
- `PSI with Area Cutoff and Other Category Ignored.ipynb` - exclusions variant
- `Computing Neighbors by Settlement Type.ipynb` - neighbor cross-tab by
  settlement type (logic not in spatial_index_utils.py)
- `Transforms for Skewed Data.ipynb` - transform exploration (none adopted;
  final index uses min-max normalization)
- `barrier_clip.ipynb`, `Merge Colonies Datasets.ipynb` - one-off data prep

The full commit history of this code is reachable from `main` via the
history merge (see merge commit "Merge master history into main").

## Cell outputs were cleared before public release

Every notebook here has had its execution outputs removed (7 Sep 2026). The
code cells, markdown and cell order are untouched — 657 code cells, exactly
as they were.

The reason is not tidiness. These notebooks were saved with their outputs,
and those outputs contained samples of the real Delhi layers: colony names,
`USO_AREA_U` identifiers, `POLYGON Z` geometries in the projected CRS,
per-settlement population estimates and computed index columns. The
repository does not redistribute those layers (see the root `README.md`), and
a `head()` preview of a withheld dataset is still the withheld dataset. One
notebook also printed a local Windows path containing a username.

So the outputs went and the code stayed, which keeps what this archive is
for: a record of how the published numbers were originally produced. To see
the outputs, run the notebooks against the layers.
