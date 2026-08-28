"""Validation as assertions (DEL-25): the layer battery and the post-compute
sanity checks. Every check returns a Report or None; `require_*` raises.

The notebook printed these; here they raise, so a bad layer or a negative
index stops the run instead of scrolling past.
"""

import logging
from dataclasses import dataclass

from shapely.geometry import box

log = logging.getLogger(__name__)

GEOM_TYPES = ("Point", "Line", "Polygon")


class ValidationError(RuntimeError):
    """A validation check failed; the pipeline stage must not continue."""


@dataclass(frozen=True)
class LayerReport:
    name: str
    geom_type: str
    n_rows: int
    has_duplicate_rows: bool
    invalid_geometries: tuple
    none_geometries: tuple
    all_geom_type: bool
    within_bounds: bool

    @property
    def ok(self):
        return (not self.has_duplicate_rows
                and not self.invalid_geometries
                and not self.none_geometries
                and self.all_geom_type
                and self.within_bounds)

    def failures(self):
        problems = []
        if self.has_duplicate_rows:
            problems.append("has_duplicate_rows")
        if self.invalid_geometries:
            problems.append(f"invalid_geometries={list(self.invalid_geometries)}")
        if self.none_geometries:
            problems.append(f"none_geometries={list(self.none_geometries)}")
        if not self.all_geom_type:
            problems.append(f"all_geom_type is False (expected {self.geom_type})")
        if not self.within_bounds:
            problems.append("within_bounds is False")
        return problems


def has_duplicate_rows(gdf):
    return bool(len(gdf[gdf.duplicated()]) > 0)


def invalid_geometries(gdf):
    return tuple(i for i, row in gdf.iterrows()
                 if row["geometry"] is not None and not row["geometry"].is_valid)


def none_geometries(gdf):
    return tuple(gdf[gdf["geometry"].isna()].index)


def geometries_are(gdf, geom_type):
    """True if every geometry is of geom_type.

    Verbatim `check_geometries`, minus its vestigial `geom_type` column: the
    original assigned `gdf['geom_type'] = type(gdf['geometry'])` and then read
    `gdf.geom_type`, which resolves to the GeoDataFrame PROPERTY, not the
    column — so the assignment never affected the result. Dropping it means
    this function no longer mutates its argument.
    """
    if geom_type not in GEOM_TYPES:
        raise ValueError(f"unknown geom_type {geom_type!r}; allowed values: "
                         f"{list(GEOM_TYPES)}")
    geom_type_list = gdf.geom_type.unique()
    geom_is_geom_type = [geom_type in geom for geom in geom_type_list]
    return False not in geom_is_geom_type


def within_bounds(gdf, bounds_gdf):
    """True if the layer's total extent sits inside the bounds polygon."""
    reprojected = gdf.to_crs(bounds_gdf.crs)
    extent = box(reprojected.total_bounds[0], reprojected.total_bounds[1],
                 reprojected.total_bounds[2], reprojected.total_bounds[3])
    return bool(bounds_gdf.contains(extent).iloc[0])


def check_layer(gdf, *, name, geom_type, bounds_gdf):
    """Run the whole battery; never raises."""
    assert "geometry" in gdf.columns, 'there is no "geometry" column'
    report = LayerReport(
        name=name,
        geom_type=geom_type,
        n_rows=len(gdf),
        has_duplicate_rows=has_duplicate_rows(gdf),
        invalid_geometries=invalid_geometries(gdf),
        none_geometries=none_geometries(gdf),
        all_geom_type=geometries_are(gdf, geom_type),
        within_bounds=within_bounds(gdf, bounds_gdf),
    )
    log.info("layer %s: %d rows, ok=%s", name, report.n_rows, report.ok)
    return report


def require_layer(gdf, *, name, geom_type, bounds_gdf):
    """check_layer, but a failure raises ValidationError."""
    report = check_layer(gdf, name=name, geom_type=geom_type,
                         bounds_gdf=bounds_gdf)
    if not report.ok:
        raise ValidationError(
            f"layer {name!r} failed validation: {'; '.join(report.failures())}")
    return report


def check_missing_population(missing_count, *, maximum):
    if missing_count > maximum:
        raise ValidationError(
            f"{missing_count} settlements have no population row, above the "
            f"configured maximum of {maximum} "
            "(validate.max_missing_population)")
    log.info("%d settlements missing population (max %d)", missing_count,
             maximum)


def check_no_negative(frame, *, suffixes=("_count", "_pcen", "_idx")):
    offenders = []
    for suffix in suffixes:
        for column in [c for c in frame.columns if str(c).endswith(suffix)]:
            n_negative = int((frame[column] < 0).sum())
            if n_negative:
                offenders.append(f"{column}: {n_negative} negative value(s)")
    if offenders:
        raise ValidationError("negative values in derived columns: "
                              + "; ".join(offenders))


def check_crs_match(frames):
    """Every frame must share one CRS (the reprojection target)."""
    seen = {name: gdf.crs for name, gdf in frames.items()}
    distinct = {str(crs) for crs in seen.values()}
    if len(distinct) > 1:
        detail = ", ".join(f"{name}={crs}" for name, crs in seen.items())
        raise ValidationError(f"CRS mismatch across layers: {detail}")


def check_crs_defined(frames):
    """Every frame must carry a CRS; a CRS-less layer would be silently
    reprojected from nothing (spec § 6, compute-stage CRS check)."""
    missing = [name for name, gdf in frames.items() if gdf.crs is None]
    if missing:
        raise ValidationError(f"layers without a CRS: {', '.join(missing)}")
