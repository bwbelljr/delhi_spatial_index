"""Settlement source type -> analysis category (spec 3B § 3).

Pure: this module never imports `delhi_psi.config`. The pipeline hands it a
plain `{source type: category}` mapping, so running a different city, or a
different collapse of the same city, is a different mapping — not different
code.

An unmapped source type is an ERROR: never a warning, never a fallback
category. Silence is the failure mode this layer exists to prevent.
"""

from delhi_psi.validate import ValidationError

CATEGORY_COLUMN = "category"


def categories_of(mapping):
    """The set of category names `mapping` produces (its values).

    Used by the config loader (`methodology.exclusion.types` must be a
    subset of it), by the run-time repeat of that check in the pipeline
    prelude, and by tests.
    """
    return frozenset(mapping.values())


def apply_mapping(frame, *, type_col, mapping, out_col=CATEGORY_COLUMN):
    """Return a COPY of `frame` with `out_col` = mapping[frame[type_col]].

    The raw source column is kept as-is; the category is an added label.
    Raises ValidationError naming EVERY source type with no mapping entry,
    with its row count, so one run diagnoses the whole layer. A mapping entry
    for a type that is absent from the data is fine — a scheme may be
    broader than one city.
    """
    if type_col not in frame.columns:
        raise ValidationError(
            f"settlement frame has no type column {type_col!r}; columns: "
            f"{sorted(str(column) for column in frame.columns)}")
    # dropna=False: a null type must be reported as unmapped, not dropped
    # from the diagnosis and then silently mapped to NaN.
    counts = frame[type_col].value_counts(dropna=False)
    unmapped = sorted(((value, int(n)) for value, n in counts.items()
                       if value not in mapping),
                      key=lambda item: str(item[0]))
    if unmapped:
        detail = ", ".join(f"{value!r} ({n} row{'' if n == 1 else 's'})"
                           for value, n in unmapped)
        raise ValidationError(
            f"settlement types with no categories.mapping entry: {detail}; "
            f"the mapping covers: {sorted(mapping)}")
    out = frame.copy()
    out[out_col] = frame[type_col].map(dict(mapping))
    return out
