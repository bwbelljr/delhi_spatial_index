"""The settlement-category mapping layer (spec 3B §§ 3, 5).

Frame in, frame out: no config, no pipeline, no geo fixtures. That the whole
module needs nothing but a mapping is the point — a different city is a
different mapping, not different code.
"""
import pandas as pd
import pytest

from delhi_psi.categories import CATEGORY_COLUMN, apply_mapping, categories_of
from delhi_psi.validate import ValidationError

# The identity over Delhi's 10 USO_FINAL source types (spec 3B § 1).
USO_10 = {"Planned": "Planned", "UAC": "UAC", "JJC": "JJC", "RUAC": "RUAC",
          "RV": "RV", "UV": "UV", "SDA": "SDA", "JJR": "JJR",
          "Industrial": "Industrial", "Other": "Other"}

# The Phase 4 working candidate: five urban categories plus the `non-urban`
# bucket a run then excludes. UV/SDA/Other are Raj's call (DEL-29) and are
# deliberately absent here — under this mapping a layer carrying them errors.
URBAN_5 = {"Planned": "planned", "UAC": "unauthorized",
           "RUAC": "regularized", "JJR": "resettlement", "JJC": "jjc",
           "RV": "non-urban", "Industrial": "non-urban"}


def frame(types):
    return pd.DataFrame({"USO_AREA_U": [f"s{i}" for i in range(len(types))],
                         "USO_FINAL": list(types)})


def test_identity_mapping_copies_the_type_into_the_category_column():
    got = apply_mapping(frame(["Planned", "JJC", "RV"]),
                        type_col="USO_FINAL", mapping=USO_10)
    assert list(got[CATEGORY_COLUMN]) == ["Planned", "JJC", "RV"]
    assert list(got["USO_FINAL"]) == ["Planned", "JJC", "RV"], \
        "the raw source column is kept as-is alongside the category"


def test_many_to_one_collapse_is_the_point():
    got = apply_mapping(frame(["RV", "Industrial", "UAC", "Planned"]),
                        type_col="USO_FINAL", mapping=URBAN_5)
    assert list(got[CATEGORY_COLUMN]) == [
        "non-urban", "non-urban", "unauthorized", "planned"]


def test_a_mapping_broader_than_the_data_is_fine():
    """A scheme may be broader than one city (spec 3B § 2): an entry for a
    type absent from the data is not an error."""
    got = apply_mapping(frame(["Planned"]), type_col="USO_FINAL",
                        mapping=USO_10)
    assert list(got[CATEGORY_COLUMN]) == ["Planned"]


def test_unmapped_types_raise_naming_every_offender_with_counts():
    """One run diagnoses the WHOLE layer: every unmapped type with its row
    count, not just the first one hit."""
    with pytest.raises(ValidationError) as excinfo:
        apply_mapping(frame(["Planned", "UC", "IND", "UC"]),
                      type_col="USO_FINAL", mapping=USO_10)
    message = str(excinfo.value)
    assert "'IND' (1 row)" in message
    assert "'UC' (2 rows)" in message
    assert "categories.mapping" in message
    assert "Planned" in message, "the message lists what the mapping covers"


def test_a_custom_out_column_is_honoured():
    got = apply_mapping(frame(["Planned"]), type_col="USO_FINAL",
                        mapping=USO_10, out_col="cat")
    assert list(got["cat"]) == ["Planned"]
    assert CATEGORY_COLUMN not in got.columns


def test_the_input_frame_is_not_mutated():
    original = frame(["Planned", "JJC"])
    before = original.copy()
    apply_mapping(original, type_col="USO_FINAL", mapping=USO_10)
    pd.testing.assert_frame_equal(original, before)
    assert CATEGORY_COLUMN not in original.columns


def test_a_missing_type_column_is_named():
    with pytest.raises(ValidationError) as excinfo:
        apply_mapping(pd.DataFrame({"USO_AREA_U": ["s0"]}),
                      type_col="USO_FINAL", mapping=USO_10)
    assert "USO_FINAL" in str(excinfo.value)


def test_categories_of_is_the_set_of_category_names():
    assert categories_of(USO_10) == frozenset(USO_10)
    assert categories_of(URBAN_5) == frozenset(
        {"planned", "unauthorized", "regularized", "resettlement", "jjc",
         "non-urban"})
    assert len(URBAN_5) == 7 and len(categories_of(URBAN_5)) == 6, \
        "X:1 collapse: 7 source types produce 6 categories"
