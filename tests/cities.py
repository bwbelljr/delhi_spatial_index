"""The fixture cities: Oraculum (hand-ratifiable) and Messy (pathologies).

Fixture PLUMBING only — where a city's files live, what its vocabulary is,
which scenarios it is scored under. Deliberately importing nothing from this
repo: the reference implementation module imports this one, and its
INDEPENDENCE RULE forbids it from seeing the production library; importing
the reference implementation module from here would also be a cycle. The
index math lives there and in the production package, never here.

A `Scenario` carries ONE drop set and ONE flag, because that is exactly what
production does: `dropped = excluded_ids(types) ∪ missing`, with a single
`exclusion.stage` applied to all of it. `exclusion_types` and `stage` are the
production-side spelling of the same scenario; `tests/test_cities.py` pins
that the two spellings select the same rows.
"""

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd

FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures"
DEFAULT_EPSG = 7760


@dataclass(frozen=True)
class Scenario:
    """One row of a city's scenario table, in both spellings.

    name: the reference implementation's scenario name (also the value in
        `expected_values.csv`'s `scenario` column).
    dropped: settlement IDS the reference drops.
    dropped_before_neighbors: True iff the drop happens before neighbour
        construction.
    exclusion_types: CATEGORY names for production's
        `methodology.exclusion.types`.
    stage: production's `methodology.exclusion.stage`.
    """

    name: str
    dropped: frozenset
    dropped_before_neighbors: bool
    exclusion_types: tuple
    stage: str


@dataclass(frozen=True)
class City:
    """One fixture city: its files, its vocabulary, its scenario table."""

    name: str
    fixtures: Path
    scheme: str
    vocabulary: tuple
    scenarios: tuple
    epsg: int = DEFAULT_EPSG

    def mapping(self):
        """The identity over this city's source types.

        The fixture cities are not Delhi, so the shipped profiles' `uso-10`
        mapping does not cover them; every test that runs production on a
        fixture city swaps in this identity (spec 3B § 2).
        """
        return {source: source for source in self.vocabulary}

    def _read(self, filename):
        gdf = gpd.read_file(self.fixtures / filename)
        return gdf.set_crs(epsg=self.epsg, allow_override=True)

    def load_settlements(self):
        return self._read("settlements.geojson")

    def load_barriers(self):
        """May be an EMPTY collection (the messy city has no barriers): an
        empty GeoJSON FeatureCollection reads back as a 0-row frame with a
        geometry column, which both implementations short-circuit on."""
        return self._read("barriers.geojson")

    def load_services(self):
        gdf = self._read("services.geojson")
        return {name: grp.reset_index(drop=True)
                for name, grp in gdf.groupby("service")}


ORACULUM = City(
    name="oraculum",
    fixtures=FIXTURES_ROOT / "oraculum",
    scheme="oracle-6",
    vocabulary=("Planned", "UC", "JJC", "RV", "RUAC", "IND"),
    # ORDER IS LOAD-BEARING: it is today's SCENARIOS order (reference impl.),
    # which fixes `expected_values.csv`'s row order — round-trip tested byte
    # for byte.
    scenarios=(
        Scenario(name="baseline", dropped=frozenset(),
                 dropped_before_neighbors=False, exclusion_types=(),
                 stage="post_neighbors"),
        Scenario(name="excl_contributing", dropped=frozenset({"RV", "IND"}),
                 dropped_before_neighbors=False,
                 exclusion_types=("RV", "IND"), stage="post_neighbors"),
        Scenario(name="excl_removed", dropped=frozenset({"RV", "IND"}),
                 dropped_before_neighbors=True,
                 exclusion_types=("RV", "IND"), stage="pre_neighbors"),
        Scenario(name="excl_ind_removed", dropped=frozenset({"IND"}),
                 dropped_before_neighbors=True, exclusion_types=("IND",),
                 stage="pre_neighbors"),
        Scenario(name="excl_rv_only", dropped=frozenset({"RV"}),
                 dropped_before_neighbors=False, exclusion_types=("RV",),
                 stage="post_neighbors"),
    ),
)

# Every messy scenario drops `U` with the scenario's own flag: production
# drops a no-population id unconditionally and applies its single `stage` to
# the whole drop set, so under `pre_neighbors` `U` leaves the neighbour lists
# too (spec § 3, rev 3). That is why the no-population pathology lives in its
# own settlement and not in the RV one.
MESSY = City(
    name="messy",
    fixtures=FIXTURES_ROOT / "messy",
    scheme="messy-2",
    vocabulary=("Planned", "RV"),
    scenarios=(
        Scenario(name="nopop_only", dropped=frozenset({"U"}),
                 dropped_before_neighbors=False, exclusion_types=(),
                 stage="post_neighbors"),
        Scenario(name="excl_rv_post", dropped=frozenset({"U", "N"}),
                 dropped_before_neighbors=False, exclusion_types=("RV",),
                 stage="post_neighbors"),
        Scenario(name="excl_rv_pre", dropped=frozenset({"U", "N"}),
                 dropped_before_neighbors=True, exclusion_types=("RV",),
                 stage="pre_neighbors"),
    ),
)

CITIES = (ORACULUM, MESSY)
