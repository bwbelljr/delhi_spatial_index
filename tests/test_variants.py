

# --- DEL-62: the services-level variant channel --------------------------
def test_variant_overrides_ignore_a_services_block():
    """`VARIANTS` may carry `services` (DEL-62); it is not methodology, so
    the reference-kwarg walker must skip it rather than reject it."""
    from tests.reference_impl import _variant_overrides

    assert _variant_overrides(
        {"services": {"point": {"bank": "x"}}}) == {}


def test_a_variant_services_block_reaches_the_derived_profile(tmp_path):
    """The channel's other half: `oracle_profile_path` writes the overlay
    into `raw["services"]` verbatim, beside (not through) the methodology
    overrides."""
    from delhi_psi.config import load_config
    from tests.oraculum_fixtures import oracle_profile_path

    path = oracle_profile_path(
        "code-2025", tmp_path,
        services_overrides={"point": {"bank": "Public Services/Ration/Ration.shp"}})
    cfg = load_config(path, data_dir=tmp_path)
    assert cfg.services.point["bank"] == "Public Services/Ration/Ration.shp"
    assert cfg.services.point["ration"] == "Public Services/Ration/Ration.shp"
    assert cfg.services.point["school"].endswith("schools7760.shp")
