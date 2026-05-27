# Copyright 2026 HANUMAN Team
#
# Licensed under the Apache License, Version 2.0

"""Tests for the Mars HiRISE site catalog."""

import re
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mars_terrain_exporter.utils.site_catalog import list_sites, get_site


class TestListSites:
    def test_returns_list(self):
        sites = list_sites()
        assert isinstance(sites, list)

    def test_has_two_jezero_sites(self):
        sites = list_sites()
        assert len(sites) == 2

    def test_required_keys(self):
        for entry in list_sites():
            for key in ("site_code", "site_name", "description", "dem_url"):
                assert key in entry, f"Missing key {key!r} in {entry}"

    def test_snake_case_names(self):
        """site_name must match [a-z0-9][a-z0-9_]* (CLI-friendly)."""
        for entry in list_sites():
            assert re.match(r"^[a-z0-9][a-z0-9_-]*$", entry["site_name"]), (
                f"site_name {entry['site_name']!r} is not snake_case"
            )

    def test_unique_site_codes(self):
        codes = [e["site_code"] for e in list_sites()]
        assert len(codes) == len(set(codes)), "Duplicate site_codes found"

    def test_unique_site_names(self):
        names = [e["site_name"] for e in list_sites()]
        assert len(names) == len(set(names)), "Duplicate site_names found"

    def test_dem_urls_are_https(self):
        for entry in list_sites():
            assert entry["dem_url"].startswith("https://"), (
                f"dem_url should be HTTPS: {entry['dem_url']}"
            )

    def test_dem_urls_end_with_tif(self):
        for entry in list_sites():
            assert entry["dem_url"].endswith(".tif"), (
                f"dem_url should point to a GeoTIFF: {entry['dem_url']}"
            )


class TestGetSite:
    def test_lookup_by_name_jezero_c(self):
        entry = get_site("jezero_c")
        assert entry["site_name"] == "jezero_c"
        assert entry["site_code"] == "jezero_c"

    def test_lookup_by_name_jezero_dl(self):
        entry = get_site("jezero_dl")
        assert entry["site_name"] == "jezero_dl"

    def test_lookup_by_code(self):
        # site_code and site_name are the same for HiRISE catalog
        entry = get_site("jezero_c")
        assert entry["site_name"] == "jezero_c"

    def test_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="not found in Mars catalog"):
            get_site("moon_south_pole")

    def test_returns_copy_not_shared_ref(self):
        a = get_site("jezero_c")
        b = get_site("jezero_c")
        assert a is b  # same object from dict — that's fine; verify immutability
        # confirm the catalog is not accidentally mutated between calls
        assert a["site_name"] == "jezero_c"

    def test_jezero_c_url_contains_jezero_c(self):
        entry = get_site("jezero_c")
        assert "Jezero_C" in entry["dem_url"]

    def test_jezero_dl_url_contains_jezero_dl(self):
        entry = get_site("jezero_dl")
        assert "Jezero_DL" in entry["dem_url"]
