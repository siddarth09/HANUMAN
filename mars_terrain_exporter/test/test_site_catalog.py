# Copyright 2026 HANUMAN — MIT License.

"""Tests for the Mars HiRISE site catalog."""

import pytest

from mars_terrain_exporter.utils.site_catalog import list_sites, get_site


class TestCatalogEntry:
    def test_entry_has_required_keys(self):
        entry = get_site("jezero_c")
        assert {"site_code", "site_name", "description", "dem_filename"} <= entry.keys()

    def test_entry_values(self):
        entry = get_site("jezero_c")
        assert entry["site_code"] == "Jezero_C"
        assert entry["site_name"] == "jezero_c"
        assert entry["description"] != ""
        assert entry["dem_filename"].endswith(".tif")


class TestSiteCatalog:
    def test_known_sites_present(self):
        names = {e["site_name"] for e in list_sites()}
        for name in ["jezero_c", "jezero_n", "jezero_e", "jezero_w",
                     "jezero_dl", "jezero_cr_north", "jezero_cr_south"]:
            assert name in names

    def test_all_names_are_snake_case(self):
        for entry in list_sites():
            name = entry["site_name"]
            assert name == name.lower()
            assert " " not in name

    def test_all_site_codes_unique(self):
        codes = [e["site_code"] for e in list_sites()]
        assert len(codes) == len(set(codes))


class TestGetSite:
    def test_lookup_by_name(self):
        assert get_site("jezero_c")["site_code"] == "Jezero_C"

    def test_lookup_by_code(self):
        assert get_site("Jezero_C")["site_name"] == "jezero_c"

    def test_unknown_site_raises(self):
        with pytest.raises(KeyError, match="no_such_site"):
            get_site("no_such_site")

    def test_error_lists_available(self):
        with pytest.raises(KeyError, match="jezero_c"):
            get_site("bad_name")
