# Copyright 2026 HANUMAN Team
#
# Licensed under the Apache License, Version 2.0

"""Tests for FileDownloader — cache hit, cache miss, URL→filename mapping."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

import pytest
from mars_terrain_exporter.utils.file_downloader import FileDownloader

_SAMPLE_URL = (
    "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn"
    "/HiRISE/DTM_MOLAtopography_DeltaGeoid_Jezero_C_Edited_affine_1m"
    "_Eqc_latTs0_lon0.tif"
)
_EXPECTED_FILENAME = (
    "DTM_MOLAtopography_DeltaGeoid_Jezero_C_Edited_affine_1m_Eqc_latTs0_lon0.tif"
)


class TestFileDownloader:
    def setup_method(self):
        self._tmp = tempfile.mkdtemp()
        self._cache_dir = Path(self._tmp)
        self._dl = FileDownloader(self._cache_dir)

    def test_cache_hit_no_network(self):
        """If the file already exists, no HTTP request should be made."""
        cached = self._cache_dir / _EXPECTED_FILENAME
        cached.write_bytes(b"fake tiff data")

        with patch("requests.get") as mock_get:
            result = self._dl.download(_SAMPLE_URL)
            mock_get.assert_not_called()

        assert result == cached

    def test_cache_miss_downloads_file(self):
        """On a cache miss the downloader streams the response to disk."""
        fake_content = b"\x00" * 256

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.raise_for_status = MagicMock()
        mock_resp.headers = {"content-length": str(len(fake_content))}
        mock_resp.iter_content = MagicMock(return_value=[fake_content])

        with patch("requests.get", return_value=mock_resp):
            result = self._dl.download(_SAMPLE_URL)

        assert result.exists()
        assert result.read_bytes() == fake_content

    def test_url_to_filename_mapping(self):
        """Filename extracted from URL basename must match expectation."""
        cached = self._cache_dir / _EXPECTED_FILENAME
        cached.write_bytes(b"x")
        result = self._dl.download(_SAMPLE_URL)
        assert result.name == _EXPECTED_FILENAME

    def test_cache_dir_created_if_missing(self):
        new_dir = self._cache_dir / "sub" / "deep"
        FileDownloader(new_dir)
        assert new_dir.is_dir()
