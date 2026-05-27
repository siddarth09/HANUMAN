# Copyright 2026 HANUMAN Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""File download utility with local filename-based caching.

Identical in structure to the lunar_terrain_exporter downloader so that
the rest of the pipeline can reuse it without modification.
"""

from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm


class FileDownloader:
    """Downloads remote files with simple filename-based caching.

    The cache key is the basename of the URL path, so two different URLs
    that produce the same filename will collide — acceptable for the HiRISE
    catalog because product filenames are unique.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self, url: str) -> Path:
        """Return the local path of *url*, downloading if not already cached."""
        filename = Path(urlparse(url).path).name or "download.tif"
        dest = self._cache_dir / filename

        if dest.exists():
            print(f"  Using cached: {dest}")
            return dest

        print(f"  Downloading: {url}")
        print("  Connecting…", end=" ", flush=True)

        with requests.get(url, stream=True, timeout=(15, 300)) as resp:
            resp.raise_for_status()
            print("connected.")
            total = int(resp.headers.get("content-length", 0))

            with (
                open(dest, "wb") as fh,
                tqdm(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"  {filename}",
                    leave=True,
                ) as bar,
            ):
                for chunk in resp.iter_content(chunk_size=65536):
                    fh.write(chunk)
                    bar.update(len(chunk))

        print(f"  DTM downloaded → {dest}")
        return dest
