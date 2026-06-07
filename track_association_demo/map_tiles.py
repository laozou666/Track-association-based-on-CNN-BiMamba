"""Download map tiles for a track bounding box and crop to the view area."""

import functools
import math
import urllib.error
import urllib.request
from io import BytesIO

import numpy as np
from PIL import Image

TILE_SIZE = 256
MAX_TILES = 36
USER_AGENT = "track-association-demo/1.0"

TILE_SOURCES = [
    "https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
]


def deg2tile(lat, lon, zoom):
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def tile_bounds(x, y, zoom):
    n = 2 ** zoom
    lon_w = x / n * 360.0 - 180.0
    lon_e = (x + 1) / n * 360.0 - 180.0
    lat_n = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_s = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lon_w, lon_e, lat_s, lat_n


def lonlat_to_global_pixel(lon, lat, zoom):
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n * TILE_SIZE
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n * TILE_SIZE
    return x, y


def _download_tile(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return Image.open(BytesIO(resp.read())).convert("RGB")


def _expand_bounds(lon_min, lon_max, lat_min, lat_max, padding_ratio=0.45):
    lon_span = max(lon_max - lon_min, 0.008)
    lat_span = max(lat_max - lat_min, 0.008)
    pad_lon = max(lon_span * padding_ratio, 0.025)
    pad_lat = max(lat_span * padding_ratio, 0.025)
    return (
        lon_min - pad_lon,
        lon_max + pad_lon,
        lat_min - pad_lat,
        lat_max + pad_lat,
    )


def _pick_zoom(lon_min, lon_max, lat_min, lat_max):
    for zoom in range(15, 5, -1):
        x0, y0 = deg2tile(lat_max, lon_min, zoom)
        x1, y1 = deg2tile(lat_min, lon_max, zoom)
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        count = (x1 - x0 + 1) * (y1 - y0 + 1)
        if count <= MAX_TILES:
            return zoom, x0, x1, y0, y1
    zoom = 6
    x0, y0 = deg2tile(lat_max, lon_min, zoom)
    x1, y1 = deg2tile(lat_min, lon_max, zoom)
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    return zoom, x0, x1, y0, y1


def fetch_map_background(lon_min, lon_max, lat_min, lat_max):
    """Return cropped RGB image and extent [lon_w, lon_e, lat_s, lat_n]."""
    view_lon_min, view_lon_max, view_lat_min, view_lat_max = _expand_bounds(
        lon_min, lon_max, lat_min, lat_max
    )
    key = (
        round(view_lon_min, 5),
        round(view_lon_max, 5),
        round(view_lat_min, 5),
        round(view_lat_max, 5),
    )
    return _fetch_map_background_cached(*key)


@functools.lru_cache(maxsize=24)
def _fetch_map_background_cached(view_lon_min, view_lon_max, view_lat_min, view_lat_max):
    zoom, x0, x1, y0, y1 = _pick_zoom(
        view_lon_min, view_lon_max, view_lat_min, view_lat_max
    )
    tiles_x = x1 - x0 + 1
    tiles_y = y1 - y0 + 1
    canvas = Image.new("RGB", (tiles_x * TILE_SIZE, tiles_y * TILE_SIZE), (12, 78, 110))

    origin_x = x0 * TILE_SIZE
    origin_y = y0 * TILE_SIZE

    for ix, x in enumerate(range(x0, x1 + 1)):
        for iy, y in enumerate(range(y0, y1 + 1)):
            tile = None
            for template in TILE_SOURCES:
                url = template.format(z=zoom, x=x, y=y)
                try:
                    tile = _download_tile(url)
                    break
                except (urllib.error.URLError, TimeoutError, OSError, ValueError):
                    continue
            if tile is not None:
                canvas.paste(tile, (ix * TILE_SIZE, iy * TILE_SIZE))

    left, top = lonlat_to_global_pixel(view_lon_min, view_lat_max, zoom)
    right, bottom = lonlat_to_global_pixel(view_lon_max, view_lat_min, zoom)
    left -= origin_x
    right -= origin_x
    top -= origin_y
    bottom -= origin_y

    left = max(0, int(math.floor(left)))
    top = max(0, int(math.floor(top)))
    right = min(canvas.width, int(math.ceil(right)))
    bottom = min(canvas.height, int(math.ceil(bottom)))

    if right - left < 32 or bottom - top < 32:
        cropped = canvas
        extent = [
            tile_bounds(x0, y0, zoom)[0],
            tile_bounds(x1, y1, zoom)[1],
            tile_bounds(x1, y1, zoom)[2],
            tile_bounds(x0, y0, zoom)[3],
        ]
    else:
        cropped = canvas.crop((left, top, right, bottom))
        extent = [view_lon_min, view_lon_max, view_lat_min, view_lat_max]

    return np.asarray(cropped), extent
