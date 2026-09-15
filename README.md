# GIS-Data-Clustering

Cursor Agent Skill for **CRS-safe** geospatial clustering (points, polygons, raster samples).

Algorithms: **HDBSCAN**, **DBSCAN**, **KMeans**, **Agglomerative**, plus spatial + attribute features.

Distances are computed in a **metric projected CRS**. Do not run Euclidean clustering on WGS84 lon/lat.

| File | Audience | Role |
| --- | --- | --- |
| `SKILL.md` | Agent | When to use, workflow, hard rules |
| `README.md` | Humans | Install and quick start |
| `reference.md` | On demand | CRS, tuning, metrics |
| `examples.md` | Both | Copy-paste templates |

## Install (Cursor)

Copy this folder to:

```text
~/.cursor/skills/gis-data-clustering/
```

Or into a project:

```text
.cursor/skills/gis-data-clustering/
```

Then mention `@gis-data-clustering` or ask for spatial / POI clustering.

## Dependencies

```bash
pip install geopandas scikit-learn hdbscan pyproj shapely
# optional: rasterio libpysal
```

No API keys. No remote backend. Runs locally.

## Algorithm choice

| Situation | Default |
| --- | --- |
| Uneven density (POIs / events) | **HDBSCAN** |
| Known neighborhood radius (meters) | **DBSCAN** (`eps` in meters) |
| Known number of groups | **KMeans** |
| Hierarchical / connectivity constraints | **Agglomerative** |

## Quick start

```python
import numpy as np
import geopandas as gpd
from sklearn.cluster import DBSCAN

gdf = gpd.read_file("pois.geojson")
gdf_m = gdf.to_crs(gdf.estimate_utm_crs())
coords = np.column_stack([gdf_m.geometry.x, gdf_m.geometry.y])

gdf_m["cluster"] = DBSCAN(eps=400, min_samples=5).fit_predict(coords)
gdf_m.to_crs(gdf.crs).to_file("clustered.gpkg", layer="clusters", driver="GPKG")
```

`-1` means noise (DBSCAN / HDBSCAN). Keep it unless you explicitly decide otherwise.

More templates: [`examples.md`](examples.md). Tuning notes: [`reference.md`](reference.md).

## Hard rules

1. Project first, cluster second.
2. State units: `eps` is always meters.
3. Do not invent silhouette / other scores when they do not apply.
4. Prefer HDBSCAN or MiniBatchKMeans for very large point sets.

## License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
