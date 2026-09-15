# GIS Clustering Examples

可复制模板。默认假设输入为点图层。

## 1. DBSCAN（已知半径，米）

```python
import numpy as np
import geopandas as gpd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

def cluster_dbscan(path, eps_m=400, min_samples=5, out_path="clustered_dbscan.gpkg"):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError("输入缺少 CRS")

    src_crs = gdf.crs
    gdf_m = gdf.to_crs(gdf.estimate_utm_crs())
    coords = np.column_stack([gdf_m.geometry.x, gdf_m.geometry.y])

    labels = DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(coords)
    gdf_m["cluster"] = labels

    mask = labels >= 0
    n_clusters = len(set(labels[mask])) if mask.any() else 0
    noise_ratio = float((labels < 0).mean())
    sil = None
    if n_clusters >= 2 and mask.sum() > n_clusters:
        sil = float(silhouette_score(coords[mask], labels[mask]))

    summary = {
        "algorithm": "DBSCAN",
        "eps_m": eps_m,
        "min_samples": min_samples,
        "projected_crs": gdf_m.crs.to_string(),
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "silhouette": sil,
    }

    gdf_m.to_crs(src_crs).to_file(out_path, layer="clusters", driver="GPKG")
    return gdf_m, summary
```

## 2. HDBSCAN（密度不均，默认首选）

```python
import numpy as np
import geopandas as gpd
import hdbscan
from sklearn.metrics import silhouette_score

def cluster_hdbscan(path, min_cluster_size=10, min_samples=5, out_path="clustered_hdbscan.gpkg"):
    gdf = gpd.read_file(path)
    src_crs = gdf.crs
    gdf_m = gdf.to_crs(gdf.estimate_utm_crs())
    coords = np.column_stack([gdf_m.geometry.x, gdf_m.geometry.y])

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(coords)
    gdf_m["cluster"] = labels
    gdf_m["prob"] = clusterer.probabilities_

    mask = labels >= 0
    n_clusters = len(set(labels[mask])) if mask.any() else 0
    sil = None
    if n_clusters >= 2 and mask.sum() > n_clusters:
        sil = float(silhouette_score(coords[mask], labels[mask]))

    summary = {
        "algorithm": "HDBSCAN",
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "projected_crs": gdf_m.crs.to_string(),
        "n_clusters": n_clusters,
        "noise_ratio": float((labels < 0).mean()),
        "silhouette": sil,
    }

    gdf_m.to_crs(src_crs).to_file(out_path, layer="clusters", driver="GPKG")
    return gdf_m, summary
```

## 3. KMeans（已知簇数）

```python
import numpy as np
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def cluster_kmeans(path, n_clusters=5, out_path="clustered_kmeans.gpkg"):
    gdf = gpd.read_file(path)
    src_crs = gdf.crs
    gdf_m = gdf.to_crs(gdf.estimate_utm_crs())
    coords = np.column_stack([gdf_m.geometry.x, gdf_m.geometry.y])

    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = model.fit_predict(coords)
    gdf_m["cluster"] = labels

    sil = float(silhouette_score(coords, labels)) if n_clusters >= 2 else None
    centers = gpd.GeoDataFrame(
        {"cluster": range(n_clusters)},
        geometry=gpd.points_from_xy(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1]),
        crs=gdf_m.crs,
    )

    summary = {
        "algorithm": "KMeans",
        "n_clusters": n_clusters,
        "projected_crs": gdf_m.crs.to_string(),
        "silhouette": sil,
        "inertia": float(model.inertia_),
    }

    gdf_m.to_crs(src_crs).to_file(out_path, layer="clusters", driver="GPKG")
    return gdf_m, centers.to_crs(src_crs), summary
```

## 4. 空间 + 属性混合

```python
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def cluster_spatial_attribute(
    path,
    attr_cols,
    n_clusters=5,
    spatial_weight=1.0,
    out_path="clustered_mixed.gpkg",
):
    gdf = gpd.read_file(path)
    src_crs = gdf.crs
    gdf_m = gdf.to_crs(gdf.estimate_utm_crs())

    xy = np.column_stack([gdf_m.geometry.x, gdf_m.geometry.y])
    attrs = gdf_m[attr_cols].to_numpy(dtype=float)

    xy_s = StandardScaler().fit_transform(xy) * spatial_weight
    attr_s = StandardScaler().fit_transform(attrs)
    X = np.hstack([xy_s, attr_s])

    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(X)
    gdf_m["cluster"] = labels
    gdf_m.to_crs(src_crs).to_file(out_path, layer="clusters", driver="GPKG")
    return gdf_m
```

## 5. 簇面（凸包）导出

```python
def clusters_to_hulls(gdf_m, out_path="cluster_hulls.gpkg"):
    parts = []
    for cid, sub in gdf_m[gdf_m["cluster"] >= 0].groupby("cluster"):
        if len(sub) < 3:
            continue
        geom = sub.geometry.union_all().convex_hull
        parts.append({"cluster": int(cid), "n": len(sub), "geometry": geom})

    hulls = gpd.GeoDataFrame(parts, crs=gdf_m.crs)
    hulls.to_file(out_path, layer="hulls", driver="GPKG")
    return hulls
```

点数极少的簇跳过凸包。用户要更贴合边界时改用 buffer + unary_union，并声明不是 alpha shape。

## 6. 扫 `eps`（DBSCAN）

```python
def sweep_eps(coords, eps_list_m, min_samples=5):
    from sklearn.cluster import DBSCAN

    rows = []
    for eps in eps_list_m:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
        mask = labels >= 0
        n_clusters = len(set(labels[mask])) if mask.any() else 0
        rows.append({
            "eps_m": eps,
            "n_clusters": n_clusters,
            "noise_ratio": float((labels < 0).mean()),
        })
    return rows
```

把表给用户看，再锁定最终 `eps`。不要自动挑选“最优”却不解释业务半径。
