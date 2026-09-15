---
name: gis-data-clustering
description: Cluster geospatial points, polygons, and raster samples with CRS-safe pipelines (DBSCAN, HDBSCAN, KMeans, agglomerative). Use when the user asks for GIS clustering, spatial hotspots, POI grouping, facility zoning, or unsupervised spatial groups; also when writing technical articles about GIS clustering.
---

# GIS Data Clustering

对矢量点/面或栅格采样做空间聚类。投影坐标系下算距离。禁止在 WGS84 经纬度上直接欧氏聚类。

## When to use

- 用户提到：空间聚类、POI 聚团、热点分区、设施分组、无监督分区
- 需要在 GeoPandas / scikit-learn / HDBSCAN 里落地聚类
- 需要按 `prompt.txt` 写 GIS 聚类技术文（见文末写作约束）

## Quick decision

| 场景 | 默认算法 | 关键参数 |
| --- | --- | --- |
| 密度不均的 POI / 事件点 | **HDBSCAN** | `min_cluster_size`, `min_samples` |
| 有明确邻域半径（米） | **DBSCAN** | `eps`（米）, `min_samples` |
| 已知簇数、偏紧凑分区 | **KMeans** | `n_clusters` |
| 需要层次树 / 可变切分 | **Agglomerative** | `n_clusters` 或 `distance_threshold` |
| 仅属性、弱空间约束 | 属性聚类 + 事后空间校验 | 特征缩放 + silhouette |

默认优先 **HDBSCAN**。用户明确给出半径时用 **DBSCAN**。用户明确给出簇数时用 **KMeans**。

## Workflow

复制进度并逐步完成：

```
Task Progress:
- [ ] 1. 读数据与 CRS
- [ ] 2. 投影到米制 CRS
- [ ] 3. 选算法与参数
- [ ] 4. 聚类
- [ ] 5. 评价与噪声检查
- [ ] 6. 写回矢量并可视化摘要
```

### 1. 读数据与 CRS

```python
import geopandas as gpd

gdf = gpd.read_file(path)
assert gdf.crs is not None, "输入必须带 CRS"
# 点图层：用 centroid 把面转点再聚类（需用户确认）
```

### 2. 投影到米制 CRS

```python
# 中国范围优先 UTM 或 CGCS2000 / 3-degree Gauss-Kruger
# 未知区域：用估计 UTM
gdf_m = gdf.to_crs(gdf.estimate_utm_crs())
coords = gdf_m.get_coordinates().to_numpy()  # GeoPandas >= 0.14
# 兼容写法：
# coords = np.column_stack([gdf_m.geometry.x, gdf_m.geometry.y])
```

> 距离单位必须是米。`eps=0.01` 在经纬度里没有业务含义。

### 3. 选算法

- 噪声点可接受 → HDBSCAN / DBSCAN
- 每个点必须入簇 → KMeans / Agglomerative（或 HDBSCAN 后把噪声并到最近簇，需显式说明）
- 混合属性（收入、流量）+ 坐标 → 坐标标准化到同类量纲后再拼特征；或先空间聚类再汇总属性

### 4. 聚类（默认模板）

详细代码见 [examples.md](examples.md)。最小可运行路径：

```python
from sklearn.cluster import DBSCAN
import numpy as np

coords = np.column_stack([gdf_m.geometry.x, gdf_m.geometry.y])
labels = DBSCAN(eps=400, min_samples=5, metric="euclidean").fit_predict(coords)
gdf_m["cluster"] = labels  # -1 = 噪声
```

HDBSCAN：

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric="euclidean")
gdf_m["cluster"] = clusterer.fit_predict(coords)
```

### 5. 评价

```python
from sklearn.metrics import silhouette_score

mask = gdf_m["cluster"] >= 0
n_clusters = gdf_m.loc[mask, "cluster"].nunique()
noise_ratio = (gdf_m["cluster"] < 0).mean()

if n_clusters >= 2 and mask.sum() > n_clusters:
    score = silhouette_score(coords[mask], gdf_m.loc[mask, "cluster"])
else:
    score = None
```

报告至少给出：簇数、噪声比例、各簇点数、投影 CRS、算法与参数。

### 6. 写出

```python
out = gdf_m.to_crs(gdf.crs)  # 写回原 CRS，除非用户指定
out.to_file("clustered.gpkg", layer="clusters", driver="GPKG")
```

可选：按簇 `dissolve` 生成簇凸包 / alpha shape（大数据慎用凸包）。

## Hard rules

1. **先投影，再聚类。** 禁止对 EPSG:4326 坐标直接 `euclidean`。
2. **参数单位写清楚。** `eps` / 半径一律标注米。
3. **噪声是结果的一部分。** DBSCAN/HDBSCAN 的 `-1` 不要默默丢掉，除非用户要求。
4. **不编造指标。** silhouette、CHI、DBI 算不出来就写条件不足。
5. **大样本。** 点数 > 5e4 时优先 HDBSCAN / MiniBatchKMeans，避免全连接层次聚类。
6. **面要素。** 默认对质心聚类；用户要按邻接图聚类时改用空间权重（见 reference）。

## Algorithm notes (short)

- **DBSCAN**：`eps` = 邻域半径（米）。过小 → 全噪声；过大 → 并成一簇。
- **HDBSCAN**：少调 `eps`，调 `min_cluster_size`。密度不均时通常稳于 DBSCAN。
- **KMeans**：假设近似球形、簇大小接近。空间上条带/环状结构会失真。
- **属性+空间**：坐标与属性量纲差几个数量级时必须缩放；或对坐标单独加权。

完整参数与坑见 [reference.md](reference.md)。可复制脚本见 [examples.md](examples.md)。

## Output checklist for the user

交付时按这个顺序说清：

1. 输入图层、点数、原 CRS、投影 CRS
2. 算法与参数（含单位）
3. 簇数、噪声比例、每簇规模
4. 评价指标（有则报，无则说明原因）
5. 输出文件路径与字段名（`cluster`）

## Writing WeChat / 技术文（可选）

用户要求写公众号文、复盘文、教程文时，严格遵循工作区 `prompt.txt`：

- 短句：单句尽量 ≤25 字；一句话一个意思
- 禁用复合长句、翻译腔、感叹号、自我指涉
- 禁用「本文将」「让我们」「总之」「赋能」等套话
- 标题与一/二级标题各只表达一个意思
- 代码只留关键逻辑，标注语言
- 文末单独一行：`日期：YYYY-MM-DD`（用当天）
- 技术事实与代码风格对齐本 skill 与 [examples.md](examples.md)；不虚构论文与性能数字

## Additional resources

- [reference.md](reference.md) — CRS、参数调优、邻接图聚类、评价
- [examples.md](examples.md) — DBSCAN / HDBSCAN / KMeans / 属性混合完整模板
