# GIS Clustering Reference

Agent 按需阅读。日常流程以 `SKILL.md` 为准。

## CRS choices

| 范围 | 建议 |
| --- | --- |
| 单城市 / 省级局部 | `gdf.estimate_utm_crs()` 或当地高斯克吕格 3 度带 |
| 全国跨带 | 分区聚类后合并标签；或等面积投影做粗分区，细部再投 UTM |
| 已是投影（米） | 确认单位为 metre；勿重复投影到错误带 |

检查单位：

```python
assert gdf_m.crs.is_projected
assert gdf_m.crs.axis_info[0].unit_name in ("metre", "meter")
```

## Parameter tuning

### DBSCAN

| 症状 | 调整 |
| --- | --- |
| 几乎全是 `-1` | 增大 `eps`，或减小 `min_samples` |
| 并成 1～2 个大簇 | 减小 `eps` |
| 想对齐步行圈 / 服务半径 | 直接把业务半径赋给 `eps`（米） |

`min_samples` 经验：二维点数据常用 `4～10`。事件稀疏时取小，噪声敏感时取大。

### HDBSCAN

| 参数 | 作用 |
| --- | --- |
| `min_cluster_size` | 最小成簇点数（主旋钮） |
| `min_samples` | 越大越保守，噪声越多 |
| `cluster_selection_method` | `"eom"` 默认；`"leaf"` 更碎、更小簇 |

不必先估 `eps`。密度差一个数量级时优先 HDBSCAN。

### KMeans

- `n_clusters`：用肘部法或 silhouette 扫 `k=2..K`
- 对坐标做 `StandardScaler` 仅在与属性拼接时需要；纯米制坐标通常不缩放
- `MiniBatchKMeans`：大样本默认项

### Agglomerative

- `linkage="ward"` 需欧氏距离，适合紧凑团
- 自定义距离矩阵时用 `linkage="average"` / `"complete"`，并设 `metric="precomputed"`
- 全连接距离矩阵内存约 `O(n²)`，`n>2e4` 慎用

## Spatial + attribute features

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

xy = np.column_stack([gdf_m.geometry.x, gdf_m.geometry.y])
attrs = gdf_m[["attr_a", "attr_b"]].to_numpy(dtype=float)

# 空间权重：把米缩放到与属性同一量级，或显式乘 weight
xy_s = StandardScaler().fit_transform(xy)
attr_s = StandardScaler().fit_transform(attrs)
X = np.hstack([xy_s * spatial_weight, attr_s])
```

`spatial_weight > 1` 强化地理邻近；`< 1` 强化属性相似。写进报告。

## Contiguity clustering (polygons)

面要素按邻接而不是质心距离：

```python
import libpysal
from sklearn.cluster import AgglomerativeClustering

w = libpysal.weights.Queen.from_dataframe(gdf_m, use_index=False)
# 将稀疏邻接转为连通约束较复杂；实务上常用：
# 1) 质心 + 距离聚类
# 2) 或 Spatially Constrained Clustering（如 sklearn 的 connectivity=knn_graph）
from sklearn.neighbors import kneighbors_graph

coords = np.column_stack([gdf_m.geometry.centroid.x, gdf_m.geometry.centroid.y])
connectivity = kneighbors_graph(coords, n_neighbors=8, include_self=False)
model = AgglomerativeClustering(
    n_clusters=8,
    connectivity=connectivity,
    linkage="ward",
)
labels = model.fit_predict(coords)
```

真正的区划约束（强制邻接连通）可用 `spopt`（Region-K-Means、Max-p 等）。用户要行政连续片区时再引入。

## Raster / image samples

对齐 `machine-learning.md` 的栅格采样思路：在有效像元上取坐标+波段，再聚类。

```python
import rasterio
import numpy as np
from sklearn.cluster import MiniBatchKMeans

with rasterio.open(raster_path) as src:
    data = src.read()  # (bands, H, W)
    transform = src.transform
    nodata = src.nodata

valid = np.ones(data.shape[1:], dtype=bool)
if nodata is not None:
    valid &= ~np.any(data == nodata, axis=0)

rows, cols = np.where(valid)
vals = data[:, rows, cols].T
# 可选：把地理坐标并入特征（需投影栅格）
xs, ys = rasterio.transform.xy(transform, rows, cols)
# 先对 vals 标准化再 KMeans
labels = MiniBatchKMeans(n_clusters=5, random_state=42).fit_predict(vals)
```

分类图写回时保持 `transform` 与 `crs`。

## Metrics

| 指标 | 用途 | 注意 |
| --- | --- | --- |
| silhouette | 簇内紧、簇间离 | 需 ≥2 簇；忽略噪声点后算 |
| noise ratio | DBSCAN/HDBSCAN | 业务可接受阈值要用户定 |
| CHI / Davies-Bouldin | 扫参对比 | 勿跨算法硬比绝对值 |
| 业务规则 | 服务半径覆盖、最小规模 | 优先于纯统计分 |

空间自相关（Moran's I）用于检查残差，不替代聚类评价。

## Pitfalls

1. 地理坐标系下 `eps=0.01` ≈ 纬度方向约 1 km，经度方向随纬度变化——结果不可解释。
2. 跨 UTM 带硬投一个带：边缘距离变形，分区失真。
3. 对重复坐标点：DBSCAN 易成微簇；先 `drop_duplicates` 或抖动需声明。
4. 把 `-1` 映射成 `0` 再和真实簇 0 混淆。
5. Web Mercator（EPSG:3857）可聚类，但高纬距离偏差大；局部分析仍优先 UTM / 地方投影。

## Library stack

默认：

- `geopandas`, `pyproj`, `shapely`
- `scikit-learn`
- `hdbscan`（密度不均时）
- `rasterio`（栅格）
- 可选：`libpysal`, `spopt`, `folium` / `matplotlib`

不引入重型深度学习做普通点聚类，除非用户明确要求嵌入式空间聚类。
