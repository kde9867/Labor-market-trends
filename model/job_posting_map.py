import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import umap.umap_ as umap
from pathlib import Path

# 데이터 로딩
def load_embedding_data(filepath: str):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["metadata"], data["embeddings"]


def reduce_to_2d(embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, 
                 cache_path: str = None, force_recompute: bool = False) -> np.ndarray:

    if cache_path and not force_recompute:
        cache_file = Path(cache_path)
        if cache_file.exists():
            coords_2d = np.load(cache_path)
            
            if coords_2d.shape[0] == embeddings.shape[0]:
                return coords_2d
    
    # UMAP 계산
    reducer = umap.UMAP(
        n_components=2, 
        n_neighbors=n_neighbors, 
        min_dist=min_dist, 
        random_state=42, 
        metric="cosine"
    )
    coords_2d = reducer.fit_transform(embeddings)
 
    # 좌표 저장
    if cache_path:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, coords_2d)
        print(f"좌표 저장: {cache_path}")
    
    return coords_2d

# KDE 계산
def compute_kde(X: np.ndarray, Y: np.ndarray, axis_range: tuple, grid_size: int = 200, bw_adjust: float = 0.5):
    xmin, xmax, ymin, ymax = axis_range
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, grid_size), 
                         np.linspace(ymin, ymax, grid_size))
    
    vals = np.vstack([X, Y])
    kde = gaussian_kde(vals)
    kde.set_bandwidth(bw_method=kde.factor * bw_adjust)
    z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    
    return xx, yy, z

# 축 범위 계산
def get_axis_range(coords: np.ndarray, padding: float = 0.05):
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    pad_x = (x_max - x_min) * padding
    pad_y = (y_max - y_min) * padding
    return (x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y)


# 시각화: 전체 데이터 KDE
def plot_kde_all(coords: np.ndarray, title: str = "All Platforms KDE"):
    axis_range = get_axis_range(coords)
    X, Y = coords[:, 0], coords[:, 1]
    
    # 샘플링 (너무 많으면)
    if len(X) > 300000:
        idx = np.random.choice(len(X), 300000, replace=False)
        X, Y = X[idx], Y[idx]
    
    xx, yy, z = compute_kde(X, Y, axis_range)
    
    fig = go.Figure(go.Contour(
        x=xx[0, :], y=yy[:, 0], z=z,
        colorscale="Turbo", showscale=True,
        contours=dict(coloring="heatmap", showlines=True),
        line=dict(width=0.5)
    ))
    
    fig.update_layout(
        title=title, height=600, width=800,
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, range=[axis_range[0], axis_range[1]]),
        yaxis=dict(showgrid=False, zeroline=False, range=[axis_range[2], axis_range[3]], 
                   scaleanchor="x", scaleratio=1)
    )
    
    return fig

# 시각화: 플랫폼 별 KDE
def plot_kde_by_platform(metadata: pd.DataFrame, coords: np.ndarray, 
                         platforms: list = None, colors: dict = None):
    if platforms is None:
        platforms = ["work24", "saramin", "wanted"]
    if colors is None:
        colors = {"work24": "#88b4a2", "saramin": "#d6997b", "wanted": "royalblue"}
    
    axis_range = get_axis_range(coords)
    fig = make_subplots(rows=1, cols=len(platforms), subplot_titles=platforms)
    
    for i, plat in enumerate(platforms, 1):
        mask = metadata["platform"].str.lower() == plat
        if not mask.any():
            continue
            
        XY = coords[mask.values]
        X, Y = XY[:, 0], XY[:, 1]
        
        if len(X) > 50000:
            idx = np.random.choice(len(X), 50000, replace=False)
            X, Y = X[idx], Y[idx]
        
        xx, yy, z = compute_kde(X, Y, axis_range)
        
        fig.add_trace(go.Contour(
            x=xx[0, :], y=yy[:, 0], z=z,
            colorscale=[[0, "white"], [1, colors[plat]]],
            showscale=False,
            contours=dict(coloring="heatmap"),
            line=dict(width=0.5)
        ), row=1, col=i)
        
        fig.update_xaxes(showgrid=False, zeroline=False, range=[axis_range[0], axis_range[1]], row=1, col=i)
        fig.update_yaxes(showgrid=False, zeroline=False, range=[axis_range[2], axis_range[3]], 
                        scaleanchor=f"x{i}", scaleratio=1, row=1, col=i)
    
    fig.update_layout(height=500, width=400*len(platforms), plot_bgcolor="white")
    return fig

# Wanted: 시간대 별 KDE 시각화
def plot_kde_by_year(metadata: pd.DataFrame, coords: np.ndarray, platform: str = "wanted", bins: list = None, labels: list = None,
                     date_col: str = None):
    if bins is None:
        bins = [2013, 2020, 2023, 2025]
        labels = ["2014-2020", "2021-2023", "2024-2025"]
    
    mask_plat = metadata["platform"].str.lower() == platform
    md = metadata.loc[mask_plat].reset_index(drop=True)
    coords_plat = coords[mask_plat.values]
    
    if date_col is None:
        if "expiration_date" in md.columns:
            date_col = "expiration_date"
        else:
            raise ValueError(f"날짜 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {list(md.columns)}")
    
    md["_date"] = pd.to_datetime(md[date_col], errors="coerce")
    valid_date_mask = md["_date"].notna()
  
    md = md[valid_date_mask].reset_index(drop=True)
    coords_plat = coords_plat[valid_date_mask.values]
    
    if len(md) == 0:
        raise ValueError(f"{platform}에 유효한 날짜 데이터가 없습니다")
    
    md["year_bin"] = pd.cut(md["_date"].dt.year, bins=bins, labels=labels, right=True)
    groups = md.groupby("year_bin", observed=True)
    
    print(f"[{platform}] 연도 구간별 분포:")
    for label in labels:
        count = len(groups.get_group(label)) if label in groups.groups else 0
        print(f"  {label}: {count:,}개")
    
    axis_range = get_axis_range(coords_plat)
    fig = make_subplots(rows=1, cols=len(labels), 
                       subplot_titles=[f"{l} (n={len(groups.get_group(l)) if l in groups.groups else 0})" 
                                      for l in labels])
    
    for i, label in enumerate(labels, 1):
        if label not in groups.groups:
            fig.update_xaxes(showgrid=False, zeroline=False, range=[axis_range[0], axis_range[1]], row=1, col=i)
            fig.update_yaxes(showgrid=False, zeroline=False, range=[axis_range[2], axis_range[3]], 
                            scaleanchor=f"x{i}", scaleratio=1, row=1, col=i)
            continue
            
        group_data = groups.get_group(label)
        group_indices = group_data.index.values
        XY = coords_plat[group_indices]
        X, Y = XY[:, 0], XY[:, 1]
        
        if len(X) < 20:
            fig.update_xaxes(showgrid=False, zeroline=False, range=[axis_range[0], axis_range[1]], row=1, col=i)
            fig.update_yaxes(showgrid=False, zeroline=False, range=[axis_range[2], axis_range[3]], 
                            scaleanchor=f"x{i}", scaleratio=1, row=1, col=i)
            continue
        
        if len(X) > 30000:
            idx = np.random.choice(len(X), 30000, replace=False)
            X, Y = X[idx], Y[idx]
        
        xx, yy, z = compute_kde(X, Y, axis_range, grid_size=150)
        
        fig.add_trace(go.Contour(
            x=xx[0, :], y=yy[:, 0], z=z,
            colorscale="Blues", showscale=False,
            contours=dict(coloring="heatmap"),
            line=dict(width=0.5)
        ), row=1, col=i)
        
        fig.update_xaxes(showgrid=False, zeroline=False, range=[axis_range[0], axis_range[1]], row=1, col=i)
        fig.update_yaxes(showgrid=False, zeroline=False, range=[axis_range[2], axis_range[3]], 
                        scaleanchor=f"x{i}", scaleratio=1, row=1, col=i)
    
    fig.update_layout(height=500, width=400*len(labels), plot_bgcolor="white",
                     title=f"{platform.title()} | Year-wise KDE (by {date_col})")
    return fig


# 피크 및 제목 추출 
def find_top_density_regions(metadata: pd.DataFrame, coords: np.ndarray, 
                             n_regions: int = 5, n_titles: int = 5):
    axis_range = get_axis_range(coords)
    X, Y = coords[:, 0], coords[:, 1]
    
    # KDE 계산
    xx, yy, z = compute_kde(X, Y, axis_range, grid_size=150)
    
    z_flat = z.ravel()
    top_indices = np.argsort(z_flat)[-n_regions * 100:][::-1]
    
    peaks = []
    for idx in top_indices:
        if len(peaks) >= n_regions:
            break
        i, j = idx // z.shape[1], idx % z.shape[1]
        px, py = xx[0, j], yy[i, 0]
        
        too_close = any(np.sqrt((px - p[0])**2 + (py - p[1])**2) < 1.0 for p in peaks)
        if not too_close:
            peaks.append((px, py, z[i, j]))

    results = []
    for region_id, (px, py, pz) in enumerate(peaks, 1):
        dist = np.sqrt((X - px)**2 + (Y - py)**2)
        nearest = np.argsort(dist)[:n_titles]
        
        for rank, idx in enumerate(nearest, 1):
            row = metadata.iloc[idx]
            title = row.get("title") or row.get("position") or row.get("text", "")
            results.append({
                "region": region_id,
                "rank": rank,
                "title": str(title)[:100],
                "platform": row.get("platform", ""),
                "distance": dist[idx]
            })
    
    return pd.DataFrame(results)


def save_html(fig: go.Figure, filename: str):
    """HTML 저장"""
    fig.write_html(filename)
    print(f"[저장] {filename}")