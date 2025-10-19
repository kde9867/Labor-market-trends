"""
Job Posting Clustering Analysis Module
센트로이드 기반 시계열 분석
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import re

# ============================================================
# 데이터 필터링
# ============================================================

def filter_platform(metadata: pd.DataFrame, embeddings: np.ndarray, 
                    platform: str) -> Tuple[pd.DataFrame, np.ndarray]:
    mask = metadata["platform"].str.lower() == platform.lower()
    md_filtered = metadata.loc[mask].reset_index(drop=True)
    emb_filtered = embeddings[mask.values]
    print(f"[{platform}] 필터링: {len(md_filtered):,}개")
    return md_filtered, emb_filtered


def filter_ai_jobs(metadata: pd.DataFrame, embeddings: np.ndarray,
                   ai_keywords: list = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """AI 관련 직무 필터링"""
    if ai_keywords is None:
        ai_keywords = ['인공지능', 'AI', '딥러닝', 'ML', 'SW', '로보틱스', 'LLM', 'ChatGPT']
    
    pattern = '|'.join([re.escape(kw) for kw in ai_keywords])
    mask = pd.Series([False] * len(metadata))
    
    for col in metadata.columns:
        if metadata[col].dtype == 'object':
            col_mask = metadata[col].fillna('').astype(str).str.contains(
                pattern, case=False, na=False, regex=True
            )
            mask |= col_mask
    
    md_ai = metadata[mask].reset_index(drop=True)
    emb_ai = embeddings[mask.values]
    
    print(f"AI 직무 필터링: {len(md_ai):,}개 ({len(md_ai)/len(metadata)*100:.1f}%)")
    return md_ai, emb_ai


# ============================================================
# 시간 기간 설정
# ============================================================

def add_time_periods(metadata: pd.DataFrame, date_col: str, 
                     period_months: int = 24) -> pd.DataFrame:
    """날짜 기반 기간 추가"""
    md = metadata.copy()
    
    # 날짜 파싱
    md["date"] = pd.to_datetime(md[date_col], errors="coerce")
    md = md[md["date"].notna()].reset_index(drop=True)
    
    if len(md) == 0:
        raise ValueError(f"유효한 날짜 데이터가 없습니다 (컬럼: {date_col})")
    
    # 연도/월 추출
    md["year"] = md["date"].dt.year
    md["month"] = md["date"].dt.month
    
    # 기간 계산 (월 단위 기준)
    min_date = pd.Timestamp(md["date"].min())
    min_year = min_date.year
    min_month = min_date.month
    
    md["months_from_start"] = ((md["year"] - min_year) * 12 + 
                                (md["month"] - min_month))
    md["period"] = (md["months_from_start"] // period_months) * period_months
    
    # 기간 라벨 (YYYY-MM 형식)
    md["period_start"] = min_date + pd.to_timedelta(md["period"] * 30.44, unit='D')
    md["period_label"] = md["period_start"].dt.strftime("%Y-%m")
    
    print(f"날짜 범위: {md['date'].min().date()} ~ {md['date'].max().date()}")
    print(f"기간 단위: {period_months}개월")
    print(f"총 기간 수: {md['period'].nunique()}")
    
    return md


# ============================================================
# PCA 차원 축소
# ============================================================

def perform_pca(embeddings: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, PCA, StandardScaler]:
    """PCA 차원 축소"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)
    
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)
    
    print(f"PCA 설명 분산: {pca.explained_variance_ratio_}")
    print(f"총 설명 분산: {pca.explained_variance_ratio_.sum():.2%}")
    
    return X_reduced, pca, scaler


# ============================================================
# 센트로이드 계산 및 대표 문서 추출
# ============================================================

def calculate_centroids(metadata: pd.DataFrame, embeddings_2d: np.ndarray,
                       min_samples: int = 10) -> pd.DataFrame:
    """기간별 센트로이드 계산"""
    periods = sorted(metadata["period"].unique())
    centroids = []
    
    for period in periods:
        period_mask = metadata["period"] == period
        period_data = embeddings_2d[period_mask]
        
        if len(period_data) < min_samples:
            continue
        
        # 기간 라벨 (첫 번째 항목에서 가져오기)
        period_label = metadata.loc[period_mask, "period_label"].iloc[0]
        
        centroids.append({
            "period": period,
            "period_label": period_label,
            "centroid_x": period_data[:, 0].mean(),
            "centroid_y": period_data[:, 1].mean(),
            "std_x": period_data[:, 0].std(),
            "std_y": period_data[:, 1].std(),
            "n_samples": len(period_data)
        })
    
    return pd.DataFrame(centroids)


def extract_representative_docs(metadata: pd.DataFrame, embeddings_2d: np.ndarray,
                                centroids_df: pd.DataFrame, n_docs: int = 5) -> Dict:
    """각 센트로이드에서 가장 가까운 대표 문서 추출"""
    representative_docs = {}
    
    for _, centroid in centroids_df.iterrows():
        period = centroid["period"]
        period_label = centroid["period_label"]
        cx, cy = centroid["centroid_x"], centroid["centroid_y"]
        
        # 해당 기간 데이터
        period_mask = metadata["period"] == period
        period_indices = np.where(period_mask)[0]
        period_coords = embeddings_2d[period_mask]
        
        # 센트로이드와의 거리 계산
        distances = np.sqrt((period_coords[:, 0] - cx)**2 + 
                           (period_coords[:, 1] - cy)**2)
        
        # 가장 가까운 n개 선택
        nearest_indices = np.argsort(distances)[:n_docs]
        original_indices = period_indices[nearest_indices]
        
        # 문서 정보 추출
        docs = []
        for idx in original_indices:
            row = metadata.iloc[idx]
            # 제목 추출 (여러 컬럼 시도)
            title = ""
            for col in ["title", "position", "text_original", "text"]:
                if col in row and pd.notna(row[col]) and str(row[col]).strip():
                    title = str(row[col]).strip()[:100]
                    break
            
            docs.append({
                "title": title if title else "(제목 없음)",
                "platform": row.get("platform", ""),
                "date": row.get("date", ""),
                "distance": distances[nearest_indices[len(docs)]]
            })
        
        representative_docs[period_label] = docs
    
    return representative_docs


# ============================================================
# 시각화 (라벨 겹침 개선)
# ============================================================

def plot_centroid_movement(ax, centroids_df: pd.DataFrame, title: str,
                          color_main: str = "darkblue", label_offset_scale: float = 1.5):
    """센트로이드 이동 경로 시각화 (라벨 겹침 방지 강화)"""
    if len(centroids_df) == 0:
        ax.text(0.5, 0.5, "데이터 부족", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return
    
    # 이동 경로 선
    ax.plot(centroids_df["centroid_x"], centroids_df["centroid_y"],
            "o-", color=color_main, linewidth=2.5, markersize=12, alpha=0.8)
    
    # 그라데이션 포인트
    colors = plt.cm.summer(np.linspace(0.2, 0.8, len(centroids_df)))
    for idx, row in centroids_df.iterrows():
        ax.scatter(row["centroid_x"], row["centroid_y"],
                  c=[colors[idx]], s=300, edgecolors="black", linewidth=2, zorder=5)
    
    # 라벨 배치 (겹침 방지 강화)
    placed_boxes = []  # (x, y, width, height) 형태로 저장
    
    for idx, row in centroids_df.iterrows():
        base_x, base_y = row["centroid_x"], row["centroid_y"]
        
        # 라벨 위치 후보 (거리 증가 & 더 다양한 각도)
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)  # 16방향
        distances = [60, 80, 100]  # 여러 거리
        candidates = []
        for dist in distances:
            for angle in angles:
                offset_x = dist * np.cos(angle)
                offset_y = dist * np.sin(angle)
                candidates.append((offset_x, offset_y))
        
        # 겹치지 않는 위치 찾기
        best_offset = candidates[idx % len(candidates)]
        max_min_dist = 0
        
        # 라벨 박스 크기 추정 (대략적)
        label_width = 0.8
        label_height = 0.5
        
        for offset_x, offset_y in candidates:
            ox = offset_x * label_offset_scale
            oy = offset_y * label_offset_scale
            label_x = base_x + ox / 20
            label_y = base_y + oy / 20
            
            # 현재 라벨 박스
            box = (label_x - label_width/2, label_y - label_height/2, 
                   label_width, label_height)
            
            # 기존 박스들과 겹침 체크
            min_dist = float('inf')
            overlaps = False
            
            for placed_box in placed_boxes:
                px, py, pw, ph = placed_box
                # 박스 중심 간 거리
                dist = np.sqrt((label_x - (px + pw/2))**2 + 
                              (label_y - (py + ph/2))**2)
                min_dist = min(min_dist, dist)
                
                # 박스 겹침 체크
                if not (label_x + label_width/2 < px or 
                       label_x - label_width/2 > px + pw or
                       label_y + label_height/2 < py or
                       label_y - label_height/2 > py + ph):
                    overlaps = True
                    break
            
            # 겹치지 않고 기존 라벨과 가장 먼 위치 선택
            if not overlaps and min_dist > max_min_dist:
                max_min_dist = min_dist
                best_offset = (ox, oy)
        
        offset_x, offset_y = best_offset
        
        # 라벨 위치 저장
        label_x = base_x + offset_x / 20
        label_y = base_y + offset_y / 20
        placed_boxes.append((label_x - label_width/2, label_y - label_height/2,
                            label_width, label_height))
        
        # 라벨 텍스트
        label = f"{row['period_label']}\n(n={row['n_samples']:,})"
        
        ax.annotate(label, (base_x, base_y),
                   xytext=(offset_x, offset_y),
                   textcoords="offset points",
                   fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                            alpha=0.95, edgecolor=color_main, linewidth=1.5),
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1",
                                  color="gray", alpha=0.7, linewidth=1.5))
    
    # 표준편차 원
    for _, row in centroids_df.iterrows():
        radius = np.sqrt(row["std_x"]**2 + row["std_y"]**2) / 4
        circle = plt.Circle((row["centroid_x"], row["centroid_y"]), radius,
                           fill=False, edgecolor=color_main, alpha=0.2,
                           linestyle="--", linewidth=1)
        ax.add_patch(circle)
    
    # 이동 방향 화살표
    for i in range(len(centroids_df) - 1):
        dx = centroids_df.iloc[i+1]["centroid_x"] - centroids_df.iloc[i]["centroid_x"]
        dy = centroids_df.iloc[i+1]["centroid_y"] - centroids_df.iloc[i]["centroid_y"]
        ax.arrow(centroids_df.iloc[i]["centroid_x"] + dx*0.3,
                centroids_df.iloc[i]["centroid_y"] + dy*0.3,
                dx*0.3, dy*0.3,
                head_width=0.12, head_length=0.08,
                fc=color_main, ec=color_main, alpha=0.4, zorder=1)
    
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")


# ============================================================
# 통계 분석 (대표 문서 포함)
# ============================================================

def print_statistics(centroids_df: pd.DataFrame, title: str, 
                    representative_docs: Dict = None):
    """센트로이드 통계 및 대표 문서 출력"""
 
    if len(centroids_df) == 0:
        print("데이터 없음")
        return
    
    print("\n기간별 센트로이드 및 대표 문서:")
    print("-" * 70)
    
    for _, row in centroids_df.iterrows():
        period_label = row['period_label']
        print(f"\n{period_label}: ({row['centroid_x']:.2f}, {row['centroid_y']:.2f}) "
              f"[N={row['n_samples']:,}]")
        
        # 대표 문서 출력
        if representative_docs and period_label in representative_docs:
            print("   대표 문서:")
            for i, doc in enumerate(representative_docs[period_label], 1):
                print(f"   {i}. [{doc['platform']}] {doc['title']}")
                print(f"      거리: {doc['distance']:.3f}")
    
    if len(centroids_df) > 1:
        # 총 이동 거리
        total_dist = sum(
            np.sqrt((centroids_df.iloc[i+1]["centroid_x"] - centroids_df.iloc[i]["centroid_x"])**2 +
                   (centroids_df.iloc[i+1]["centroid_y"] - centroids_df.iloc[i]["centroid_y"])**2)
            for i in range(len(centroids_df) - 1)
        )
        print(f"\n{'='*70}")
        print(f" 총 이동 거리: {total_dist:.2f}")
        
        # 분산 감소율
        std_x_change = (1 - centroids_df.iloc[-1]["std_x"] / centroids_df.iloc[0]["std_x"]) * 100
        std_y_change = (1 - centroids_df.iloc[-1]["std_y"] / centroids_df.iloc[0]["std_y"]) * 100
        print(f"분산 변화 - PC1: {std_x_change:+.1f}%")
        print(f"분산 변화 - PC2: {std_y_change:+.1f}%")