# Labor-market-trends

## 개요
다양한 플랫폼에서 노동시장 관련 데이터를 수집하고, 임베딩 기법을 활용하여 트렌드를 분석하는 종합적인 파이프라인

---

## 구조

```
Labor-market-trends/
├── data/                   # 데이터 수집 및 전처리
├── datasets/               # 데이터 저장소
├── model/                  # 임베딩 및 분석 모델
├── output/                 # 분석 결과 저장
├── Embedding_map.ipynb     # 메인 실행 파일
└── Clustering.ipynb        # 메인 실행 파일
```

---

## 상세 파이프라인

### 1. 데이터 수집 단계 (`data/` 폴더)

#### 1.1 데이터 수집 모듈
- **기능**: 각 플랫폼별 데이터 수집 스크립트
- **저장 위치**: 수집된 데이터는 `datasets/` 폴더에 CSV 형식으로 저장
- **주요 구성**:
  - 플랫폼별 수집 스크립트
  - 플랫폼별 API KEY 발급 필수

#### 1.3 데이터 전처리 모듈
- **입력**: `datasets/` 폴더의 raw 데이터
- **처리**: 데이터 정제, 형식 통일, 결측치 처리
- **출력**: 전처리된 데이터를 `datasets/` 폴더에 저장

 **유지보수 필요 사항**:
- 현재 기존에 수집된 raw 데이터를 기반으로 작동
- 지속적 데이터 수집을 위해서는 스케줄링 및 자동화 구현 필요

---

### 2. 데이터 저장 (`datasets/` 폴더)

#### 2.1 폴더 구조
```
datasets/
├── raw/                   # 원본 데이터
│   ├── platform1_raw.csv
│   ├── platform2_raw.csv
│   └── ...
└── processed/             # 전처리된 데이터
    ├── platform1_processed.csv
    ├── platform2_processed.csv
    └── ...
```

#### 2.2 데이터 형식
- **Raw 데이터**: 각 플랫폼에서 수집한 원본 형태
- **Processed 데이터**: 분석 가능한 형태로 정제된 데이터

---

### 3. 모델링 및 분석 (`model/` 폴더)

#### 3.1 임베딩 생성
- **목적**: 텍스트 데이터를 벡터 공간으로 변환
- **출력**: 각 텍스트의 벡터 표현

#### 3.2 임베딩 지형도 (Embedding Map)
- **목적**: 임베딩 벡터의 2D/3D 시각화
- **기술**: t-SNE, UMAP 등 차원 축소 기법
- **활용**: 데이터 패턴 및 관계 시각적 파악

#### 3.3 클러스터링
- **목적**: 유사한 특성의 데이터 그룹화
- **기술**: K-means, DBSCAN 등 클러스터링 알고리즘
- **결과**: 노동시장 트렌드별 그룹 식별

---

### 4. 실행 파일 (`main.ipynb`)

#### 4.1 메인 노트북 구성
```python
# 1. 데이터 로드
import pandas as pd
data = pd.read_csv('datasets/processed/data.csv')

# 2. 임베딩 지형도 생성
%run embedding_map.ipynb

# 3. 클러스터링 수행
%run clustering.ipynb
```

#### 4.2 서브 노트북
- **embedding_map.ipynb**: 임베딩 시각화 전용 노트북
- **clustering.ipynb**: 클러스터링 분석 전용 노트북

---

### 5. 결과 저장 (`output/` 폴더)

#### 5.1 저장되는 파일들
```
output/
├── embeddings/            # 임베딩 벡터
│   └── embeddings.pkl
├── visualizations/        # 시각화 결과
│   ├── embedding_map.html
│   └── clusters.png
└── reports/              # 분석 보고서
    └── analysis_report.pdf
```

#### 5.2 주요 출력물
- **임베딩 지형도**: 인터랙티브 HTML 형식
- **클러스터 분석 결과**: 그룹별 특성 정리
- **트렌드 리포트**: 시장 인사이트 정리

---

### 환경 설정
```bash
# 필요한 패키지 설치
pip install -r requirements.txt
```

