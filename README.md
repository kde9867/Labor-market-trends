# Labor-market-trends

## 개요
다양한 플랫폼에서 노동시장 관련 데이터를 수집하고, 임베딩 기법을 활용하여 트렌드를 분석하는 종합적인 파이프라인

## 구조
```
Labor-market-trends/
├── data/                   # 데이터 수집 및 전처리
├── datasets/               # 데이터 저장소
├── model/                  # 임베딩 및 분석 모델
├── output/                 # 분석 결과 저장
└──  main.ipynb             # 메인 실행 파일
```

## 환경 설정

프로젝트를 실행하기 전에 다음 파일들에서 경로를 설정해야 합니다:

### 1. `data/preprocessing.py`
```python
DATA_PATH = '.path/Labor-market-trends/datasets'
```

### 2. 데이터 수집 파일들
각 플랫폼별 데이터 수집 스크립트에서 출력 경로를 설정:
```python
# saramin 수집 스크립트
OUTPUT_CSV_FILE = ".path/Labor-market-trends/datasets/saramin_new.csv"

# wanted 수집 스크립트
OUTPUT_CSV_FILE = ".path/Labor-market-trends/datasets/wanted_new.csv"

# work24 수집 스크립트
OUTPUT_CSV_FILE = ".path/Labor-market-trends/datasets/work24_new.csv"
```

### 3. `model/embedding.py`
```python
DATA_PATH = '.path/Labor-market-trends/datasets'

# 모델 경로 설정 (환경에 따라 선택)
# 옵션 1: 로컬 모델 사용
model_path = ".path/model_name"

# 옵션 2: Hugging Face Hub에서 직접 다운로드
model_path = "jhgan/ko-sbert-multitask"

```

### 4. `main.ipynb`
```python
sys.path.append('.path/Labor-market-trends/model')
```

**참고**: 모든 `.path/`는 실제 프로젝트의 절대 경로 또는 상대 경로로 변경해야 합니다.

## 상세 파이프라인

### 1. 데이터 수집 단계 (`data/` 폴더)

#### 1.1 데이터 수집 모듈
- **기능**: 각 플랫폼별 데이터 수집 스크립트
- **저장 위치**: 수집된 데이터는 `datasets/` 폴더에 CSV 형식으로 저장
- **주요 구성**:
  - 플랫폼별 수집 스크립트
  - 플랫폼별 API KEY 발급 필수

#### 1.2 데이터 전처리 모듈
- **입력**: `datasets/` 폴더의 raw 데이터
- **처리**: 데이터 정제, 형식 통일, 결측치 처리
- **출력**: 전처리된 데이터를 `datasets/` 폴더에 저장

**유지보수 필요 사항**:
- 현재 기존에 수집된 raw 데이터를 기반으로 작동
- 지속적 데이터 수집을 위해서는 스케줄링 및 자동화 구현 필요

### 2. 데이터 저장 (`datasets/` 폴더)

#### 폴더 구조
```
datasets/
├── work24.csv            # 원본 데이터
├── saramin.csv           # 원본 데이터
├──wanted.csv             # 원본 데이터
├── work24_processed.csv  # 전처리된 데이터
├── saramin_processed.csv # 전처리된 데이터
└── wanted_processed.csv  # 전처리된 데이터
```

### 3. 모델링 및 분석 (`model/` 폴더)

#### 3.1 임베딩 생성
- **목적**: 채용 공고 텍스트 데이터를 벡터 공간으로 변환
- **출력**: 각 텍스트의 벡터 표현 -> .pkl 파일로 저장

#### 3.2 임베딩 지형도 (Embedding Map)
- **목적**: 국내 채용시장의 채용공고 임베딩 지형도 구현
- **기술**: UMAP 차원 축소 후 KDE 시각화
- **활용**: 데이터 패턴 및 관계 시각적 파악

#### 3.3 클러스터링
- **목적**: 유사한 특성의 데이터 그룹화
- **결과**: 노동시장 트렌드별 그룹 식별

### 4. 실행 파일 (`main.ipynb`)

### 5. 결과 저장 (`output/` 폴더)

#### 주요 출력물
- **임베딩 지형도**: 인터랙티브 HTML 형식
- **클러스터 분석 결과**: 그룹별 특성 정리

📩 Should you have any questions, please contact me at the following email address: **kde9867@gmail.com !**
