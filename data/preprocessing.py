"""
통합 전처리 모듈
1. 날짜 형식 정규화 (YYYY-MM-DD)
2. 텍스트 전처리 (HTML 태그 제거, URL 제거 등)
3. 플랫폼별 데이터 준비
"""
import os
import re
from datetime import datetime
from typing import List

import pandas as pd

DATA_PATH = '.path/Labor-market-trends/datasets/'

# ============================================================================
# 날짜 정규화 함수들
# ============================================================================

def normalize_work24_date(date_str) -> str:
    if pd.isna(date_str) or date_str == '':
        return None
    
    date_str = str(date_str).strip()
    
    # "채용시까지", "상시채용" 같은 텍스트 제거
    date_str = re.sub(r'채용시까지|상시채용|', '', date_str).strip()
    
    if not date_str:
        return None
    
    try:

        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            datetime.strptime(date_str, '%Y-%m-%d')
            return date_str
       
        if re.match(r'^\d{2}-\d{2}-\d{2}$', date_str):
            date_obj = datetime.strptime(date_str, '%y-%m-%d')
            return date_obj.strftime('%Y-%m-%d') 
        return None
        
    except ValueError:
        return None


def normalize_saramin_date(date_str) -> str:
    if pd.isna(date_str) or date_str == '':
        return None
    
    date_str = str(date_str).strip()
    
    try:
        for fmt in ['%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d']:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue
        return None
    except Exception:
        return None


def normalize_wanted_date(date_str) -> str:
    if pd.isna(date_str) or date_str == '':
        return None
    
    date_str = str(date_str).strip()
    
    try:
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            datetime.strptime(date_str, '%Y-%m-%d')
            return date_str
        return None
    except ValueError:
        return None

# ============================================================================
# 텍스트 전처리 함수들
# ============================================================================

def preprocess_text(text) -> str:
    if pd.isna(text) or text == '':
        return ''
    
    # 문자열로 변환
    text = str(text)
    
    # 1. HTML 태그 제거
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 2. URL 제거
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    
    # 3. 이메일 주소 제거
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)
    
    # 4. 특수문자 정리
    text = re.sub(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ(),.\-/·]', ' ', text)
    
    # 5. 공백 처리
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # 6. 너무 짧은 텍스트는 빈 문자열로 처리
    if len(text) < 2:
        return ''
    return text

def clean_and_validate_texts(texts: List, platform_name: str) -> List[str]:
    original_count = len(texts)
    cleaned_texts = [preprocess_text(text) for text in texts]
    empty_count = sum(1 for text in cleaned_texts if text == '')
    
    print(f"  원본 텍스트 수: {original_count:,}")
    print(f"  전처리 후 빈 텍스트: {empty_count:,}")
    
    return cleaned_texts

# ============================================================================
# 파일 저장 함수
# ============================================================================

def save_with_deduplication(new_data: pd.DataFrame, 
                            output_path: str, 
                            id_column: str) -> pd.DataFrame:

    if os.path.exists(output_path):
        try:
            existing_data = pd.read_csv(output_path)
            print(f"  기존 파일 발견: {len(existing_data):,}개")
            
            # 컬럼이 다르면 백업하고 새로 시작
            if set(existing_data.columns) != set(new_data.columns):
                print(f"     기존: {len(existing_data.columns)}개 컬럼")
                print(f"     신규: {len(new_data.columns)}개 컬럼")
                
                # 백업
                backup_path = output_path.replace('.csv', f'_backup_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv')
                existing_data.to_csv(backup_path, index=False, encoding='utf-8-sig')
                print(f"  → 기존 파일 백업: {backup_path}")
                combined_data = new_data
            else:
                # 컬럼 순서 맞추기
                existing_data = existing_data[new_data.columns]
                
                # ID 기준으로 중복 확인
                existing_ids = set(existing_data[id_column].astype(str))
                new_ids = set(new_data[id_column].astype(str))
                
                duplicate_count = len(existing_ids & new_ids)
                new_count = len(new_ids - existing_ids)
                
                print(f"  기존 ID 중복: {duplicate_count:,}개")
                print(f"  신규 ID: {new_count:,}개")
                
                # 병합 및 중복 제거 (ID만으로)
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                combined_data = combined_data.drop_duplicates(subset=[id_column], keep='last')
                
                final_new = len(combined_data) - len(existing_data)
        
        except Exception as e:
            combined_data = new_data
    else:
        combined_data = new_data
        
    combined_data.to_csv(output_path, index=False, encoding='utf-8-sig')
    return combined_data

# ============================================================================
# 플랫폼별 전처리 함수들
# ============================================================================

def preprocess_work24(input_path: str = os.path.join(DATA_PATH, 'work24.csv')) -> pd.DataFrame:
    output_path = os.path.join(DATA_PATH, 'work24_preprocessed.csv')
    df = pd.read_csv(input_path)
    print(f"  원본: {len(df):,}개, {len(df.columns)}개 컬럼")
    
    # 1. 날짜 정규화
    df['등록일자_normalized'] = df['등록일자'].apply(normalize_work24_date)
    df['마감일자_normalized'] = df['마감일자'].apply(normalize_work24_date)

    # 2. 텍스트 전처리
    print(f"  텍스트 전처리")
    cleaned_texts = clean_and_validate_texts(df['채용제목'].tolist(), 'Work24')
    df['text_cleaned'] = cleaned_texts
    
    df['platform'] = 'work24'
    result = save_with_deduplication(df, output_path, '구인인증번호')
    
    return result


def preprocess_saramin(input_path: str = os.path.join(DATA_PATH, 'saramin.csv')) -> pd.DataFrame:
    output_path = os.path.join(DATA_PATH, 'saramin_preprocessed.csv')
    df = pd.read_csv(input_path)
    print(f"  원본: {len(df):,}개, {len(df.columns)}개 컬럼")
    
    # 1. 날짜 정규화
    print(f"  날짜 정규화")
    df['posting_date_normalized'] = df['posting-date'].apply(normalize_saramin_date)
    df['expiration_date_normalized'] = df['expiration-date'].apply(normalize_saramin_date)

    # 2. 텍스트 전처리
    print(f"  텍스트 전처리")
    cleaned_texts = clean_and_validate_texts(df['title'].tolist(), 'Saramin')
    df['text_cleaned'] = cleaned_texts
    
    df['platform'] = 'saramin'
    result = save_with_deduplication(df, output_path, 'id')
    
    return result


def preprocess_wanted(input_path: str = os.path.join(DATA_PATH, 'wanted.csv')) -> pd.DataFrame:
    output_path = os.path.join(DATA_PATH, 'wanted_preprocessed.csv')
    df = pd.read_csv(input_path)
    print(f"  원본: {len(df):,}개, {len(df.columns)}개 컬럼")
    
    # 1. 날짜 정규화 (검증)
    print(f"  날짜 정규화")
    df['due_time_normalized'] = df['due_time'].apply(normalize_wanted_date)
    
    valid_due_time = df['due_time_normalized'].notna().sum()
    print(f"  유효한 due_time: {valid_due_time:,}개 / {len(df):,}개 ({valid_due_time/len(df)*100:.1f}%)")
    
    # 2. 텍스트 전처리
    print(f"  텍스트 전처리")
    cleaned_texts = clean_and_validate_texts(df['main_tasks'].tolist(), 'Wanted')
    df['text_cleaned'] = cleaned_texts
    
    df['platform'] = 'wanted'
    result = save_with_deduplication(df, output_path, 'job_id')
    
    return result


# ============================================================================
# 임베딩용 데이터 준비 함수들
# ============================================================================

def prepare_work24_for_embedding(work24: pd.DataFrame) -> pd.DataFrame:
    """
    Work24 임베딩용 데이터 준비
    """
    print("\n=== Work24 임베딩 데이터 준비 ===")
    
    work24_data = work24[['구인인증번호', '채용제목', 'text_cleaned', '등록일자_normalized', '마감일자_normalized']].copy()
    work24_data = work24_data.dropna(subset=['text_cleaned'])
    
    # 빈 텍스트 제거
    work24_data = work24_data[work24_data['text_cleaned'] != ''].reset_index(drop=True)
    
    work24_data['platform'] = 'work24'
    work24_data['original_id'] = work24_data['구인인증번호']
    work24_data['text_original'] = work24_data['채용제목']
    work24_data['text'] = work24_data['text_cleaned']
    work24_data['posting_date'] = work24_data['등록일자_normalized']
    work24_data['expiration_date'] = work24_data['마감일자_normalized']
    
    result = work24_data[['original_id', 'platform', 'text_original', 'text', 'posting_date', 'expiration_date']].copy()
    print(f"  Work24 임베딩 준비 완료: {len(result):,}개")
    
    return result


def prepare_saramin_for_embedding(saramin: pd.DataFrame) -> pd.DataFrame:
    """
    Saramin 임베딩용 데이터 준비
    """
    print("\n=== Saramin 임베딩 데이터 준비 ===")
    
    saramin_data = saramin[['id', 'title', 'text_cleaned', 'posting_date_normalized', 'expiration_date_normalized']].copy()
    saramin_data = saramin_data.dropna(subset=['text_cleaned'])
    
    # 빈 텍스트 제거
    saramin_data = saramin_data[saramin_data['text_cleaned'] != ''].reset_index(drop=True)
    
    saramin_data['platform'] = 'saramin'
    saramin_data['original_id'] = saramin_data['id']
    saramin_data['text_original'] = saramin_data['title']
    saramin_data['text'] = saramin_data['text_cleaned']
    saramin_data['posting_date'] = saramin_data['posting_date_normalized']
    saramin_data['expiration_date'] = saramin_data['expiration_date_normalized']
    
    result = saramin_data[['original_id', 'platform', 'text_original', 'text', 'posting_date', 'expiration_date']].copy()
    print(f"  Saramin 임베딩 준비 완료: {len(result):,}개")
    
    return result


def prepare_wanted_for_embedding(wanted: pd.DataFrame) -> pd.DataFrame:
    """
    Wanted 임베딩용 데이터 준비 (due_time이 있는 데이터만)
    """
    print("\n=== Wanted 임베딩 데이터 준비 ===")
    
    wanted_data = wanted[['job_id', 'main_tasks', 'text_cleaned', 'due_time_normalized']].copy()
    # due_time과 text_cleaned가 모두 유효한 것만 필터링
    wanted_data = wanted_data.dropna(subset=['due_time_normalized', 'text_cleaned'])
    
    # 빈 텍스트 제거
    wanted_data = wanted_data[wanted_data['text_cleaned'] != ''].reset_index(drop=True)
    
    wanted_data['platform'] = 'wanted'
    wanted_data['original_id'] = wanted_data['job_id']
    wanted_data['text_original'] = wanted_data['main_tasks']
    wanted_data['text'] = wanted_data['text_cleaned']
    wanted_data['posting_date'] = None  # Wanted는 posting_date 없음
    wanted_data['expiration_date'] = wanted_data['due_time_normalized']
    
    result = wanted_data[['original_id', 'platform', 'text_original', 'text', 'posting_date', 'expiration_date']].copy()
    
    print(f"  Wanted 임베딩 준비 완료: {len(result):,}개")
    print(f"  (due_time이 있고 유효한 텍스트만 포함)")
    
    return result


def prepare_all_data_for_embedding(
    work24_path = 'work24_preprocessed.csv',
    saramin_path = 'saramin_preprocessed.csv',
    wanted_path= 'wanted_preprocessed.csv'
) -> pd.DataFrame:

    # 전처리된 데이터 로드
    work24 = pd.read_csv(work24_path)
    saramin = pd.read_csv(saramin_path)
    wanted = pd.read_csv(wanted_path)
  
    work24_final = prepare_work24_for_embedding(work24)
    saramin_final = prepare_saramin_for_embedding(saramin)
    wanted_final = prepare_wanted_for_embedding(wanted)
    
    all_data = pd.concat([work24_final, saramin_final, wanted_final], ignore_index=True)
    all_data['unique_id'] = all_data['platform'] + '_' + all_data['original_id'].astype(str)
    
    print("\n" + "=" * 70)
    print(f"총 임베딩할 텍스트 수: {len(all_data):,}")
    print("\n플랫폼별 분포:")
    for platform, count in all_data['platform'].value_counts().items():
        print(f"  {platform}: {count:,}개 ({count/len(all_data)*100:.1f}%)")
    
    return all_data


def get_text_statistics(data: pd.DataFrame) -> dict:
    text_lengths = data['text'].str.len()
    
    return {
        'mean': text_lengths.mean(),
        'median': text_lengths.median(),
        'min': text_lengths.min(),
        'max': text_lengths.max(),
        'std': text_lengths.std()
    }

# ============================================================================
# 메인 실행 함수
# ============================================================================

def preprocess_all_platforms():
    work24_df = preprocess_work24()
    saramin_df = preprocess_saramin()
    wanted_df = preprocess_wanted()

    print("\n" + "=" * 70)
    print(f"Work24:  {len(work24_df):,}개 → {os.path.join(DATA_PATH, 'work24_preprocessed.csv')}")
    print(f"Saramin: {len(saramin_df):,}개 → {os.path.join(DATA_PATH, 'saramin_preprocessed.csv')}")
    print(f"Wanted:  {len(wanted_df):,}개 → {os.path.join(DATA_PATH, 'wanted_preprocessed.csv')}")
    print(f"총합:    {len(work24_df) + len(saramin_df) + len(wanted_df):,}개")
    print("=" * 70)
    return work24_df, saramin_df, wanted_df

if __name__ == "__main__":
    preprocess_all_platforms()