import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from tqdm import tqdm

# --- 설정 ---
AUTH_KEY = "api_key" 
OUTPUT_CSV_FILE = ".path/Labor-market-trends/datasets/work24_new.csv" # 기존 파일과의 충돌을 방지하기 위해 '_new.csv'로 저장

# API 문서의 startPage 최대값: 1000
TOTAL_PAGES = 1000
DISPLAY_PER_PAGE = 100 # 최대 100

# 대기 시간 (초)
SLEEP_SECONDS = 0.1

def fetch_and_save_all_job_postings():
    """고용24 채용정보 API의 마지막 페이지부터 전체를 수집하여 CSV 파일로 저장합니다."""

    # 1. API 요청을 위한 기본 URL 설정
    api_url = "https://www.work24.go.kr/cm/openApi/call/wk/callOpenApiSvcInfo210L01.do"
    all_job_postings_data = []

    print(f"1. 총 {TOTAL_PAGES}페이지의 채용정보 수집")

    # 2. 마지막 페이지부터 첫 페이지까지 역순으로 반복
    for page in tqdm(range(TOTAL_PAGES, 0, -1), desc="채용정보 수집 중"):
        params = {
            'authKey': AUTH_KEY,
            'callTp': 'L',
            'returnType': 'XML',
            'startPage': str(page),
            'display': str(DISPLAY_PER_PAGE)
        }
        try:
            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()

            root = ET.fromstring(response.content)

            # API 응답에 <total> 태그가 없거나 유효하지 않은 경우를 대비
            total_elements = root.findall('total')
            if total_elements:
                total_count_str = total_elements[0].text
                # total_count_str이 숫자로 변환 가능한지 확인
                if total_count_str and total_count_str.isdigit():
                    total_count = int(total_count_str)
                else:
                    total_count = 0 # 유효하지 않으면 0으로 처리
            else:
                total_count = 0

            for item in root.findall('wanted'):
                job_data = {
                    '구인인증번호': item.findtext('wantedAuthNo'),
                    '회사명': item.findtext('company'),
                    '사업자등록번호': item.findtext('busino'),
                    '업종': item.findtext('indTpNm'),
                    '채용제목': item.findtext('title'),
                    '임금형태': item.findtext('salTpNm'),
                    '급여': item.findtext('sal'),
                    '근무지역': item.findtext('region'),
                    '근무형태': item.findtext('holidayTpNm'),
                    '최소학력': item.findtext('minEdubg'),
                    '경력': item.findtext('career'),
                    '등록일자': item.findtext('regDt'),
                    '마감일자': item.findtext('closeDt'),
                    '정보제공처': item.findtext('infoSvc'),
                    '채용정보URL': item.findtext('wantedInfoUrl'),
                    '근무지_우편주소': item.findtext('zipCd'),
                    '근무지_도로명주소': item.findtext('strtnmCd'),
                    '근무지_기본주소': item.findtext('basicAddr'),
                    '근무지_상세주소': item.findtext('detailAddr'),
                    '고용형태코드': item.findtext('empTpCd'),
                    '직종코드': item.findtext('jobsCd'),
                    '최종수정일': item.findtext('smodifyDtm')
                }
                all_job_postings_data.append(job_data)
            time.sleep(SLEEP_SECONDS)

        except requests.exceptions.RequestException as e:
            print(f"\n네트워크 오류 (페이지 {page}): {e}")
            continue 
        except ET.ParseError as e:
            print(f"\nXML 파싱 오류 (페이지 {page}): {e}")
            continue

    # 3. 데이터프레임 생성 및 CSV 파일 저장
    if not all_job_postings_data:
        print("\n수집된 채용 정보가 없습니다. 인증키나 네트워크 상태를 확인해주세요.")
        return

    print(f"\n\n2. 총 {len(all_job_postings_data)}개의 채용 정보 수집")
    df = pd.DataFrame(all_job_postings_data)
    df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')

    print("-" * 50)
    print("저장된 데이터 샘플 (상위 5개):")
    print(df.head())
    print("-" * 50)


if __name__ == '__main__':
    fetch_and_save_all_job_postings()