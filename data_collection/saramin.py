import html
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Union

import pandas as pd
import requests

API_URL = "https://oapi.saramin.co.kr/job-search"
API_KEY = "hmGyQpGPOJtTMwIUnUWOouxSX8zkVxSZeStMIaujQYErrTldt0W"  
KST = timezone(timedelta(hours=9))


def _extract_text(v: Any) -> str:
    """dict/CDATA/HTML entity까지 텍스트 추출"""
    s = ""
    if isinstance(v, dict):
        for key in ("#text", "text", "name", "value"):
            if key in v and isinstance(v[key], (str, int, float)):
                s = str(v[key])
                break
        else:
            s = str(v)
    elif v is None:
        s = ""
    else:
        s = str(v)
    return html.unescape(s.strip())


def _get_nested(d: Dict[str, Any], path: List[str]) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _to_kst_iso(ts_val: Any) -> str:
    if ts_val in (None, ""):
        return ""
    try:
        ts_int = int(ts_val)
    except (ValueError, TypeError):
        return ""
    return datetime.fromtimestamp(ts_int, tz=KST).isoformat()


def _get_company_name(job: Dict[str, Any]) -> str:
    """company name 추출"""
    v = _get_nested(job, ["company", "name"])
    if v:
        return _extract_text(v)
    v = _get_nested(job, ["company", "detail", "name"])
    if v:
        return _extract_text(v)
    return _extract_text(job.get("company"))


def _parse_jobs(job_items: Union[List[Dict[str, Any]], Dict[str, Any], None]) -> List[Dict[str, str]]:
    """jobs.job 배열을 표 형태로 변환"""
    if job_items is None:
        return []
    if isinstance(job_items, dict):
        job_items = [job_items]
    elif not isinstance(job_items, list):
        job_items = []

    rows: List[Dict[str, str]] = []
    for job in job_items:
        job_id = _extract_text(job.get("id"))
        company_name = _get_company_name(job)
        title = _extract_text(_get_nested(job, ["position", "title"]))

        post_date_str = _extract_text(job.get("posting-date"))
        if not post_date_str:
            post_date_str = _to_kst_iso(job.get("posting-timestamp"))

        exp_date_str = _extract_text(job.get("expiration-date"))
        if not exp_date_str:
            exp_date_str = _to_kst_iso(job.get("expiration-timestamp"))

        loc_code = _extract_text(_get_nested(job, ["position", "location", "code"]))
        loc_name = _extract_text(_get_nested(job, ["position", "location", "name"]))
        
        jobtype_code = _extract_text(_get_nested(job, ["position", "job-type", "code"]))
        jobtype_name = _extract_text(_get_nested(job, ["position", "job-type", "name"]))
        
        industry_code = _extract_text(_get_nested(job, ["position", "industry", "code"]))
        industry_name = _extract_text(_get_nested(job, ["position", "industry", "name"]))
        
        jobmid_code = _extract_text(_get_nested(job, ["position", "job-mid-code", "code"]))
        jobmid_name = _extract_text(_get_nested(job, ["position", "job-mid-code", "name"]))
        
        jobcode_code = _extract_text(_get_nested(job, ["position", "job-code", "code"]))
        jobcode_name = _extract_text(_get_nested(job, ["position", "job-code", "name"]))

        ind_kw = job.get("industry-keyword-code")
        ind_kw_code = _extract_text(ind_kw.get("code")) if isinstance(ind_kw, dict) else ""
        ind_kw_name = _extract_text(ind_kw.get("name")) if isinstance(ind_kw, dict) else ""

        job_kw = job.get("job-code-keyword-code")
        job_kw_code = _extract_text(job_kw.get("code")) if isinstance(job_kw, dict) else ""
        job_kw_name = _extract_text(job_kw.get("name")) if isinstance(job_kw, dict) else ""

        exp_text = _extract_text(_get_nested(job, ["position", "experience-level"]))
        exp_code = _extract_text(_get_nested(job, ["position", "experience-level", "code"]))
        exp_min = _extract_text(_get_nested(job, ["position", "experience-level", "min"]))
        exp_max = _extract_text(_get_nested(job, ["position", "experience-level", "max"]))

        edu_text = _extract_text(_get_nested(job, ["position", "required-education-level"]))
        edu_code = _extract_text(_get_nested(job, ["position", "required-education-level", "code"]))

        keyword_text = _extract_text(job.get("keyword"))
        read_cnt = _extract_text(job.get("read-cnt"))
        apply_cnt = _extract_text(job.get("apply-cnt"))
        salary = job.get("salary")
        salary_text = _extract_text(salary)
        salary_code = _extract_text(salary.get("code")) if isinstance(salary, dict) else ""

        rows.append({
            "id": job_id,
            "company": company_name,
            "title": title,
            "posting-date": post_date_str,
            "expiration-date": exp_date_str,
            
            "location_code": loc_code,
            "location_name": loc_name,
            
            "job_type_code": jobtype_code,
            "job_type_name": jobtype_name,
            
            "industry_code": industry_code,
            "industry_name": industry_name,
            
            "job_mid_code": jobmid_code,
            "job_mid_name": jobmid_name,
            
            "job_code_code": jobcode_code,
            "job_code_name": jobcode_name,
            
            "industry_keyword_code": ind_kw_code,
            "industry_keyword_name": ind_kw_name,
            
            "job_code_keyword_code": job_kw_code,
            "job_code_keyword_name": job_kw_name,
            
            "experience_text": exp_text,
            "experience_code": exp_code,
            "experience_min": exp_min,
            "experience_max": exp_max,
            
            "required_education_text": edu_text,
            "required_education_code": edu_code,
            
            "keyword": keyword_text,
            "read_cnt": read_cnt,
            "apply_cnt": apply_cnt,
            "salary_text": salary_text,
            "salary_code": salary_code,
        })
    return rows


def fetch_jobs_all_fields(limit, output_path="saramin.csv"):
    """
    원하는 개수(limit)만큼 페이지네이션하여 수집하고 CSV에 추가 저장
    """
    headers = {
        "Accept": "application/json",
        "User-Agent": "python-requests/2.x (+https://requests.readthedocs.io)",
    }

    rows: List[Dict[str, str]] = []
    start = 0
    remaining = max(1, int(limit))

    # 기존 파일이 있는지 확인
    file_exists = os.path.exists(output_path)
    if file_exists:
        print(f"기존 파일: {output_path} - 데이터 추가")
    else:
        print(f"새 파일 생성: {output_path}")

    while remaining > 0:
        page_size = min(110, remaining)  # API 최대 110
        params = {
            "access-key": API_KEY,
            "fields": "keyword-code,count,posting-date",
            "start": start,
            "count": page_size,
        }

        try:
            resp = requests.get(API_URL, headers=headers, params=params, timeout=20)
        except requests.RequestException as e:
            print("[Error] 요청 예외:", str(e))
            break

        if resp.status_code != 200:
            snippet = resp.text[:1000].replace("\n", " ")
            print("[Error] HTTP", resp.status_code, ":", snippet)
            break

        try:
            data = resp.json()
        except ValueError:
            print("[Error] JSON 파싱 실패")
            break

        jobs_container = None
        if isinstance(data, dict):
            jobs_container = data.get("jobs")
            if jobs_container is None and "job-search" in data and isinstance(data["job-search"], dict):
                jobs_container = data["job-search"].get("jobs")

        if not isinstance(jobs_container, dict):
            break

        job_items = jobs_container.get("job")
        page_rows = _parse_jobs(job_items)
        if not page_rows:
            break

        for r in page_rows:
            rows.append(r)
            if len(rows) >= limit:
                break

        total_raw = jobs_container.get("total")
        try:
            total = int(total_raw) if total_raw is not None else None
        except Exception:
            total = None

        print(f"데이터: {len(rows)}/{limit}개 수집 완료")

        start += page_size
        remaining = limit - len(rows)

        if total is not None and start >= total:
            break

    if not rows:
        print("수집된 데이터가 없습니다.")
        return None

    df = pd.DataFrame(rows, columns=[
        "id", "company", "title", "posting-date", "expiration-date",
        "location_code", "location_name",
        "job_type_code", "job_type_name",
        "industry_code", "industry_name",
        "job_mid_code", "job_mid_name",
        "job_code_code", "job_code_name",
        "industry_keyword_code", "industry_keyword_name",
        "job_code_keyword_code", "job_code_keyword_name",
        "experience_text", "experience_code", "experience_min", "experience_max",
        "required_education_text", "required_education_code",
        "keyword", "read_cnt", "apply_cnt",
        "salary_text", "salary_code",
    ])

    # 추가 모드로 저장 (header는 파일이 없을 때만)
    df.to_csv(output_path, mode='a', header=not file_exists, index=False, encoding="utf-8-sig")
    
    return df


if __name__ == "__main__":
    # 사람인 API는 일일 호출 횟수가 제한되어 있어, 수집한 데이터를 동일한 파일에 덮어쓰는 방식으로 저장
    df = fetch_jobs_all_fields(limit=2000, output_path="saramin.csv")
    
    if df is not None:
        print(f"\n{len(df)}개 데이터 수집")
        
        # 전체 파일 통계 확인
        full_df = pd.read_csv("saramin.csv", encoding="utf-8-sig")
        print(f"현재 파일 전체 데이터: {len(full_df)}개")
        print(f"고유 공고 수(id 기준): {full_df['id'].nunique()}개")
        print("\n추가된 5개 데이터:")
        print(df.head().to_string(index=False))
    else:
        print("수집 실패")