import csv
import os
import time

import pandas as pd
import requests
from tqdm import tqdm

# --- Configuration ---
CLIENT_ID = "CLIENT_ID"
CLIENT_SECRET = "CLIENT_SECRET"
BASE_URL = "https://openapi.wanted.jobs/v1/jobs/"

START_JOB_ID = 2000
END_JOB_ID = 310787
OUTPUT_FILENAME = "wanted.csv"
SAVE_INTERVAL = 100

headers = {
    'accept': 'application/json',
    'wanted-client-id': CLIENT_ID,
    'wanted-client-secret': CLIENT_SECRET,
}

# --- Data Collection ---
all_jobs_data = []
success_count = 0
error_count = 0
error_jobs = []

print(f"Starting detailed job posting collection (Job ID: {START_JOB_ID} to {END_JOB_ID})")

for job_id in tqdm(range(START_JOB_ID, END_JOB_ID + 1), desc="Collection Progress"):
    try:
        url = f"{BASE_URL}{job_id}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            job_data = response.json()
            
            detail = job_data.get('detail') or {}
            company = job_data.get('company') or {}
            address = job_data.get('address') or {}
            category_tags = job_data.get('category_tags') or {}
            
            parent_tag = category_tags.get('parent_tag')
            
            child_tags_list = [tag.get('title') for tag in category_tags.get('child_tags', []) if tag.get('title')]
            skill_tags_list = [tag.get('title') for tag in job_data.get('skill_tags', []) if tag.get('title')]

            def clean_text(text):
                if text is None:
                    return None
                return str(text).replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

            job_info = {
                'job_id': job_data.get('id'),
                'status': job_data.get('status'),
                'due_time': job_data.get('due_time'),
                'company_name': clean_text(company.get('name')),
                'position': clean_text(detail.get('name')),
                'intro': clean_text(detail.get('intro')),
                'main_tasks': clean_text(detail.get('main_tasks')),
                'requirements': clean_text(detail.get('requirements')),
                'preferred_points': clean_text(detail.get('preferred_points')),
                'benefits': clean_text(detail.get('benefits')),
                'parent_category': clean_text(parent_tag.get('title')) if parent_tag else None,
                'child_categories': ", ".join(child_tags_list),
                'skill_tags': ", ".join(skill_tags_list),
                'registration_number': clean_text(company.get('registration_number')),
                'full_address': clean_text(address.get('full_location')),
                'country': clean_text(address.get('country')),
                'country_code': clean_text(address.get('country_code')),
                'location': clean_text(address.get('location')),
                'full_location': clean_text(address.get('full_location')),
                'experience_years_from': job_data.get('annual_from'),
                'experience_years_to': job_data.get('annual_to'),
                'url': job_data.get('url'),
            }

            all_jobs_data.append(job_info)
            success_count += 1

            if success_count > 0 and success_count % SAVE_INTERVAL == 0:
                temp_df = pd.DataFrame(all_jobs_data)
                temp_df.to_csv(
                    OUTPUT_FILENAME,
                    index=False,
                    encoding='utf-8-sig',
                    quoting=csv.QUOTE_ALL,
                    escapechar='\\'
                )
                tqdm.write(f"✅ {success_count} posts collected. Backed up data to '{OUTPUT_FILENAME}'.")

        elif response.status_code == 404:
            # Job doesn't exist, skip silently
            pass
        elif response.status_code == 400:
            # Bad request, skip silently (400 오류 출력 안 함)
            pass
        else:
            # 400과 404 외의 다른 오류만 출력
            tqdm.write(f"⚠️ Job ID {job_id}: HTTP {response.status_code}")

        time.sleep(0.1)

    except requests.exceptions.RequestException as e:
        error_count += 1
        error_jobs.append(job_id)
        tqdm.write(f"❌ Network error on Job ID {job_id}: {e}")
        time.sleep(5)
        
    except Exception as e:
        error_count += 1
        error_jobs.append(job_id)
        tqdm.write(f"❌ Unexpected error on Job ID {job_id}: {e}")
        if all_jobs_data and len(all_jobs_data) > 0:
            try:
                temp_df = pd.DataFrame(all_jobs_data)
                temp_df.to_csv(
                    f"backup_{OUTPUT_FILENAME}",
                    index=False,
                    encoding='utf-8-sig',
                    quoting=csv.QUOTE_ALL,
                    escapechar='\\'
                )
                tqdm.write(f"💾 Emergency backup saved to 'backup_{OUTPUT_FILENAME}'")
            except:
                pass

# --- Final Data Save ---
if all_jobs_data:
    print(f"\n{'='*50}")
    print(f"Final data collection complete.")
    print(f"✅ Successfully collected: {success_count} jobs")
    print(f"❌ Errors encountered: {error_count} jobs")
    
    if error_jobs:
        print(f"\nFailed Job IDs (first 10): {error_jobs[:10]}")
    
    print(f"\nSaving to CSV...")
    final_df = pd.DataFrame(all_jobs_data)
    final_df.to_csv(
        OUTPUT_FILENAME,
        index=False,
        encoding='utf-8-sig',
        quoting=csv.QUOTE_ALL,
        escapechar='\\'
    )
    print(f"✅ Successfully saved all data to '{OUTPUT_FILENAME}'.")
    print(f"Total rows saved: {len(final_df)}")
    print(f"{'='*50}")
else:
    print("\n⚠️ No new job postings were collected.")

if error_jobs:
    with open('error_jobs.txt', 'w') as f:
        for job_id in error_jobs:
            f.write(f"{job_id}\n")
    print(f"Error job IDs saved to 'error_jobs.txt' for retry")