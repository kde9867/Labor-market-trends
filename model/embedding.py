import os
import pickle
import sys
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_PATH = '.path/Labor-market-trends/datasets/'
sys.path.append('.path/Labor-market-trends/data')
from preprocessing import get_text_statistics, prepare_all_data_for_embedding


class JobEmbeddingGenerator: 
    def __init__(self, model_path='.model_path/ko-sbert-multitask'):
        """
        SentenceTransformer 임베딩 모델 초기화
        
        기본 설정:
            model_path='.model_path/ko-sbert-multitask'
            → 로컬에 저장된 Ko-SBERT 멀티태스크 모델을 불러옴

        환경에 따라 수정 가능:
            1. 로컬에 모델이 없는 경우:
                model_path="jhgan/ko-sbert-multitask"
                → Hugging Face Hub에서 직접 다운로드하여 사용 가능

            2. 다른 모델을 사용하고 싶다면:
                model_path=".path/model_name"  # 등 경로 수정 하여 사용 가능
        """
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_path)
        self.model = self.model.to(self.device)

    def create_embeddings_batch(self, texts: list, batch_size: int = 32) -> np.ndarray:
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch_texts = texts[i:i+batch_size]

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False,
                normalize_embeddings=True  
            )
            
            embeddings.append(batch_embeddings.cpu().numpy())
        
        all_embeddings = np.vstack(embeddings)
        print(f"Embedding shape: {all_embeddings.shape}")
        return all_embeddings
    
    def save_results(self, data: pd.DataFrame, embeddings: np.ndarray, output_prefix='job_embeddings_ko_sbert_preprocessed') -> None:
        stats = get_text_statistics(data)
        result_data = {
            'metadata': data,
            'embeddings': embeddings,
            'embedding_dim': embeddings.shape[1],
            'total_samples': len(data),
            'platform_counts': data['platform'].value_counts().to_dict(),
            'preprocessing_info': {'text_length_stats': stats}
        }

        # 저장 경로 통일
        pkl_filename = os.path.join(DATA_PATH, f'{output_prefix}.pkl')
        npy_filename = os.path.join(DATA_PATH, f'{output_prefix}_embeddings.npy')

        with open(pkl_filename, 'wb') as f:
            pickle.dump(result_data, f)
        np.save(npy_filename, embeddings)

        print(f"\n Pickle 파일 저장: {pkl_filename}")
        print(f" Numpy 파일 저장: {npy_filename}")
        
        # # 메타데이터만 CSV로 저장
        # csv_filename = f'{output_prefix}_metadata.csv'
        # data.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        # print(f"✓ 메타데이터 저장: {csv_filename}")
    
    def cleanup(self):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()


def load_and_verify_results(pkl_filename: str = os.path.join(DATA_PATH, 'job_embeddings_ko_sbert_preprocessed.pkl')) -> Dict:
    with open(pkl_filename, 'rb') as f:
        loaded_data = pickle.load(f)
    metadata = loaded_data['metadata']
    embeddings = loaded_data['embeddings']
    
    print(f"\n로드된 데이터:")
    print(f"  메타데이터 shape: {metadata.shape}")
    print(f"  임베딩 shape: {embeddings.shape}")
    print(f"  임베딩 차원: {loaded_data['embedding_dim']}")
    print(f"  총 샘플 수: {loaded_data['total_samples']:,}개")
    
    print(f"\n플랫폼별 분포:")
    for platform, count in loaded_data['platform_counts'].items():
        print(f"  {platform}: {count:,}개")
    
    return loaded_data


def main(model_path='.model_path/ko-sbert-multitask', batch_size=32, output_prefix='job_embeddings_ko_sbert_preprocessed'):
    print("\n" + "=" * 70)
    print("임베딩 생성 프로세스 시작")
    print("=" * 70)

    # 1. 전처리된 데이터 준비
    work24_path = os.path.join(DATA_PATH, 'work24_preprocessed.csv')
    saramin_path = os.path.join(DATA_PATH, 'saramin_preprocessed.csv')
    wanted_path = os.path.join(DATA_PATH, 'wanted_preprocessed.csv')

    data = prepare_all_data_for_embedding(work24_path, saramin_path, wanted_path)

    # 2. 임베딩 생성
    generator = JobEmbeddingGenerator(model_path)
    texts = data['text'].tolist()
    embeddings = generator.create_embeddings_batch(texts, batch_size=batch_size)

    # 3. 결과 저장
    generator.save_results(data, embeddings, output_prefix)
    generator.cleanup()

    # 4. 결과 검증
    load_and_verify_results(os.path.join(DATA_PATH, f'{output_prefix}.pkl'))
    return data, embeddings

if __name__ == "__main__":
    data, embeddings = main()