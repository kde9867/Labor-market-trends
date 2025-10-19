import pickle
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from preprocessing import get_text_statistics, prepare_all_data_for_embedding


class JobEmbeddingGenerator: 
    def __init__(self, model_path='/model_path/ko-sbert-multitask'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_path)
        self.model = self.model.to(self.device)

    def create_embeddings_batch(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """
        배치 단위로 임베딩 생성
        
        Args:
            texts: 텍스트 리스트
            batch_size: 배치 크기
            
        Returns:
            np.ndarray: 임베딩 배열
        """
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
    
    def save_results(self, 
                    data: pd.DataFrame, 
                    embeddings: np.ndarray,
                    output_prefix='job_embeddings_ko_sbert_preprocessed') -> None:
        """
        임베딩 결과 저장
        
        Args:
            data: 메타데이터 데이터프레임
            embeddings: 임베딩 배열
            output_prefix: 출력 파일명 접두사
        """
        
        # 통계 정보 수집
        stats = get_text_statistics(data)
        
        # 메타데이터와 임베딩을 함께 저장
        result_data = {
            'metadata': data,
            'embeddings': embeddings,
            'embedding_dim': embeddings.shape[1],
            'total_samples': len(data),
            'platform_counts': data['platform'].value_counts().to_dict(),
            'preprocessing_info': {
                'text_length_stats': stats
            }
        }

        # pickle로 전체 데이터 저장
        pkl_filename = f'{output_prefix}.pkl'
        with open(pkl_filename, 'wb') as f:
            pickle.dump(result_data, f)
        print(f"\n✓ Pickle 파일 저장: {pkl_filename}")

        # 임베딩만 numpy 배열로 저장
        npy_filename = f'{output_prefix}_embeddings.npy'
        np.save(npy_filename, embeddings)
        print(f"✓ Numpy 파일 저장: {npy_filename}")
        
        # # 메타데이터만 CSV로 저장
        # csv_filename = f'{output_prefix}_metadata.csv'
        # data.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        # print(f"✓ 메타데이터 저장: {csv_filename}")
    
    def cleanup(self):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()


def load_and_verify_results(pkl_filename: str = 'job_embeddings_ko_sbert_preprocessed.pkl') -> Dict:
    """
    저장된 결과를 로드하고 확인
    
    Args:
        pkl_filename: pickle 파일명
        
    Returns:
        Dict: 로드된 데이터
    """
    
    # pickle 파일 로드
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
    
    print(f"\n✓ 검증 완료: 데이터 정상 로드됨")
    print("=" * 70)
    
    return loaded_data


def main(work24_path='work24_preprocessed.csv',
         saramin_path='saramin_preprocessed.csv',
         wanted_path='wanted_preprocessed.csv',
         model_path='/model_path/ko-sbert-multitask',
         batch_size=32,
         output_prefix='job_embeddings_ko_sbert_preprocessed'):
    """
    메인 실행 함수
    
    Args:
        work24_path: 전처리된 Work24 CSV 경로
        saramin_path: 전처리된 Saramin CSV 경로
        wanted_path: 전처리된 Wanted CSV 경로
        model_path: Ko-SBERT 모델 경로
        batch_size: 임베딩 배치 크기
        output_prefix: 출력 파일명 접두사
    """
    print("\n" + "=" * 70)
    print("임베딩 생성 프로세스 시작")
    print("=" * 70)
    
    # 1. 전처리된 데이터를 임베딩용으로 준비
    data = prepare_all_data_for_embedding(work24_path, saramin_path, wanted_path)
    
    # 2. 임베딩 생성
    print("\n" + "=" * 70)
    print("임베딩 생성 중...")
    print("=" * 70)
    generator = JobEmbeddingGenerator(model_path)
    texts = data['text'].tolist()
    embeddings = generator.create_embeddings_batch(texts, batch_size=batch_size)
    
    # 3. 결과 저장
    print("\n" + "=" * 70)
    print("결과 저장 중...")
    print("=" * 70)
    generator.save_results(data, embeddings, output_prefix)
    
    generator.cleanup()
    
    # 6. 저장된 결과 검증
    load_and_verify_results(f'{output_prefix}.pkl')
    
    return data, embeddings


if __name__ == "__main__":
    # 전처리된 데이터로부터 임베딩 생성
    data, embeddings = main(
        work24_path='/path/Labor-market-trends/datasets/work24_preprocessed.csv',
        saramin_path='/path/Labor-market-trends/datasets/saramin_preprocessed.csv',
        wanted_path='/path/Labor-market-trends/datasets/wanted_preprocessed.csv',
        model_path='/model_path/ko-sbert-multitask',
        batch_size=32,
        output_prefix='/path/Labor-market-trends/job_embeddings_ko_sbert_preprocessed'
    )