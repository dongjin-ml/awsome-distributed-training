# import torch
# from torch.utils.data import Dataset
# import numpy as np
# import time

# class SequenceClassificationDataset(Dataset):
#     """
#     CPU 바운드 작업이 강화된 시퀀스 분류 데이터셋
#     """
#     def __init__(self, seq_len=512, dataset_size=1024, seed=42, heavy_transform=True):
#         """
#         Parameters:
#         -----------
#         seq_len : int
#             시퀀스의 길이
#         dataset_size : int
#             데이터셋 샘플 수
#         seed : int
#             난수 생성을 위한 시드값
#         heavy_transform : bool
#             무거운 CPU 작업 활성화 여부
#         """
#         # 재현성을 위한 난수 시드 설정
#         np.random.seed(seed)
        
#         # 가상 데이터 생성
#         self.input_ids = np.random.randint(100, 30000, (dataset_size, seq_len))
#         self.attention_mask = np.random.randint(0, 2, (dataset_size, seq_len))
#         self.labels = np.random.randint(0, 2, (dataset_size))
        
#         # 설정 저장
#         self.dataset_size = dataset_size
#         self.heavy_transform = heavy_transform
#         self.seq_len = seq_len
    
#     def __len__(self):
#         """데이터셋의 총 샘플 수를 반환"""
#         return self.dataset_size
    
#     def __getitem__(self, idx):
#         """
#         주어진 인덱스의 샘플을 가져오고 CPU 바운드 작업 시뮬레이션
#         """
#         # 원본 데이터 복사
#         input_ids_copy = self.input_ids[idx].copy()
#         attention_mask_copy = self.attention_mask[idx].copy()
        
#         # 무거운 CPU 작업 시뮬레이션 (GPU에 영향 없음)
#         if self.heavy_transform:
#             # 1. 순수 CPU 연산 (배열 변환)
#             for _ in range(30):
#                 # 정렬 작업 - CPU 집약적이지만 단순
#                 sorted_array = np.sort(input_ids_copy)
#                 # 역정렬
#                 reversed_array = sorted_array[::-1]
#                 # 마스킹 및 변환
#                 masked = reversed_array * attention_mask_copy
#                 # 누적합
#                 cumsum = np.cumsum(masked)
                
#                 # 배열의 크기가 충분히 큰지 확인 후 reshape
#                 # 나누어 떨어지는 크기로 reshape 해야 함
#                 chunk_size = 16
#                 num_chunks = self.seq_len // chunk_size
#                 if num_chunks > 0:
#                     # reshape가 가능한 부분만 사용
#                     reshape_size = num_chunks * chunk_size
#                     reshaped = cumsum[:reshape_size].reshape(num_chunks, chunk_size)
#                     medians = np.median(reshaped, axis=1)
#                     # 중간값을 원본 크기에 맞게 반복
#                     full_medians = np.repeat(medians, chunk_size)
#                     # 나머지 부분 처리 (reshape에 포함되지 않은 부분)
#                     if reshape_size < self.seq_len:
#                         remaining = np.median(cumsum[reshape_size:]) * np.ones(self.seq_len - reshape_size)
#                         full_medians = np.concatenate([full_medians, remaining])
#                 else:
#                     # 배열이 너무 작은 경우, 단일 중간값 사용
#                     full_medians = np.median(cumsum) * np.ones_like(input_ids_copy)
                
#                 # 결과 다시 적용 (크기가 일치하는지 확인)
#                 input_ids_copy = (input_ids_copy + full_medians.astype(int)) % 30000
            
#             # 2. 추가 CPU 작업 - 더 많은 연산 수행
#             for _ in range(5):
#                 # 행렬 연산 시뮬레이션
#                 random_weights = np.random.rand(self.seq_len, self.seq_len)
#                 # 행렬곱 - CPU 집약적 연산
#                 result = np.dot(input_ids_copy, random_weights)
#                 # 소프트맥스 유사 연산
#                 exp_values = np.exp(result - np.max(result))
#                 softmax_result = exp_values / np.sum(exp_values)
#                 # 결과 다시 스케일링하여 적용
#                 input_ids_copy = (input_ids_copy + (softmax_result * 1000).astype(int)) % 30000
            
#             # 3. 인위적인 지연 추가 (CPU 시간만 소모)
#             time.sleep(0.005)  # 5ms 지연
        
#         # NumPy 배열을 PyTorch 텐서로 변환
#         input_ids_tensor = torch.tensor(input_ids_copy, dtype=torch.long)
#         attention_mask_tensor = torch.tensor(attention_mask_copy, dtype=torch.long)
#         label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
#         # 딕셔너리 형태로 반환
#         return {
#             'input_ids': input_ids_tensor,
#             'attention_mask': attention_mask_tensor,
#             'labels': label_tensor
#         }

import torch
from torch.utils.data import Dataset
import numpy as np
import time
from scipy import signal
from sklearn.preprocessing import PolynomialFeatures

class SequenceClassificationDataset(Dataset):
    """
    매우 무거운 CPU 계산이 포함된 시퀀스 분류 데이터셋
    """
    def __init__(self, seq_len=512, dataset_size=512, seed=42, heavy_transform=False):
        """
        Parameters:
        -----------
        seq_len : int
            시퀀스의 길이
        dataset_size : int
            데이터셋 샘플 수
        seed : int
            난수 생성을 위한 시드값
        heavy_transform : bool
            극도로 무거운 CPU 작업 활성화 여부
        """
        # 재현성을 위한 난수 시드 설정
        np.random.seed(seed)
        
        # 가상 데이터 생성
        self.input_ids = np.random.randint(100, 30000, (dataset_size, seq_len))
        self.attention_mask = np.random.randint(0, 2, (dataset_size, seq_len))
        self.labels = np.random.randint(0, 2, (dataset_size))
        
        # 미리 계산된 무거운 작업용 데이터
        if heavy_transform:
            # 각 샘플당 대용량 행렬 저장 (메모리 사용량 증가)
            matrix_size = min(2048, seq_len * 4)  # 더 큰 행렬 사용
            self.large_matrices = np.random.randn(dataset_size, matrix_size, matrix_size // 8)
        
        # 설정 저장
        self.dataset_size = dataset_size
        self.heavy_transform = heavy_transform
        self.seq_len = seq_len
    
    def __len__(self):
        """데이터셋의 총 샘플 수를 반환"""
        return self.dataset_size
    
    def __getitem__(self, idx):
        """
        주어진 인덱스의 샘플을 가져오고 극도로 무거운 CPU 작업 수행
        """
        # 원본 데이터 복사
        input_ids_copy = self.input_ids[idx].copy()
        attention_mask_copy = self.attention_mask[idx].copy()
        
        # 극도로 무거운 CPU 작업 시뮬레이션
        if self.heavy_transform:
            # 1. 다차원 행렬 연산 (매우 CPU 집약적)
            matrix = self.large_matrices[idx].copy()
            
            # SVD 계산 (매우 무거운 CPU 연산)
            try:
                U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
                # 특이값 조작
                s_modified = np.log1p(s) * 10
                # 재구성
                reconstructed = U @ np.diag(s_modified) @ Vh
                # 결과 요약
                feature_vector = np.mean(reconstructed, axis=1)[:self.seq_len]
                # 결과 적용
                input_ids_copy = (input_ids_copy + feature_vector.astype(int)) % 30000
                
            except:
                # SVD가 실패할 경우 대체 무거운 연산
                for _ in range(3):
                    # 행렬곱 체인 (매우 CPU 집약적)
                    result = matrix.copy()
                    for i in range(5):
                        result = result @ matrix[:, :result.shape[1]]
                    feature_vector = np.sum(result, axis=1)[:self.seq_len]
                    input_ids_copy = (input_ids_copy + feature_vector.astype(int)) % 30000

            
            # 2. 다항식 특성 변환 (scikit-learn, 매우 무거운 연산)
            try:
                # 입력 데이터를 2D 형태로 변환
                X = input_ids_copy.reshape(-1, 1)
                # 다항식 특성 생성 (차수가 높을수록 계산량 증가)
                poly = PolynomialFeatures(degree=5, include_bias=False)
                X_poly = poly.fit_transform(X)
                # 결과 요약 및 적용
                poly_features = np.mean(X_poly, axis=0)[:self.seq_len]
                input_ids_copy = (input_ids_copy + poly_features.astype(int)) % 30000
            except:
                # 실패할 경우 수동으로 다항식 계산
                x = input_ids_copy.astype(float)
                poly_result = x + x**2/2 + x**3/6 + x**4/24 + x**5/120
                input_ids_copy = (input_ids_copy + poly_result.astype(int)) % 30000
            
            # 3. 신호 처리 연산 (매우 CPU 집약적)
            for _ in range(2):
                # 다양한 필터 적용
                # 저역 통과 필터
                b, a = signal.butter(8, 0.125)
                filtered1 = signal.filtfilt(b, a, input_ids_copy.astype(float))
                
                # 고역 통과 필터
                b, a = signal.butter(8, 0.75, 'high')
                filtered2 = signal.filtfilt(b, a, input_ids_copy.astype(float))
                
                # 밴드 패스 필터
                b, a = signal.butter(8, [0.25, 0.75], 'band')
                filtered3 = signal.filtfilt(b, a, input_ids_copy.astype(float))
                
                # 결과 조합 및 적용
                combined = (filtered1 + filtered2 + filtered3) / 3
                input_ids_copy = (input_ids_copy + combined.astype(int)) % 30000
            
            # 4. 푸리에 변환 및 역변환 (CPU 집약적)
            # 실수 데이터로 변환
            real_data = input_ids_copy.astype(float)
            # FFT 계산
            fft_result = np.fft.fft(real_data)
            # 주파수 영역에서 연산
            modified_spectrum = fft_result * np.exp(1j * np.angle(fft_result))
            # 역 FFT
            ifft_result = np.fft.ifft(modified_spectrum)
            # 결과 적용
            input_ids_copy = (input_ids_copy + np.abs(ifft_result).astype(int)) % 30000
            
            # 5. 인위적인 지연 추가 (필요한 경우)
            time.sleep(0.01)  # 10ms 지연
            
            # 6. 최종 무거운 계산: 행렬 분해와 재구성
            data_matrix = np.outer(input_ids_copy, attention_mask_copy)
            for i in range(3):
                # QR 분해 (매우 CPU 집약적)
                q, r = np.linalg.qr(data_matrix)
                # 재구성 및 새로운 행렬 생성
                data_matrix = q @ r
                # 대각합 추출
                diag_sum = np.diag(data_matrix)[:self.seq_len]
                # 결과 적용
                input_ids_copy = (input_ids_copy + diag_sum.astype(int)) % 30000
        
        # NumPy 배열을 PyTorch 텐서로 변환
        input_ids_tensor = torch.tensor(input_ids_copy, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask_copy, dtype=torch.long)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # 딕셔너리 형태로 반환
        return {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor,
            'labels': label_tensor
        }