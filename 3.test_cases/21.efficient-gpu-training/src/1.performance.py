#!/usr/bin/env python
# coding=utf-8

import os
import time
import yaml
import argparse
from pynvml import *
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from accelerate import Accelerator
from accelerate.utils import set_seed
import pandas as pd
import matplotlib.pyplot as plt
import torch.distributed as dist

# 커스텀 데이터셋 클래스 임포트
from utils.custom_dataset import SequenceClassificationDataset


def print_gpu_utilization():
    """GPU 메모리 사용량 통계 출력"""
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used//1024**2  # MB 단위로 반환


def is_main_process():
    """현재 프로세스가 메인 프로세스(global_rank=0)인지 확인합니다."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def main_process_print(*args, **kwargs):
    """메인 프로세스(global_rank=0)에서만 출력합니다."""
    if is_main_process():
        print(*args, **kwargs)


# Prefetch를 지원하는 DataLoader 래퍼 클래스
class PrefetchDataLoader:
    """
    데이터 프리페치 기능을 지원하는 DataLoader 래퍼 클래스.
    다음 배치를 미리 GPU로 전송하여 I/O와 컴퓨팅을 오버랩합니다.
    """
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.iter = None
        self.next_batch = None
        self.stream = torch.cuda.Stream(device=device)

    def __iter__(self):
        self.iter = iter(self.dataloader)
        self.preload()
        return self

    def preload(self):
        try:
            self.next_batch = next(self.iter)
            with torch.cuda.stream(self.stream):
                for k, v in self.next_batch.items():
                    if isinstance(v, torch.Tensor):
                        self.next_batch[k] = v.to(device=self.device, non_blocking=True)
        except StopIteration:
            self.next_batch = None

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration

        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch

    def __len__(self):
        return len(self.dataloader)


def benchmark_dataloader(dataset, batch_size, num_workers, pin_memory, prefetch_factor, num_epochs, 
                         use_fp16=False, use_prefetch=False):
    """
    특정 DataLoader 설정에 대한 성능을 벤치마킹합니다.
    
    Parameters:
    -----------
    dataset : Dataset
        벤치마킹에 사용할 데이터셋
    batch_size : int
        배치 크기
    num_workers : int
        DataLoader의 worker 수
    pin_memory : bool
        pin_memory 옵션 사용 여부
    prefetch_factor : int
        prefetch_factor 값 (num_workers > 0일 때만 유효)
    num_epochs : int
        학습할 에폭 수
    use_fp16 : bool
        FP16 혼합 정밀도 사용 여부
    use_prefetch : bool
        데이터 프리페치 기능 사용 여부
        
    Returns:
    --------
    dict
        벤치마크 결과 (처리량, 메모리 사용량, 소요 시간 등)
    """
    # DataLoader 설정
    # prefetch_factor는 num_workers가 0보다 클 때만 사용
    dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': True
    }
    
    # num_workers > 0일 때만 prefetch_factor 설정 추가
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
    
    # DataLoader 구성
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    
    # Accelerator 초기화
    accelerator = Accelerator(mixed_precision='fp16' if use_fp16 else 'no')
    
    # 모델 초기화 (더 작은 모델로 벤치마킹 - 더 빠름)
    model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased", hidden_dropout_prob=0.2)
    
    # 옵티마이저 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # 옵티마이저 준비
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # Accelerator prepare 이후에 prefetch 래퍼 적용
    if use_prefetch:
        dataloader = PrefetchDataLoader(dataloader, accelerator.device)
    
    # 초기 GPU 메모리 측정
    initial_memory = print_gpu_utilization()
    
    # 학습 루프 시간 측정
    model.train()
    start_time = time.time()
    processed_samples = 0
    total_batches = min(len(dataloader), 20)  # 20 배치만 테스트
    
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            if step >= total_batches:
                break
                
            # 순전파
            outputs = model(**batch)
            loss = outputs.loss
            
            # 역전파
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            processed_samples += batch_size * accelerator.num_processes
    
    # 종료 시간 및 메모리 측정
    end_time = time.time()
    final_memory = print_gpu_utilization()
    
    # 결과 계산
    elapsed_time = end_time - start_time
    throughput = processed_samples / elapsed_time if elapsed_time > 0 else 0
    
    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "prefetch_factor": prefetch_factor if num_workers > 0 else 0,
        "use_prefetch": use_prefetch,
        "throughput": throughput,
        "initial_memory_mb": initial_memory,
        "final_memory_mb": final_memory,
        "memory_increase_mb": final_memory - initial_memory,
        "elapsed_time": elapsed_time
    }


def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description="DataLoader Performance Benchmark")
    parser.add_argument("--config", type=str, required=True, help="설정 파일 경로")
    parser.add_argument("--output", type=str, default="dataloader_benchmark_results.csv",
                      help="결과를 저장할 CSV 파일 경로")
    parser.add_argument("--local_rank", type=int, default=-1,
                      help="분산 훈련을 위한 로컬 랭크 (일반적으로 자동으로 설정됨)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 설정 파일 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if is_main_process():
        main_process_print(config)
    
    # 재현성을 위한 랜덤 시드 설정
    set_seed(42)
    
    # 전통적인 PyTorch Dataset 클래스를 사용한 데이터셋 생성
    dataset = SequenceClassificationDataset(
        seq_len=config['seq_len'],
        dataset_size=config['dataset_size'],
        seed=42
    )
    
    # 테스트할 worker 수 범위
    worker_counts = [8] if torch.cuda.is_available() else [0, 1, 2]
    
    # pin_memory 옵션
    pin_memory_options = [True]
    
    # prefetch 옵션 추가
    prefetch_options = [False, True]
    
    # prefetch_factor 옵션 추가 (테스트할 prefetch_factor 값들)
    prefetch_factor_options = [2, 4, 8]
    
    # 배치 크기
    batch_size = config['per_device_train_batch_size']

    # num_epochs 설정 (config에서 가져오거나 기본값 사용)
    num_epochs = config.get('num_epochs', 1)
    
    # 결과 저장할 리스트
    results = []
    
    # 모든 설정 조합에 대해 벤치마크 실행
    if is_main_process():
        main_process_print(f"{'=' * 70}")
        main_process_print(f"DataLoader 성능 벤치마크 시작 (num_epochs: {num_epochs}, batch_size: {batch_size})")
        main_process_print(f"테스트 파라미터: workers, pin_memory, prefetch_factor, custom prefetch 구현")
        main_process_print(f"{'=' * 70}")
    
    for num_workers in worker_counts:
        for pin_memory in pin_memory_options:
            for use_prefetch in prefetch_options:
                # num_workers가 0인 경우에는 prefetch_factor를 테스트하지 않음
                if num_workers == 0:
                    # num_workers가 0일 때는 prefetch_factor가 의미 없으므로 기본값 사용
                    if is_main_process():
                        main_process_print(f"\n테스트 설정: workers={num_workers}, pin_memory={pin_memory}, "
                                         f"custom prefetch={use_prefetch}")
                    
                    # 벤치마크 실행
                    result = benchmark_dataloader(
                        dataset=dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        prefetch_factor=2,  # 기본값 사용 (의미 없음)
                        num_epochs=num_epochs,
                        use_fp16=config['fp16'],
                        use_prefetch=use_prefetch
                    )
                    
                    # 결과 출력 - 메인 프로세스에서만
                    if is_main_process():
                        main_process_print(f"  처리량: {result['throughput']:.2f} samples/sec")
                        main_process_print(f"  소요 시간: {result['elapsed_time']:.2f} seconds")
                        main_process_print(f"  메모리 사용량: {result['final_memory_mb']} MB")
                    
                    # 결과 저장 - 메인 프로세스에서만
                    if is_main_process():
                        results.append(result)
                else:
                    # num_workers > 0일 때는 다양한 prefetch_factor 값 테스트
                    for prefetch_factor in prefetch_factor_options:
                        if is_main_process():
                            main_process_print(f"\n테스트 설정: workers={num_workers}, pin_memory={pin_memory}, "
                                             f"prefetch_factor={prefetch_factor}, custom prefetch={use_prefetch}")
                        
                        # 벤치마크 실행
                        result = benchmark_dataloader(
                            dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            prefetch_factor=prefetch_factor,
                            num_epochs=num_epochs,
                            use_fp16=config['fp16'],
                            use_prefetch=use_prefetch
                        )
                        
                        # 결과 출력 - 메인 프로세스에서만
                        if is_main_process():
                            main_process_print(f"  처리량: {result['throughput']:.2f} samples/sec")
                            main_process_print(f"  소요 시간: {result['elapsed_time']:.2f} seconds")
                            main_process_print(f"  메모리 사용량: {result['final_memory_mb']} MB")
                        
                        # 결과 저장 - 메인 프로세스에서만
                        if is_main_process():
                            results.append(result)
    
    # 결과를 DataFrame으로 변환하고 저장 - 메인 프로세스에서만
    if is_main_process():
        df_results = pd.DataFrame(results)
        
        # CSV 파일로 저장
        df_results.to_csv(args.output, index=False)
        main_process_print(f"\n결과가 {args.output}에 저장되었습니다.")
        
        # 결과 시각화 - 다양한 그래프
        create_visualization_plots(df_results, prefetch_factor_options)
        
        # 프리페치 유무에 따른 향상도 분석 및 prefetch_factor에 따른 영향 분석
        analyze_prefetch_improvement(df_results, worker_counts, pin_memory_options, prefetch_factor_options)


def create_visualization_plots(df_results, prefetch_factor_options):
    """
    벤치마크 결과를 다양한 관점에서 시각화합니다.
    
    Parameters:
    -----------
    df_results : DataFrame
        벤치마크 결과가 저장된 DataFrame
    prefetch_factor_options : list
        테스트한 prefetch_factor 값 목록
    """
    # 1. Custom Prefetch 사용 유무에 따른 처리량 비교
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 처리량 그래프 (custom prefetch 없을 때)
    ax = axes[0, 0]
    for pf in prefetch_factor_options:
        data = df_results[(df_results['use_prefetch'] == False) & 
                         (df_results['prefetch_factor'] == pf)]
        if not data.empty:
            ax.plot(data['num_workers'], data['throughput'], 
                   marker='o', label=f'prefetch_factor={pf}')
    
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Throughput (samples/sec)')
    ax.set_title('DataLoader Throughput (without Custom Prefetch)')
    ax.legend()
    ax.grid(True)
    
    # 처리량 그래프 (custom prefetch 있을 때)
    ax = axes[0, 1]
    for pf in prefetch_factor_options:
        data = df_results[(df_results['use_prefetch'] == True) & 
                         (df_results['prefetch_factor'] == pf)]
        if not data.empty:
            ax.plot(data['num_workers'], data['throughput'], 
                   marker='o', label=f'prefetch_factor={pf}')
    
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Throughput (samples/sec)')
    ax.set_title('DataLoader Throughput (with Custom Prefetch)')
    ax.legend()
    ax.grid(True)
    
    # 소요 시간 그래프 (custom prefetch 없을 때)
    ax = axes[1, 0]
    for pf in prefetch_factor_options:
        data = df_results[(df_results['use_prefetch'] == False) & 
                         (df_results['prefetch_factor'] == pf)]
        if not data.empty:
            ax.plot(data['num_workers'], data['elapsed_time'], 
                   marker='o', label=f'prefetch_factor={pf}')
    
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Elapsed Time (seconds)')
    ax.set_title('Training Time (without Custom Prefetch)')
    ax.legend()
    ax.grid(True)
    
    # 소요 시간 그래프 (custom prefetch 있을 때)
    ax = axes[1, 1]
    for pf in prefetch_factor_options:
        data = df_results[(df_results['use_prefetch'] == True) & 
                         (df_results['prefetch_factor'] == pf)]
        if not data.empty:
            ax.plot(data['num_workers'], data['elapsed_time'], 
                   marker='o', label=f'prefetch_factor={pf}')
    
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Elapsed Time (seconds)')
    ax.set_title('Training Time (with Custom Prefetch)')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('dataloader_benchmark_by_prefetch_factor.png')
    main_process_print(f"prefetch_factor에 따른 시각화 결과가 dataloader_benchmark_by_prefetch_factor.png에 저장되었습니다.")
    
    # 2. prefetch_factor에 따른 메모리 사용량 비교
    plt.figure(figsize=(15, 8))
    
    for use_prefetch in [False, True]:
        for pf in prefetch_factor_options:
            data = df_results[(df_results['use_prefetch'] == use_prefetch) & 
                             (df_results['prefetch_factor'] == pf)]
            if not data.empty:
                plt.plot(data['num_workers'], data['memory_increase_mb'], 
                       marker='o', label=f'custom_prefetch={use_prefetch}, prefetch_factor={pf}')
    
    plt.xlabel('Number of Workers')
    plt.ylabel('Memory Increase (MB)')
    plt.title('Memory Usage Increase by prefetch_factor and Custom Prefetch')
    plt.legend()
    plt.grid(True)
    plt.savefig('memory_usage_by_prefetch_settings.png')
    main_process_print(f"메모리 사용량 분석 결과가 memory_usage_by_prefetch_settings.png에 저장되었습니다.")


def analyze_prefetch_improvement(df_results, worker_counts, pin_memory_options, prefetch_factor_options):
    """
    프리페치 관련 설정의 성능 향상도를 분석합니다.
    
    Parameters:
    -----------
    df_results : DataFrame
        벤치마크 결과가 저장된 DataFrame
    worker_counts : list
        테스트한 worker 수 목록
    pin_memory_options : list
        테스트한 pin_memory 옵션 목록
    prefetch_factor_options : list
        테스트한 prefetch_factor 값 목록
    """
    main_process_print(f"\n{'=' * 70}")
    main_process_print(f"프리페치 설정에 따른 성능 분석")
    main_process_print(f"{'=' * 70}")
    
    # 1. Custom Prefetch 사용 효과 분석
    main_process_print(f"\n1. Custom Prefetch 래퍼 사용 효과:")
    custom_prefetch_improvement = []
    
    for num_workers in worker_counts:
        if num_workers == 0:
            continue  # num_workers=0일 때는 prefetch_factor가 의미 없음
            
        for pin_memory in pin_memory_options:
            for prefetch_factor in prefetch_factor_options:
                # 동일 설정에서 custom prefetch 사용 여부만 다른 두 결과 찾기
                no_custom_prefetch = df_results[(df_results['num_workers'] == num_workers) & 
                                             (df_results['pin_memory'] == pin_memory) &
                                             (df_results['prefetch_factor'] == prefetch_factor) &
                                             (df_results['use_prefetch'] == False)]
                
                with_custom_prefetch = df_results[(df_results['num_workers'] == num_workers) & 
                                               (df_results['pin_memory'] == pin_memory) &
                                               (df_results['prefetch_factor'] == prefetch_factor) &
                                               (df_results['use_prefetch'] == True)]
                
                if len(no_custom_prefetch) == 1 and len(with_custom_prefetch) == 1:
                    # 첫 번째 행만 가져옴
                    no_custom_prefetch = no_custom_prefetch.iloc[0]
                    with_custom_prefetch = with_custom_prefetch.iloc[0]
                    
                    # 향상도 계산
                    throughput_increase = ((with_custom_prefetch['throughput'] - no_custom_prefetch['throughput']) / 
                                          no_custom_prefetch['throughput'] * 100)
                    
                    time_decrease = ((no_custom_prefetch['elapsed_time'] - with_custom_prefetch['elapsed_time']) / 
                                    no_custom_prefetch['elapsed_time'] * 100)
                    
                    memory_increase = ((with_custom_prefetch['memory_increase_mb'] - no_custom_prefetch['memory_increase_mb']) /
                                      max(1, no_custom_prefetch['memory_increase_mb']) * 100)
                    
                    main_process_print(f"  workers={num_workers}, pin_memory={pin_memory}, prefetch_factor={prefetch_factor}:")
                    main_process_print(f"    처리량 향상: {throughput_increase:.2f}%")
                    main_process_print(f"    소요 시간 감소: {time_decrease:.2f}%")
                    main_process_print(f"    메모리 사용량 변화: {memory_increase:.2f}%")
                    
                    custom_prefetch_improvement.append({
                        'num_workers': num_workers,
                        'pin_memory': pin_memory,
                        'prefetch_factor': prefetch_factor,
                        'throughput_increase_pct': throughput_increase,
                        'time_decrease_pct': time_decrease,
                        'memory_increase_pct': memory_increase
                    })
    
    # 2. prefetch_factor 영향 분석
    main_process_print(f"\n2. prefetch_factor 값에 따른 영향 분석:")
    prefetch_factor_analysis = []
    
    # 기준이 되는 prefetch_factor (가장 작은 값)
    base_prefetch_factor = min(prefetch_factor_options)
    
    for num_workers in worker_counts:
        if num_workers == 0:
            continue  # num_workers=0일 때는 prefetch_factor가 의미 없음
            
        for pin_memory in pin_memory_options:
            for use_prefetch in [False, True]:
                # 기준 prefetch_factor의 결과
                base_result = df_results[(df_results['num_workers'] == num_workers) & 
                                       (df_results['pin_memory'] == pin_memory) &
                                       (df_results['prefetch_factor'] == base_prefetch_factor) &
                                       (df_results['use_prefetch'] == use_prefetch)]
                
                if len(base_result) == 1:
                    base_result = base_result.iloc[0]
                    
                    main_process_print(f"\n  workers={num_workers}, pin_memory={pin_memory}, custom_prefetch={use_prefetch}:")
                    main_process_print(f"  기준 prefetch_factor={base_prefetch_factor}의 처리량: {base_result['throughput']:.2f} samples/sec")
                    
                    # 다른 prefetch_factor 값과 비교
                    for pf in [f for f in prefetch_factor_options if f != base_prefetch_factor]:
                        compare_result = df_results[(df_results['num_workers'] == num_workers) & 
                                                  (df_results['pin_memory'] == pin_memory) &
                                                  (df_results['prefetch_factor'] == pf) &
                                                  (df_results['use_prefetch'] == use_prefetch)]
                        
                        if len(compare_result) == 1:
                            compare_result = compare_result.iloc[0]
                            
                            # 향상도 계산
                            throughput_change = ((compare_result['throughput'] - base_result['throughput']) / 
                                               base_result['throughput'] * 100)
                            
                            time_change = ((base_result['elapsed_time'] - compare_result['elapsed_time']) / 
                                         base_result['elapsed_time'] * 100)
                            
                            memory_change = ((compare_result['memory_increase_mb'] - base_result['memory_increase_mb']) /
                                           max(1, base_result['memory_increase_mb']) * 100)
                            
                            main_process_print(f"    prefetch_factor={pf} 비교:")
                            main_process_print(f"      처리량 변화: {throughput_change:.2f}%")
                            main_process_print(f"      소요 시간 변화: {time_change:.2f}%")
                            main_process_print(f"      메모리 사용량 변화: {memory_change:.2f}%")
                            
                            prefetch_factor_analysis.append({
                                'num_workers': num_workers,
                                'pin_memory': pin_memory,
                                'use_prefetch': use_prefetch,
                                'base_prefetch_factor': base_prefetch_factor,
                                'compare_prefetch_factor': pf,
                                'throughput_change_pct': throughput_change,
                                'time_change_pct': time_change,
                                'memory_change_pct': memory_change
                            })
    
    # 분석 결과 저장
    if custom_prefetch_improvement:
        df_custom_prefetch = pd.DataFrame(custom_prefetch_improvement)
        df_custom_prefetch.to_csv('custom_prefetch_improvement_analysis.csv', index=False)
        main_process_print(f"\nCustom Prefetch 성능 향상 분석이 custom_prefetch_improvement_analysis.csv에 저장되었습니다.")
    
    if prefetch_factor_analysis:
        df_prefetch_factor = pd.DataFrame(prefetch_factor_analysis)
        df_prefetch_factor.to_csv('prefetch_factor_impact_analysis.csv', index=False)
        main_process_print(f"prefetch_factor 영향 분석이 prefetch_factor_impact_analysis.csv에 저장되었습니다.")


if __name__ == "__main__":
    # 분산 환경 초기화를 보장
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
    main()

# #!/usr/bin/env python
# # coding=utf-8

# import os
# import time
# import yaml
# import argparse
# from pynvml import *
# import torch
# from torch.utils.data import DataLoader
# from transformers import AutoModelForSequenceClassification
# from accelerate import Accelerator
# from accelerate.utils import set_seed
# import pandas as pd
# import matplotlib.pyplot as plt
# import torch.distributed as dist

# # 커스텀 데이터셋 클래스 임포트
# from utils.custom_dataset import SequenceClassificationDataset


# def print_gpu_utilization():
#     """GPU 메모리 사용량 통계 출력"""
#     nvmlInit()
#     handle = nvmlDeviceGetHandleByIndex(0)
#     info = nvmlDeviceGetMemoryInfo(handle)
#     return info.used//1024**2  # MB 단위로 반환


# def is_main_process():
#     """현재 프로세스가 메인 프로세스(global_rank=0)인지 확인합니다."""
#     if dist.is_initialized():
#         return dist.get_rank() == 0
#     return True


# def main_process_print(*args, **kwargs):
#     """메인 프로세스(global_rank=0)에서만 출력합니다."""
#     if is_main_process():
#         print(*args, **kwargs)


# # Prefetch를 지원하는 DataLoader 래퍼 클래스
# class PrefetchDataLoader:
#     """
#     데이터 프리페치 기능을 지원하는 DataLoader 래퍼 클래스.
#     다음 배치를 미리 GPU로 전송하여 I/O와 컴퓨팅을 오버랩합니다.
#     """
#     def __init__(self, dataloader, device):
#         self.dataloader = dataloader
#         self.device = device
#         self.iter = None
#         self.next_batch = None
#         self.stream = torch.cuda.Stream(device=device)

#     def __iter__(self):
#         self.iter = iter(self.dataloader)
#         self.preload()
#         return self

#     def preload(self):
#         try:
#             self.next_batch = next(self.iter)
#             with torch.cuda.stream(self.stream):
#                 for k, v in self.next_batch.items():
#                     if isinstance(v, torch.Tensor):
#                         self.next_batch[k] = v.to(device=self.device, non_blocking=True)
#         except StopIteration:
#             self.next_batch = None

#     def __next__(self):
#         if self.next_batch is None:
#             raise StopIteration

#         torch.cuda.current_stream().wait_stream(self.stream)
#         batch = self.next_batch
#         self.preload()
#         return batch

#     def __len__(self):
#         return len(self.dataloader)


# def benchmark_dataloader(dataset, batch_size, num_workers, pin_memory, num_epochs, use_fp16=False, use_prefetch=False):
#     """
#     특정 DataLoader 설정에 대한 성능을 벤치마킹합니다.
    
#     Parameters:
#     -----------
#     dataset : Dataset
#         벤치마킹에 사용할 데이터셋
#     batch_size : int
#         배치 크기
#     num_workers : int
#         DataLoader의 worker 수
#     pin_memory : bool
#         pin_memory 옵션 사용 여부
#     num_epochs : int
#         학습할 에폭 수
#     use_fp16 : bool
#         FP16 혼합 정밀도 사용 여부
#     use_prefetch : bool
#         데이터 프리페치 기능 사용 여부
        
#     Returns:
#     --------
#     dict
#         벤치마크 결과 (처리량, 메모리 사용량, 소요 시간 등)
#     """
#     # DataLoader 구성
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         drop_last=True
#     )
    
#     # Accelerator 초기화
#     accelerator = Accelerator(mixed_precision='fp16' if use_fp16 else 'no')
    
#     # 모델 초기화 (더 작은 모델로 벤치마킹 - 더 빠름)
#     model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    
#     # 옵티마이저 설정
#     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
#     # Prefetch를 사용하는 경우, prefetch wrapper 적용
#     # 프리페치는 Accelerator prepare 이전에 적용하지 않음 (device 설정 문제 방지)
#     use_prefetch_dataloader = use_prefetch
    
#     # Accelerator로 준비
#     model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
#     # Accelerator prepare 이후에 prefetch 래퍼 적용
#     if use_prefetch_dataloader:
#         dataloader = PrefetchDataLoader(dataloader, accelerator.device)
    
#     # 초기 GPU 메모리 측정
#     initial_memory = print_gpu_utilization()
    
#     # 학습 루프 시간 측정
#     model.train()
#     start_time = time.time()
#     processed_samples = 0
#     total_batches = min(len(dataloader), 20)  # 20 배치만 테스트
    
#     for epoch in range(num_epochs):
#         for step, batch in enumerate(dataloader):
#             if step >= total_batches:
#                 break
                
#             # 순전파
#             outputs = model(**batch)
#             loss = outputs.loss
            
#             # 역전파
#             accelerator.backward(loss)
#             optimizer.step()
#             optimizer.zero_grad()
            
#             processed_samples += batch_size * accelerator.num_processes
    
#     # 종료 시간 및 메모리 측정
#     end_time = time.time()
#     final_memory = print_gpu_utilization()
    
#     # 결과 계산
#     elapsed_time = end_time - start_time
#     throughput = processed_samples / elapsed_time if elapsed_time > 0 else 0
    
#     return {
#         "num_workers": num_workers,
#         "pin_memory": pin_memory,
#         "use_prefetch": use_prefetch,
#         "throughput": throughput,
#         "initial_memory_mb": initial_memory,
#         "final_memory_mb": final_memory,
#         "memory_increase_mb": final_memory - initial_memory,
#         "elapsed_time": elapsed_time
#     }


# def parse_args():
#     """명령줄 인수 파싱"""
#     parser = argparse.ArgumentParser(description="DataLoader Performance Benchmark")
#     parser.add_argument("--config", type=str, required=True, help="설정 파일 경로")
#     parser.add_argument("--output", type=str, default="dataloader_benchmark_results.csv",
#                       help="결과를 저장할 CSV 파일 경로")
#     parser.add_argument("--local_rank", type=int, default=-1,
#                       help="분산 훈련을 위한 로컬 랭크 (일반적으로 자동으로 설정됨)")
#     return parser.parse_args()


# def main():
#     args = parse_args()
    
#     # 설정 파일 로드
#     with open(args.config, 'r') as f:
#         config = yaml.safe_load(f)

#     if is_main_process():
#         main_process_print(config)
    
#     # 재현성을 위한 랜덤 시드 설정
#     set_seed(42)
    
#     # 전통적인 PyTorch Dataset 클래스를 사용한 데이터셋 생성
#     dataset = SequenceClassificationDataset(
#         seq_len=config['seq_len'],
#         dataset_size=config['dataset_size'],
#         seed=42
#     )
    
#     # 테스트할 worker 수 범위
#     worker_counts = [8] if torch.cuda.is_available() else [0, 1, 2]
    
#     # pin_memory 옵션
#     pin_memory_options = [False, True]
    
#     # prefetch 옵션 추가
#     prefetch_options = [False, True]
    
#     # 배치 크기
#     batch_size = config['per_device_train_batch_size']

#     # num_epochs 설정 (config에서 가져오거나 기본값 사용)
#     num_epochs = config.get('num_epochs', 1)
    
#     # 결과 저장할 리스트
#     results = []
    
#     # 모든 설정 조합에 대해 벤치마크 실행
#     if is_main_process():
#         main_process_print(f"{'=' * 60}")
#         main_process_print(f"DataLoader 성능 벤치마크 시작 (num_epochs: {num_epochs})")
#         main_process_print(f"DataLoader 성능 벤치마크 시작 (batch_size: {batch_size})")
#         main_process_print(f"{'=' * 60}")
    
#     for num_workers in worker_counts:
#         for pin_memory in pin_memory_options:
#             for use_prefetch in prefetch_options:
#                 if is_main_process():
#                     main_process_print(f"\n테스트 설정: workers={num_workers}, pin_memory={pin_memory}, prefetch={use_prefetch}")
                
#                 # 벤치마크 실행
#                 result = benchmark_dataloader(
#                     dataset=dataset,
#                     batch_size=batch_size,
#                     num_workers=num_workers,
#                     pin_memory=pin_memory,
#                     num_epochs=num_epochs,
#                     use_fp16=config['fp16'],
#                     use_prefetch=use_prefetch
#                 )
                
#                 # 결과 출력 - 메인 프로세스에서만
#                 if is_main_process():
#                     main_process_print(f"  처리량: {result['throughput']:.2f} samples/sec")
#                     main_process_print(f"  소요 시간: {result['elapsed_time']:.2f} seconds")
#                     main_process_print(f"  메모리 사용량: {result['final_memory_mb']} MB")
                
#                 # 결과 저장 - 메인 프로세스에서만
#                 if is_main_process():
#                     results.append(result)
    
#     # 결과를 DataFrame으로 변환하고 저장 - 메인 프로세스에서만
#     if is_main_process():
#         df_results = pd.DataFrame(results)
        
#         # CSV 파일로 저장
#         df_results.to_csv(args.output, index=False)
#         main_process_print(f"\n결과가 {args.output}에 저장되었습니다.")
        
#         # 결과 시각화 - 처리량 그래프
#         fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
#         # 1. 처리량 그래프 (prefetch 없을 때)
#         ax = axes[0, 0]
#         for pin_memory in pin_memory_options:
#             data = df_results[(df_results['pin_memory'] == pin_memory) & (df_results['use_prefetch'] == False)]
#             ax.plot(data['num_workers'], data['throughput'], 
#                    marker='o', label=f'pin_memory={pin_memory}')
        
#         ax.set_xlabel('Number of Workers')
#         ax.set_ylabel('Throughput (samples/sec)')
#         ax.set_title('DataLoader Throughput (without Prefetch)')
#         ax.legend()
#         ax.grid(True)
        
#         # 2. 처리량 그래프 (prefetch 있을 때)
#         ax = axes[0, 1]
#         for pin_memory in pin_memory_options:
#             data = df_results[(df_results['pin_memory'] == pin_memory) & (df_results['use_prefetch'] == True)]
#             ax.plot(data['num_workers'], data['throughput'], 
#                    marker='o', label=f'pin_memory={pin_memory}')
        
#         ax.set_xlabel('Number of Workers')
#         ax.set_ylabel('Throughput (samples/sec)')
#         ax.set_title('DataLoader Throughput (with Prefetch)')
#         ax.legend()
#         ax.grid(True)
        
#         # 3. 소요 시간 그래프 (prefetch 없을 때)
#         ax = axes[1, 0]
#         for pin_memory in pin_memory_options:
#             data = df_results[(df_results['pin_memory'] == pin_memory) & (df_results['use_prefetch'] == False)]
#             ax.plot(data['num_workers'], data['elapsed_time'], 
#                    marker='o', label=f'pin_memory={pin_memory}')
        
#         ax.set_xlabel('Number of Workers')
#         ax.set_ylabel('Elapsed Time (seconds)')
#         ax.set_title('Training Time (without Prefetch)')
#         ax.legend()
#         ax.grid(True)
        
#         # 4. 소요 시간 그래프 (prefetch 있을 때)
#         ax = axes[1, 1]
#         for pin_memory in pin_memory_options:
#             data = df_results[(df_results['pin_memory'] == pin_memory) & (df_results['use_prefetch'] == True)]
#             ax.plot(data['num_workers'], data['elapsed_time'], 
#                    marker='o', label=f'pin_memory={pin_memory}')
        
#         ax.set_xlabel('Number of Workers')
#         ax.set_ylabel('Elapsed Time (seconds)')
#         ax.set_title('Training Time (with Prefetch)')
#         ax.legend()
#         ax.grid(True)
        
#         plt.tight_layout()
#         plt.savefig('dataloader_benchmark_results.png')
#         main_process_print(f"시각화 결과가 dataloader_benchmark_results.png에 저장되었습니다.")
        
#         # 프리페치 유무에 따른 향상도 분석
#         main_process_print(f"\n{'=' * 60}")
#         main_process_print(f"프리페치 사용 효과 분석")
#         main_process_print(f"{'=' * 60}")
        
#         improvement_data = []
        
#         for num_workers in worker_counts:
#             for pin_memory in pin_memory_options:
#                 # 동일한 구성에서 프리페치 사용 여부만 다른 두 결과 찾기
#                 no_prefetch = df_results[(df_results['num_workers'] == num_workers) & 
#                                        (df_results['pin_memory'] == pin_memory) & 
#                                        (df_results['use_prefetch'] == False)]
                
#                 with_prefetch = df_results[(df_results['num_workers'] == num_workers) & 
#                                           (df_results['pin_memory'] == pin_memory) & 
#                                           (df_results['use_prefetch'] == True)]
                
#                 if len(no_prefetch) == 1 and len(with_prefetch) == 1:
#                     # 첫 번째 행만 가져옴
#                     no_prefetch = no_prefetch.iloc[0]
#                     with_prefetch = with_prefetch.iloc[0]
                    
#                     # 향상도 계산
#                     throughput_increase = ((with_prefetch['throughput'] - no_prefetch['throughput']) / 
#                                           no_prefetch['throughput'] * 100)
                    
#                     time_decrease = ((no_prefetch['elapsed_time'] - with_prefetch['elapsed_time']) / 
#                                     no_prefetch['elapsed_time'] * 100)
                    
#                     main_process_print(f"workers={num_workers}, pin_memory={pin_memory}:")
#                     main_process_print(f"  처리량 향상: {throughput_increase:.2f}%")
#                     main_process_print(f"  소요 시간 감소: {time_decrease:.2f}%")
                    
#                     improvement_data.append({
#                         'num_workers': num_workers,
#                         'pin_memory': pin_memory,
#                         'throughput_increase_pct': throughput_increase,
#                         'time_decrease_pct': time_decrease
#                     })
        
#         # 향상도 데이터 저장
#         if improvement_data:
#             df_improvement = pd.DataFrame(improvement_data)
#             df_improvement.to_csv('prefetch_improvement_analysis.csv', index=False)
#             main_process_print(f"\n프리페치 향상도 분석이 prefetch_improvement_analysis.csv에 저장되었습니다.")


# if __name__ == "__main__":
#     # 분산 환경 초기화를 보장
#     if "WORLD_SIZE" in os.environ:
#         dist.init_process_group(backend="nccl")
#     main()

