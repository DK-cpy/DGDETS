import os
import sys
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

from experiments import (
    exp_with_fixed_seed_in_nb201,
    exp_with_rand_seed_in_nb201,
    exp_with_fixed_seed_in_meta_predictor,
    exp_with_rand_seed_in_meta_predictor,
)

# ================== 缓存清理 ==================
def clear_fitness_caches(dataset: str, seeds: list):
    """删除适应度缓存文件"""
    cache_dir = "/results/meta/fitness_cache"
    if not os.path.exists(cache_dir):
        return
    for seed in seeds:
        cache_file = os.path.join(cache_dir, f"fitness_{dataset}_{seed}.pth")
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"[Cache Clean] Removed fitness cache: {cache_file}")

def clear_search_log(dataset: str):
    """删除搜索日志文件，确保所有种子重新运行"""
    log_path = f"results/search_log/{dataset}_search.pth"
    if os.path.exists(log_path):
        os.remove(log_path)
        print(f"[Cache Clean] Removed search log: {log_path}")
    else:
        print(f"[Cache Clean] Search log {log_path} not found, skip")

# ================== 主函数 ==================
def main(args):
    dataset_map = {
        'cifar10': 'cifar10',
        'cifar100': 'cifar100',
        'imagenet': 'ImageNet16-120',
        'aircraft': 'aircraft',
        'pets': 'pets',
    }
    dataset = dataset_map[args.dataset.lower()]
    seeds = [
        2345, 333, 777, 1234, 9012, 5678, 111, 222, 444, 555,
        666, 3456, 4567, 6789, 7890, 8901, 1001, 2002, 3003,
        4004, 5005, 6006, 7007, 8008, 9009, 42, 78, 63
    ]

    # 1. 清除适应度缓存 (fitness cache)
    clear_fitness_caches(dataset, seeds)
    # 2. 清除搜索日志 (search log)
    clear_search_log(dataset)

    exp_type = args.exp_type
    if exp_type == 'reproduce' or exp_type == 'meta_fixed':
        exp_with_fixed_seed_in_meta_predictor(dataset=dataset)
    elif exp_type == 'random' or exp_type == 'meta_rand':
        exp_with_rand_seed_in_meta_predictor(dataset=dataset)
    elif exp_type == 'nb201_fixed':
        exp_with_fixed_seed_in_nb201(dataset=dataset)
    elif exp_type == 'nb201_rand':
        exp_with_rand_seed_in_nb201(dataset=dataset)
    else:
        raise ValueError(f"Unknown exp_type: {exp_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--exp_type', default='meta_fixed')
    args = parser.parse_args()

    # 日志记录
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"20-0.5-0.02-CDENAG_exp_{args.dataset}_{args.exp_type}_{time.strftime('%Y%m%d_%H%M%S')}.txt")

    class SmartDualLogger:
        def __init__(self, path):
            self.terminal = sys.stdout
            self.log = open(path, "w", encoding="utf-8")
        def write(self, msg):
            self.terminal.write(msg)
            # 避免进度条内容写入日志文件
            if '\r' not in msg and '%|' not in msg:
                self.log.write(msg)
        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = SmartDualLogger(log_filename)
    sys.stderr = sys.stdout

    print(f">>> Running experiment with dataset: {args.dataset}")
    print(f">>> Logs will be saved to: {log_filename}")
    main(args)



'''
#没有清理缓存，有问题
import warnings
import argparse
import os
warnings.filterwarnings('ignore')
from experiments import exp_with_fixed_seed_in_nb201, exp_with_rand_seed_in_nb201, exp_with_fixed_seed_in_meta_predictor, exp_with_rand_seed_in_meta_predictor

def main(args):
    dataset_name = {'cifar10': 'cifar10', 'cifar100': 'cifar100', 'imagenet': 'ImageNet16-120', 'aircraft': 'aircraft', 'pets': 'pets'}
    assert args.dataset.lower() in list(dataset_name.keys()), f'ERROR: invalid dataset {args.dataset}'
    assert args.exp_type.lower() in ['reproduce', 'random'], f'ERROR: invalid exp_type {args.exp_type}'

    if args.exp_type.lower() == 'reproduce':
        if args.dataset.lower() in ['cifar10', 'cifar100', 'imagenet']:
            exp_with_fixed_seed_in_nb201(dataset=dataset_name[args.dataset.lower()])
        else:
            exp_with_fixed_seed_in_meta_predictor(dataset=dataset_name[args.dataset.lower()])
    else:
        if args.dataset.lower() in ['cifar10', 'cifar100', 'imagenet']:
            exp_with_rand_seed_in_nb201(dataset=dataset_name[args.dataset.lower()])
        else:
            exp_with_rand_seed_in_meta_predictor(dataset=dataset_name[args.dataset.lower()])

# ---------------- 主程序入口：参数解析 + 日志保存 + 实验运行 ----------------
if __name__ == "__main__":
    import sys
    import time
    import argparse

    # 1. 定义命令行参数
    parser = argparse.ArgumentParser(description='Run Evolutionary Diffusion NAS Experiments.')
    
    # 添加 --dataset 参数（必填），例如 --dataset pets 或 --dataset aircraft
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['cifar10', 'cifar100', 'aircraft', 'pets'], # 根据你的 config 里的列表调整
                        help='Name of the dataset to use (e.g., pets, aircraft)')
    
    # (可选) 添加一个参数来选择实验类型，方便切换
    parser.add_argument('--exp_type', type=str, default='meta_fixed', 
                        choices=['nb201_rand', 'nb201_fixed', 'meta_rand', 'meta_fixed'],
                        help='Type of experiment to run')

    args = parser.parse_args()

    # 2. 设置日志文件名（包含数据集名称和时间戳，方便区分）
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # 日志文件名示例：logs/exp_pets_meta_fixed_20260415_143022.txt
    log_filename = f"{log_dir}/100-step-CDENAG_exp_{args.dataset}_{args.exp_type}_{time.strftime('%Y%m%d_%H%M%S')}.txt"

    # 3. 定义日志类（同时输出到终端和文件）
    class SmartDualLogger:
        def __init__(self, log_file_path):
            self.terminal = sys.stdout
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            self.log_file = open(log_file_path, "w", encoding="utf-8")
           

        def write(self, message):
            self.terminal.write(message)
            is_progress_bar = '\r' in message or '%|' in message
            if not is_progress_bar:
                self.log_file.write(message)
            self.flush()

        def flush(self):
            self.terminal.flush()
            self.log_file.flush()

    # 启动日志重定向
    sys.stdout = SmartDualLogger(log_filename)
    sys.stderr = sys.stdout

    print(f">>> Running experiment with dataset: {args.dataset}")
    print(f">>> Logs will be saved to: {log_filename}")

    # 4. 根据参数运行对应的实验
    # 你可以根据自己的需求修改这里的逻辑
    if args.exp_type == 'meta_fixed':
        # 运行元学习预测器的固定种子实验（你之前运行 aircraft 的那个）
        exp_with_fixed_seed_in_meta_predictor(dataset=args.dataset)
    elif args.exp_type == 'meta_rand':
        # 运行元学习预测器的随机种子实验
        exp_with_rand_seed_in_meta_predictor(dataset=args.dataset)
    elif args.exp_type == 'nb201_fixed':
        exp_with_fixed_seed_in_nb201(dataset=args.dataset)
    elif args.exp_type == 'nb201_rand':
        exp_with_rand_seed_in_nb201(dataset=args.dataset)
'''
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EvoDiff-NAS')
    parser.add_argument('--exp_type', type=str, default='reproduce', help='reproduce, random')
    parser.add_argument('--dataset', type=str, help='cifar10, cifar100, imagenet, aircraft, pets')
    args = parser.parse_args()
    main(args)
'''