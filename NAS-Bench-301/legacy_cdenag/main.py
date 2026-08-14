import torch
import time
import os
import sys
import random
import warnings
import numpy as np
from config.config import hyper_params_setting
from evo_diff import evo_diff
from utils.NB301 import get_api

warnings.filterwarnings("ignore")


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Tee:
    """同时将输出写入控制台和日志文件"""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def main_exp():
    nb_api = get_api()
    args = hyper_params_setting
    seed_list = args["seed"]
    save_dir = args.get("save_dir", "./results")
    os.makedirs(save_dir, exist_ok=True)

    # 所有种子共用一个日志文件（追加模式）
    log_path = os.path.join(save_dir, "nas301_all_seeds.log")
    log_file = open(log_path, 'a', encoding='utf-8')

    # 保存原始输出流并重定向
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    print(f"\n{'#'*60}")
    print(f"Experiment started. Seeds: {seed_list}")
    print(f"Log file: {log_path}")
    print(f"{'#'*60}\n")

    # 用于收集所有种子的最终结果
    results = []

    for seed in seed_list:
        set_random_seed(seed)
        print(f"\n{'='*60}")
        print(f"Running experiment with seed {seed}")
        print(f"Hyper-params: num_step={args['num_step']}, "
              f"population={args['population_num']}, temperature={args['temperature']}")
        print(f"{'='*60}\n")

        max_acc, duration, _ = evo_diff(
            nb_api=nb_api,
            num_step=args["num_step"],
            population_num=args["population_num"],
            geno_shape=args["geno_shape"],
            temperature=args["temperature"],
            diver_rate=args["diver_rate"],
            noise_scale=args["noise_scale"],
            mutate_rate=args["mutate_rate"],
            elite_rate=args["elite_rate"],
            mutate_distri_index=args["mutate_distri_index"],
            seed=seed,
            plot_results=False,
            save_dir=save_dir,
            max_iter_time=args["max_iter_time"],
        )

        # 记录结果
        results.append((seed, max_acc, duration))

        # 即时输出单个种子完成信息
        print(f">>> Seed {seed} finished. Max accuracy: {max_acc:.2f}%, Duration: {duration:.2f}s")

    # ===== 汇总输出 =====
    print(f"\n{'#'*60}")
    print("Summary of all seeds:")
    for seed, acc, dur in results:
        print(f">>> Seed {seed} finished. Max accuracy: {acc:.2f}%, Duration: {dur:.2f}s")
    print(f"{'#'*60}\n")

    # 恢复原始输出流，关闭日志
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()
    print(f"All experiments completed. Results saved to {log_path}")


if __name__ == "__main__":
    main_exp()

'''
import torch
import time
import os
import random
import warnings
import numpy as np
from config.config import (
    hyper_params_setting,
)
from evo_diff import evo_diff
from utils.NB301 import get_api
warnings.filterwarnings("ignore")


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main_exp():
    nb_api = get_api()

    # Hyper-parameters
    args = hyper_params_setting
    seed_list = args["seed"]

    for seed in seed_list:
        #
        set_random_seed(seed)
        max_acc, duration, _ = evo_diff(
            nb_api=nb_api,
            num_step=args["num_step"],
            population_num=args["population_num"],
            geno_shape=args["geno_shape"],
            temperature=args["temperature"],  # ，
            diver_rate=args["diver_rate"],  #
            noise_scale=args["noise_scale"],  # ，
            mutate_rate=args["mutate_rate"],
            elite_rate=args["elite_rate"],
            mutate_distri_index=args["mutate_distri_index"],
            seed=seed,
            plot_results=False,
            save_dir=args["save_dir"],
            max_iter_time=args["max_iter_time"],
        )
        print(f">>> Running on cifar10 with seed {seed}, max accuracy: {max_acc:.2f} %")


if __name__ == "__main__":
    main_exp()
'''