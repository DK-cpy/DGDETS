import torch
import time
import os
import random
import warnings
import numpy as np
import logging
from config.config import hyper_params_setting
from evo_diff import evo_diff
from utils.nb101_api import get_nb101_api

warnings.filterwarnings("ignore")


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_logging(save_dir):
    """配置日志：所有种子共用一个日志文件，同时输出到控制台"""
    log_filename = os.path.join(save_dir, "nas_bench101_all_seeds.log")
    os.makedirs(save_dir, exist_ok=True)

    # 获取根日志器并设置级别
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除已有处理器，避免重复添加
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 格式：时间 + 级别 + 消息
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器（追加模式，以便之后可以多次运行写入同一个文件）
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def main_exp():
    nb_api = get_nb101_api()

    args = hyper_params_setting
    seed_list = args["seed"]
    save_dir = args["save_dir"]

    # 所有种子共用一个日志文件
    logger = setup_logging(save_dir)

    logger.info(f"{'='*30} Experiment started {'='*30}")
    logger.info(f"Seeds: {seed_list}")

    # 用于收集每个种子的最终结果
    results = []

    for seed in seed_list:
        set_random_seed(seed)
        logger.info(f"----- Seed {seed} started -----")
        logger.info(f"Hyper-parameters: num_step={args['num_step']}, "
                    f"population_num={args['population_num']}, "
                    f"temperature={args['temperature']}, "
                    f"noise_scale={args['noise_scale']}, "
                    f"mutate_rate={args['mutate_rate']}, "
                    f"elite_rate={args['elite_rate']}, "
                    f"diver_rate={args['diver_rate']}, "
                    f"max_iter_time={args['max_iter_time']}")

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

        result_str = f"Seed {seed}: Max accuracy = {max_acc:.2f}%, Duration = {duration:.2f} seconds"
        logger.info(result_str)
        results.append((seed, max_acc, duration))

    # 全部运行结束后打印汇总
    logger.info(f"{'='*30} Summary of all seeds {'='*30}")
    for seed, acc, dur in results:
        logger.info(f"Seed {seed}: Max accuracy = {acc:.2f}%, Duration = {dur:.2f} seconds")
    logger.info(f"{'='*60}\n")

    # 清理日志处理器（可选）
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


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
from utils.nb101_api import get_nb101_api

warnings.filterwarnings("ignore")


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main_exp():
    nb_api = get_nb101_api()

    # 导入Hyper-parameters设置
    args = hyper_params_setting
    seed_list = args["seed"]

    for seed in seed_list:
        set_random_seed(seed)
        max_acc, duration, _ = evo_diff(
            nb_api=nb_api,
            num_step=args["num_step"],
            population_num=args["population_num"],
            geno_shape=args["geno_shape"],
            temperature=args["temperature"],  # 温度参数，控制适应度值的数值规模
            diver_rate=args["diver_rate"],  # 多样性程度
            noise_scale=args["noise_scale"],  # 噪声强度，控制扩散的探索性与稳定性
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