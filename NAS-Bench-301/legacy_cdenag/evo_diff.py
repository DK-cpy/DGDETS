# ==================== evo_diff.py (NAS-Bench-301 修改版) ====================
import torch
import tqdm
import time
import math
import copy
from utils.mapping import Power, Energy, Identity
from utils.plot import plot_denoise
from utils.corrector import crossover, mutate, select
from utils.predictor import ConsistencyGenerator, BayesianGenerator
from utils.ddim import DDIMSchedulerCosine
from utils.corrector import TimeoutException
from utils.fitness import arch_fitness   # NB301 适应度函数


def evo_diff(
    nb_api,
    num_step,
    population_num,
    geno_shape,
    temperature,
    diver_rate,
    noise_scale,
    mutate_rate,
    elite_rate,
    mutate_distri_index,
    seed,
    plot_results,
    save_dir,
    max_iter_time,
    # ConsistencyGenerator 参数（完全对齐 NB201，含 SATS）
    use_consistency_generator=True,
    teacher_model=None,
    lambda_kl=0.5,
    lambda_ce=1.0,
    lambda_consistency=0.5,
    consistency_type='mse',
    perturb_scale=0.02,
    tournament_size=5,
    history_max_len=10,
    stability_perturb_num=5,
    stability_perturb_scale=1e-3,
    verbose=True,
    print_freq=10,
):
    start_time = time.time()
    is_single_step = (num_step == 1)

    if verbose:
        print(f"\n{'='*60}")
        print(f"[EvoDiff-301] Starting evolution process")
        print(f"  - Total steps: {num_step}, Population size: {population_num}")
        print(f"  - Using ConsistencyGenerator: {use_consistency_generator}")
        if use_consistency_generator:
            print(f"  - SATS perturb num: {stability_perturb_num}, scale: {stability_perturb_scale}")
            print(f"  - Lambda consistency: {lambda_consistency}, perturb scale: {perturb_scale}")
        print(f"{'='*60}\n")

    # 初始化种群
    x = torch.randn(population_num, geno_shape[0] * geno_shape[1])
    x_prev = copy.deepcopy(x)

    avg_acc_trace = []
    max_acc_trace = []
    history_x = [x.clone()]
    history_fitness = []

    scheduler = DDIMSchedulerCosine(num_step=num_step)
    mapping_fn = Energy(temperature=temperature)

    iter_start_time = time.time()
    bar = tqdm.tqdm(scheduler, ncols=120)

    try:
        for t, alpha in bar:
            if time.time() - iter_start_time > max_iter_time:
                raise TimeoutException
            iter_start_time = time.time()

            # 计算适应度（NB301 调用： arch_fitness(adj_matrix, nb_api)）
            accurancy, fitness = arch_fitness(adj_matrix=x, nb_api=nb_api)
            fitness = mapping_fn(fitness)
            max_acc = accurancy.max().item()
            avg_acc = accurancy.mean().item()

            # 更新历史
            if not history_fitness:
                history_fitness = [fitness.clone()]
            else:
                history_x.append(x.clone())
                history_fitness.append(fitness.clone())
                if len(history_x) > history_max_len:
                    history_x.pop(0)
                    history_fitness.pop(0)

            # 生成器
            if use_consistency_generator:
                generator = ConsistencyGenerator(
                    x=x, fitness=fitness, alpha=alpha, density='uniform', h=0.1,
                    elite_strategy=False,
                    teacher_model=teacher_model,
                    lambda_kl=lambda_kl, lambda_ce=lambda_ce,
                    lambda_consistency=lambda_consistency,
                    consistency_type=consistency_type,
                    perturb_scale=perturb_scale,
                    arch_fitness_fn=arch_fitness,   # NB301 适应度函数引用
                    api=nb_api,                     # 传入 NB301 api 对象
                    tournament_size=tournament_size,
                    history_max_len=history_max_len,
                    history_x=history_x,
                    history_fitness=history_fitness,
                    verbose=verbose and (t == 0),
                    print_freq=print_freq,
                    is_single_step=is_single_step,
                    stability_perturb_num=stability_perturb_num,
                    stability_perturb_scale=stability_perturb_scale
                )
            else:
                generator = BayesianGenerator(
                    x=x, fitness=fitness, alpha=alpha, density='uniform', h=0.1,
                    elite_strategy=False, verbose=verbose and (t == 0)
                )

            x = generator.generate(x=x, noise=noise_scale, elite_rate=elite_rate)

            # Corrector（保持原有逻辑，适配 NB301 的 select）
            if not is_single_step:
                if t != num_step - 1:
                    x = mutate(population=population_num, x=x, mut_rate=mutate_rate, eta=mutate_distri_index)
                    x = select(x_prev=x_prev, x_next=x, elite_rate=elite_rate,
                               diver_rate=diver_rate, nb_api=nb_api, max_iter_time=max_iter_time)
                    x_prev = copy.deepcopy(x)
                else:
                    x = mutate(population=population_num, x=x, mut_rate=mutate_rate,
                               eta=math.ceil(mutate_distri_index * 1.5))
                    x = select(x_prev=x_prev, x_next=x, elite_rate=elite_rate,
                               diver_rate=diver_rate, nb_api=nb_api, max_iter_time=max_iter_time)
                    x_prev = copy.deepcopy(x)
            else:
                x = mutate(population=population_num, x=x, mut_rate=min(mutate_rate * 0.5, 0.1),
                           eta=mutate_distri_index)
                x_prev = copy.deepcopy(x)

            avg_acc_trace.append(avg_acc)
            max_acc_trace.append(max_acc)
            bar.set_postfix({"max_acc": f"{max_acc:.2f}", "avg_acc": f"{avg_acc:.2f}"})

    except TimeoutException:
        end_time = time.time()
        print(f"\n>>> Timeout after {max_iter_time} seconds. Terminating...")
        last_max_acc = max_acc_trace[-1] if max_acc_trace else 0.0
        return last_max_acc, end_time - start_time, x

    if plot_results:
        plot_denoise(save_dir=save_dir, avg_acc_trace=avg_acc_trace, max_acc_trace=max_acc_trace,
                     seed=seed, dataset="cifar10")  # 或 "ImageNet16-120"，根据实际数据集调整

    end_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"[EvoDiff-301] Evolution completed")
        print(f"  - Final max accuracy: {max_acc_trace[-1]:.4f}")
        print(f"  - Total time: {end_time - start_time:.2f}s")
        print(f"{'='*60}\n")

    # 最终评估确保最优
    accurancy, fitness = arch_fitness(adj_matrix=x, nb_api=nb_api)
    max_acc = max(accurancy.max().item(), max_acc_trace[-1])
    return max_acc, end_time - start_time, x


'''
import torch
import tqdm
import time
import math
import copy
from utils.mapping import Power, Energy, Identity
from utils.plot import plot_denoise
from utils.corrector import crossover, mutate, select
from utils.predictor import ConsistencyGenerator  # 替换为一致性生成器
from utils.ddim import DDIMSchedulerCosine
from utils.corrector import TimeoutException
from utils.fitness import arch_fitness  # 适配 NAS-Bench-301 的 fitness 计算


def evo_diff(
    nb_api,
    num_step,
    population_num,
    geno_shape,
    temperature,
    diver_rate,
    noise_scale,
    mutate_rate,
    elite_rate,
    mutate_distri_index,
    seed,
    plot_results,
    save_dir,
    max_iter_time,
    # 新增 ConsistencyGenerator 相关参数（保持扩展性）
    tau=0.05,
    lambda_kl=0.5,
    lambda_ce=1.0,
    lambda_consistency=0.2,
    consistency_type='mse',
    perturb_scale=0.1,
    verbose=True,
    print_freq=10,
):
    start_time = time.time()
    
    # 判断是否为单步模式
    is_single_step = (num_step == 1)

    # 打印启动信息（新增 ConsistencyGenerator 参数打印）
    if verbose:
        print(f"\n{'='*60}")
        print(f"[EvoDiff-301] Starting evolution process")
        print(f"  - Total steps: {num_step}")
        print(f"  - Population size: {population_num}")
        print(f"  - Single-step mode: {is_single_step}")
        print(f"  - Using ConsistencyGenerator: True")
        
        # 打印 ConsistencyGenerator 核心参数
        print(f"\n  [ConsistencyGenerator Params]")
        print(f"    - Tau:                  {tau}")
        print(f"    - Lambda KL:            {lambda_kl}")
        print(f"    - Lambda CE:            {lambda_ce}")
        print(f"    - Lambda Consistency:   {lambda_consistency}")
        print(f"    - Consistency type:     {consistency_type}")
        print(f"    - Perturb scale:        {perturb_scale}")
        print(f"{'='*60}\n")

    # 初始化种群（NAS-Bench-301 基因型空间）
    x = torch.randn(population_num, geno_shape[0] * geno_shape[1])
    x_prev = copy.deepcopy(x)

    # 记录精度轨迹
    avg_acc_trace = []
    max_acc_trace = []

    # DDIM 调度器 + Energy 映射（适配 301 分布）
    scheduler = DDIMSchedulerCosine(num_step=num_step)
    mapping_fn = Energy(temperature=temperature)

    # 迭代过程（完全复用原有逻辑）
    iter_start_time = time.time()
    bar = tqdm.tqdm(scheduler, ncols=120)
    for t, alpha in bar:
        try:
            if time.time() - iter_start_time > max_iter_time:
                raise TimeoutException
            iter_start_time = time.time()

            # 计算 NAS-Bench-301 精度和适应度
            accurancy, fitness = arch_fitness(adj_matrix=x, nb_api=nb_api)
            fitness = mapping_fn(fitness)
            max_acc = accurancy.max().item()
            avg_acc = accurancy.mean().item()

            # 替换为一致性生成器（扩展参数，保持向后兼容）
            generator = ConsistencyGenerator(
                x=x, 
                fitness=fitness, 
                alpha=alpha, 
                tau=tau, 
                elite_strategy=False,
                lambda_kl=lambda_kl,
                lambda_ce=lambda_ce,
                lambda_consistency=lambda_consistency,
                consistency_type=consistency_type,
                perturb_scale=perturb_scale,
                verbose=verbose and (t == 0),  # 仅在第一步打印生成器初始化信息
                print_freq=print_freq
            )
            x = generator.generate(x=x, noise=noise_scale, elite_rate=elite_rate)

            # 进化矫正（复用原有逻辑）
            if t != num_step - 1:
                x = mutate(
                    population=population_num,
                    x=x,
                    mut_rate=mutate_rate,
                    eta=mutate_distri_index,
                )
                x = select(
                    x_prev=x_prev,
                    x_next=x,
                    elite_rate=elite_rate,
                    diver_rate=diver_rate,
                    nb_api=nb_api,
                    max_iter_time=max_iter_time,
                )
                x_prev = copy.deepcopy(x)
            else:
                x = mutate(
                    population=population_num,
                    x=x,
                    mut_rate=mutate_rate,
                    eta=math.ceil(mutate_distri_index * 1.5),
                )
                x = select(
                    x_prev=x_prev,
                    x_next=x,
                    elite_rate=elite_rate,
                    diver_rate=diver_rate,
                    nb_api=nb_api,
                    max_iter_time=max_iter_time,
                )
                x_prev = copy.deepcopy(x)

            # 更新轨迹
            avg_acc_trace.append(avg_acc)
            max_acc_trace.append(max_acc)
            bar.set_postfix(
                {
                    "max_acc": f"{max_acc:.2f}",
                    "avg_acc": f"{avg_acc:.2f}",
                }
            )

        except TimeoutException:
            print(
                f"\n>>> Programme exceeded time limit of {max_iter_time} seconds. Terminating..."
            )
            return 0.0, 0.0, x

    # 绘图（复用原有逻辑）
    if plot_results:
        plot_denoise(
            save_dir=save_dir,
            avg_acc_trace=avg_acc_trace,
            max_acc_trace=max_acc_trace,
            seed=seed,
            dataset="cifar10",  # 适配 NAS-Bench-301 数据集
        )
    end_time = time.time()

    # 打印结束信息
    if verbose:
        print(f"\n{'='*60}")
        print(f"[EvoDiff-301] Evolution completed")
        print(f"  - Final max accuracy: {max_acc_trace[-1]:.4f}")
        print(f"  - Total time: {end_time - start_time:.2f}s")
        if is_single_step:
            print(f"  - Single-step mode: Direct denoising with consistency prior")
        print(f"{'='*60}\n")

    # 最终精度计算
    accurancy, fitness = arch_fitness(adj_matrix=x, nb_api=nb_api)
    max_acc = max(accurancy.max().item(), max_acc)

    # 保持输出参数完全不变
    return max_acc, end_time - start_time, x
'''

'''
import torch
import tqdm
import time
import math
import copy
from utils.mapping import Power, Energy, Identity
from utils.plot import plot_denoise
from utils.corrector import crossover, mutate, select
from utils.predictor import BayesianGenerator
from utils.ddim import DDIMSchedulerCosine
from utils.corrector import TimeoutException
from utils.fitness import arch_fitness


def evo_diff(
    nb_api,
    num_step,
    population_num,
    geno_shape,
    temperature,
    diver_rate,
    noise_scale,
    mutate_rate,
    elite_rate,
    mutate_distri_index,
    seed,
    plot_results,
    save_dir,
    max_iter_time,
):
    start_time = time.time()

    #
    x = torch.randn(population_num, geno_shape[0] * geno_shape[1])
    x_prev = copy.deepcopy(x)

    #
    avg_acc_trace = []
    max_acc_trace = []

    # DDIMalpha，Energy
    scheduler = DDIMSchedulerCosine(num_step=num_step)
    mapping_fn = Energy(temperature=temperature)

    #
    iter_start_time = time.time()
    bar = tqdm.tqdm(scheduler, ncols=120)
    for t, alpha in bar:
        try:
            if time.time() - iter_start_time > max_iter_time:
                raise TimeoutException
            iter_start_time = time.time()

            #
            accurancy, fitness = arch_fitness(adj_matrix=x, nb_api=nb_api)

            fitness = mapping_fn(fitness)
            max_acc = accurancy.max().item()
            avg_acc = accurancy.mean().item()

            # Predictor
            generator = BayesianGenerator(x=x, fitness=fitness, alpha=alpha)
            x = generator.generate(x=x, noise=noise_scale, elite_rate=elite_rate)

            # Corrector
            if t != num_step - 1:
                x = mutate(
                    population=population_num,
                    x=x,
                    mut_rate=mutate_rate,
                    eta=mutate_distri_index,
                )
                x = select(
                    x_prev=x_prev,
                    x_next=x,
                    elite_rate=elite_rate,
                    diver_rate=diver_rate,
                    nb_api=nb_api,
                    max_iter_time=max_iter_time,
                )
                x_prev = copy.deepcopy(x)
            else:
                x = mutate(
                    population=population_num,
                    x=x,
                    mut_rate=mutate_rate,
                    eta=math.ceil(mutate_distri_index * 1.5),
                )
                x = select(
                    x_prev=x_prev,
                    x_next=x,
                    elite_rate=elite_rate,
                    diver_rate=diver_rate,
                    nb_api=nb_api,
                    max_iter_time=max_iter_time,
                )
                x_prev = copy.deepcopy(x)

            #
            avg_acc_trace.append(avg_acc)
            max_acc_trace.append(max_acc)
            bar.set_postfix(
                {
                    "max_acc": f"{max_acc:.2f}",
                    "avg_acc": f"{avg_acc:.2f}",
                }
            )

        except TimeoutException:
            print(
                f"\n>>> Programme exceeded time limit of {max_iter_time} seconds. Terminating..."
            )
            return 0.0, 0.0, x

    if plot_results:
        plot_denoise(
            save_dir=save_dir,
            avg_acc_trace=avg_acc_trace,
            max_acc_trace=max_acc_trace,
            seed=seed,
            dataset="cifar10",
        )
    end_time = time.time()

    accurancy, fitness = arch_fitness(adj_matrix=x, nb_api=nb_api)
    max_acc = max(accurancy.max().item(), max_acc)

    return max_acc, end_time - start_time, x
'''