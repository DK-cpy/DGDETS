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
from utils.nb101_fitness import arch_fitness


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
    # ConsistencyGenerator 参数
    use_consistency_generator=True,
    teacher_model=None,
    lambda_kl=0.5,
    lambda_ce=1.0,
    lambda_consistency=0.2,
    consistency_type='mse',
    perturb_scale=0.02,
    tournament_size=5,
    history_max_len=10,
    stability_perturb_num=5,        # SATS 参数
    stability_perturb_scale=1e-3,   # SATS 参数
    verbose=True,
    print_freq=10,
):
    start_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"[EvoDiff] Starting evolution process (NAS-Bench-101)")
        print(f"  - Total steps: {num_step}, Population size: {population_num}")
        print(f"  - Using ConsistencyGenerator: {use_consistency_generator}")
        if use_consistency_generator:
            print(f"  - SATS perturb num: {stability_perturb_num}, scale: {stability_perturb_scale}")
            print(f"  - Lambda consistency: {lambda_consistency}, perturb scale: {perturb_scale}")
        print(f"{'='*60}\n")

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

            accurancy, fitness = arch_fitness(adj_matrix=x, nb_api=nb_api)
            fitness = mapping_fn(fitness)
            max_acc = accurancy.max().item()
            avg_acc = accurancy.mean().item()

            if not history_fitness:
                history_fitness = [fitness.clone()]
            else:
                history_x.append(x.clone())
                history_fitness.append(fitness.clone())
                if len(history_x) > history_max_len:
                    history_x.pop(0)
                    history_fitness.pop(0)

            if use_consistency_generator:
                generator = ConsistencyGenerator(
                    x=x, fitness=fitness, alpha=alpha, density='uniform', h=0.1,
                    elite_strategy=False,
                    teacher_model=teacher_model,
                    lambda_kl=lambda_kl, lambda_ce=lambda_ce,
                    lambda_consistency=lambda_consistency,
                    consistency_type=consistency_type,
                    perturb_scale=perturb_scale,
                    arch_fitness_fn=arch_fitness,   # NAS-Bench-101 适应度函数
                    api=nb_api,
                    tournament_size=tournament_size,
                    history_max_len=history_max_len,
                    history_x=history_x,
                    history_fitness=history_fitness,
                    verbose=verbose and (t == 0),
                    print_freq=print_freq,
                    is_single_step=(num_step == 1),
                    stability_perturb_num=stability_perturb_num,
                    stability_perturb_scale=stability_perturb_scale
                )
            else:
                generator = BayesianGenerator(
                    x=x, fitness=fitness, alpha=alpha, density='uniform', h=0.1,
                    elite_strategy=False, verbose=verbose and (t == 0)
                )

            x = generator.generate(x=x, noise=noise_scale, elite_rate=elite_rate)

            # Corrector (保持不变)
            if num_step > 1:
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
                     seed=seed, dataset="cifar10")
    end_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"[EvoDiff] Evolution completed")
        print(f"  - Final max accuracy: {max_acc_trace[-1]:.4f}")
        print(f"  - Total time: {end_time - start_time:.2f}s")
        print(f"{'='*60}\n")

    return max_acc_trace[-1], end_time - start_time, x

'''
import torch
import tqdm
import time
import math
import copy
from utils.mapping import Power, Energy, Identity
from utils.plot import plot_denoise
from utils.corrector import crossover, mutate, select
# 替换：导入 ConsistencyGenerator
from utils.predictor import ConsistencyGenerator
from utils.ddim import DDIMSchedulerCosine
from utils.corrector import TimeoutException
from utils.nb101_fitness import arch_fitness


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
    # ====================== 新增：ConsistencyGenerator 相关参数 ======================
    use_consistency_generator=True,
    teacher_model=None,
    lambda_kl=0.5,
    lambda_ce=1.0,
    lambda_consistency=0.5,
    consistency_type='mse',
    perturb_scale=0.1,
    tournament_size=5,
    history_max_len=10,
    verbose=True,
    print_freq=10,
    is_single_step=False,
    # ==================================================================================
):
    start_time = time.time()
    
    # ====================== 新增：启动信息详细打印 ======================
    if verbose:
        print(f"\n{'='*60}")
        print(f"[EvoDiff] Starting evolution process (NAS-Bench-101)")
        print(f"  - Search Space: NAS-Bench-101")
        print(f"  - Dataset: CIFAR-10")
        print(f"  - Total steps: {num_step}")
        print(f"  - Population size: {population_num}")
        print(f"  - Using ConsistencyGenerator: {use_consistency_generator}")
        print(f"  - Single-step mode: {is_single_step}")
        
        # 新增：ConsistencyGenerator 参数详细打印
        if use_consistency_generator:
            teacher_status = "Loaded" if teacher_model is not None else "None"
            print(f"\n  [ConsistencyGenerator Params]")
            print(f"    - Teacher model:        {teacher_status}")
            print(f"    - Lambda KL:            {lambda_kl}")
            print(f"    - Lambda CE:            {lambda_ce}")
            print(f"    - Lambda Consistency:   {lambda_consistency}")
            print(f"    - Consistency type:     {consistency_type}")
            print(f"    - Perturb scale:        {perturb_scale}")
            print(f"    - Tournament size:      {tournament_size}")
            print(f"    - History max len:      {history_max_len}")
        
        print(f"{'='*60}\n")
    # =======================================================================

    # 随机初始化种群样本
    x = torch.randn(population_num, geno_shape[0] * geno_shape[1])
    x_prev = copy.deepcopy(x)
    
    # 记录迭代中的种群样本和适应度值变化
    avg_acc_trace = []
    max_acc_trace = []
    
    # 初始化历史种群（用于 ConsistencyGenerator 锦标赛选择）
    history_x = [x.clone()]
    history_fitness = []
    
    # DDIM 调度器与映射函数
    scheduler = DDIMSchedulerCosine(num_step=num_step)
    mapping_fn = Energy(temperature=temperature)
    
    # 迭代去噪
    iter_start_time = time.time()
    bar = tqdm.tqdm(scheduler, ncols=120)
    
    try:
        for t, alpha in bar:
            if time.time() - iter_start_time > max_iter_time:
                raise TimeoutException
            iter_start_time = time.time()

            # 计算适应度值（NAS-Bench-101 特有逻辑）
            accurancy, fitness = arch_fitness(adj_matrix=x, nb_api=nb_api)
            fitness = mapping_fn(fitness)
            max_acc = accurancy.max().item()
            avg_acc = accurancy.mean().item()
            
            # 更新历史种群
            if not history_fitness:
                history_fitness = [fitness.clone()]
            else:
                history_x.append(x.clone())
                history_fitness.append(fitness.clone())
                if len(history_x) > history_max_len:
                    history_x.pop(0)
                    history_fitness.pop(0)

            # ====================== 修改：选择生成器 ======================
            if use_consistency_generator:
                generator = ConsistencyGenerator(
                    x=x,
                    fitness=fitness,
                    alpha=alpha,
                    density='uniform',
                    h=0.1,
                    elite_strategy=False,
                    teacher_model=teacher_model,
                    lambda_kl=lambda_kl,
                    lambda_ce=lambda_ce,
                    lambda_consistency=lambda_consistency,
                    consistency_type=consistency_type,
                    perturb_scale=perturb_scale,
                    arch_fitness_fn=arch_fitness,  # 传入 NAS-Bench-101 的适应度函数
                    api=nb_api,  # 传入 NAS-Bench-101 的 API
                    tournament_size=tournament_size,
                    history_max_len=history_max_len,
                    history_x=history_x,
                    history_fitness=history_fitness,
                    verbose=verbose and (t == 0),
                    print_freq=print_freq,
                    is_single_step=is_single_step
                )
            else:
                # 保留原 BayesianGenerator 作为基线（需确保 predictor.py 中仍有该类）
                from utils.predictor import BayesianGenerator
                generator = BayesianGenerator(
                    x=x, fitness=fitness, alpha=alpha, density='uniform', h=0.1, elite_strategy=False
                )
            # ==============================================================
            
            x = generator.generate(x=x, noise=noise_scale, elite_rate=elite_rate)

            # Corrector 阶段（NAS-Bench-101 特有逻辑）
            if not is_single_step:
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
            else:
                # 单步模式：只做轻微变异保持多样性
                x = mutate(
                    population=population_num,
                    x=x,
                    mut_rate=min(mutate_rate * 0.5, 0.1),
                    eta=mutate_distri_index,
                )
                x_prev = copy.deepcopy(x)

            # 记录指标
            avg_acc_trace.append(avg_acc)
            max_acc_trace.append(max_acc)
            bar.set_postfix(
                {
                    "max_acc": f"{max_acc:.2f}",
                    "avg_acc": f"{avg_acc:.2f}",
                }
            )

    except TimeoutException:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n>>> Programme exceeded time limit of {max_iter_time} seconds. Terminating...")
        last_max_acc = max_acc_trace[-1] if max_acc_trace else 0.0
        return last_max_acc, duration, x

    # 绘图
    if plot_results:
        plot_denoise(
            save_dir=save_dir,
            avg_acc_trace=avg_acc_trace,
            max_acc_trace=max_acc_trace,
            seed=seed,
            dataset="cifar10",
        )
    
    end_time = time.time()
    
    # ====================== 新增：结束信息详细打印 ======================
    if verbose:
        print(f"\n{'='*60}")
        print(f"[EvoDiff] Evolution completed (NAS-Bench-101)")
        print(f"  - Final max accuracy: {max_acc_trace[-1]:.4f}")
        print(f"  - Total time: {end_time - start_time:.2f}s")
        if use_consistency_generator and not is_single_step:
            print(f"  - Tournament selection was active throughout the process")
        elif use_consistency_generator and is_single_step:
            print(f"  - Single-step mode: Direct denoising with consistency prior")
        else:
            print(f"  - Using baseline BayesianGenerator")
        print(f"{'='*60}\n")
    # =======================================================================

    # 最终评估一次，确保获取最优架构
    accurancy, fitness = arch_fitness(adj_matrix=x, nb_api=nb_api)
    max_acc = max(accurancy.max().item(), max_acc_trace[-1])

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
from utils.nb101_fitness import arch_fitness


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

    # 随机初始化种群样本
    x = torch.randn(population_num, geno_shape[0] * geno_shape[1])
    x_prev = copy.deepcopy(x)

    # 记录迭代中的种群样本和适应度值变化
    avg_acc_trace = []
    max_acc_trace = []

    # 在DDIM中使用余弦alpha调度器，选择Energy映射函数
    scheduler = DDIMSchedulerCosine(num_step=num_step)
    mapping_fn = Energy(temperature=temperature)

    # 迭代去噪
    iter_start_time = time.time()
    bar = tqdm.tqdm(scheduler, ncols=120)
    for t, alpha in bar:
        try:
            if time.time() - iter_start_time > max_iter_time:
                raise TimeoutException
            iter_start_time = time.time()

            # 计算适应度值
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

            # 保存记录
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