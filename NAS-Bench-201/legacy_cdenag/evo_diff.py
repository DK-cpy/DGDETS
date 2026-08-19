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
from utils.nb201_fitness import arch_fitness
from utils.meta_fitness import meta_arch_fitness
from utils.meta_d2a import FitnessRestorer
from utils.corrector import TimeoutException
from utils.analyse import compute_uniqueness
from utils.meta_d2a import MetaSurrogateUnnoisedModel, load_graph_config, load_model

# ========================== 极简基因消毒函数 ==========================
def sanitize_genes(x):
    """
    不修改 nb201_fitness.py，直接修改基因矩阵 x。
    原理：将 x reshape 后，强制把对应 'input'(索引0) 和 'output'(索引1) 的值设为极小，
    但不是 -inf，避免数值问题。
    """
    batch_size = x.shape[0]
    # Reshape 为 (Batch, 8, 7)
    x_reshaped = x.view(batch_size, 8, 7)
    
    # 核心操作：把前两列的值设为当前行最小值减 100，确保 argmax 不会选它们
    min_vals = x_reshaped.min(dim=2, keepdim=True)[0]
    x_reshaped[:, :, 0] = min_vals[:, :, 0] - 100.0  # 屏蔽 input
    x_reshaped[:, :, 1] = min_vals[:, :, 0] - 100.0  # 屏蔽 output
    
    # 恢复形状
    return x_reshaped.view(batch_size, -1)
# ===========================================================================

def evo_diff(
    dataset,
    api,
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
    nb201_or_meta,
    max_iter_time,
    use_consistency_generator=True,
    teacher_model=None,
    lambda_kl=0.5,
    lambda_ce=1.0,
    lambda_consistency=0.5,
    consistency_type='mse',
    perturb_scale=0.02,
    tournament_size=5,
    history_max_len=10,
    verbose=True,
    print_freq=10,
):
    start_time = time.time()
    assert nb201_or_meta == "nb201", "nb201_or_meta should be nb201, but got {}".format(
        nb201_or_meta
    )
    is_single_step = (num_step == 1)
    if verbose:
        print(f"\n{'='*60}")
        print(f"[EvoDiff] Starting evolution process")
        print(f"  - Dataset: {dataset}")
        print(f"  - Total steps: {num_step}")
        print(f"  - Population size: {population_num}")
        print(f"  - Using ConsistencyGenerator: {use_consistency_generator}")
        print(f"  - Single-step mode: {is_single_step}")
        
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
    
    x = torch.randn(population_num, geno_shape[0] * geno_shape[1])
    x_prev = copy.deepcopy(x)
    avg_acc_trace = []
    max_acc_trace = []
    valid_rate_trace = []
    uniq_rate_trace = []
    history_x = [x.clone()]
    history_fitness = []
    scheduler = DDIMSchedulerCosine(num_step=num_step)
    mapping_fn = Energy(temperature=temperature)
    iter_start_time = time.time()
    bar = tqdm.tqdm(scheduler, ncols=120)
    final_uniq_rate = 0.0
    
    try:
        for t, alpha in bar:
            if time.time() - iter_start_time > max_iter_time:
                raise TimeoutException
            iter_start_time = time.time()
            
            # 计算适应度
            accurancy, fitness, valid_rate = arch_fitness(
                operation_matrix=x, api=api, dataset=dataset
            )
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
            
            # 生成器：nb201 模式下不传递任何额外参数（test_dataset 等）
            if use_consistency_generator:
                generator = ConsistencyGenerator(
                    x=x, fitness=fitness, alpha=alpha, density='uniform', h=0.1,
                    elite_strategy=False, teacher_model=teacher_model, lambda_kl=lambda_kl,
                    lambda_ce=lambda_ce, lambda_consistency=lambda_consistency,
                    consistency_type=consistency_type, perturb_scale=perturb_scale,
                    arch_fitness_fn=arch_fitness, api=api, dataset=dataset,
                    tournament_size=tournament_size, history_max_len=history_max_len,
                    history_x=history_x, history_fitness=history_fitness,
                    verbose=verbose and (t == 0), print_freq=print_freq, is_single_step=is_single_step
                    # 注意：不传递 test_dataset, meta_surrogate_unnoised_model, nasbench201, fitness_restorer
                )
            else:
                generator = BayesianGenerator(
                    x=x, fitness=fitness, alpha=alpha, density='uniform', h=0.1,
                    elite_strategy=False, verbose=verbose and (t == 0)
                )
            
            x = generator.generate(x=x, noise=noise_scale, elite_rate=elite_rate)
            
            # Corrector
            if not is_single_step:
                if t != num_step - 1:
                    x = mutate(population=population_num, x=x, mut_rate=mutate_rate, eta=mutate_distri_index)
                    x = select(x_prev=x_prev, x_next=x, fitness_prev=fitness, elite_rate=elite_rate,
                               diver_rate=diver_rate, api=api, dataset=dataset, max_iter_time=max_iter_time,
                               nb201_or_meta=nb201_or_meta, test_dataset=None, meta_surrogate_unnoised_model=None,
                               nasbench201=None, fitness_restorer=None)
                    x_prev = copy.deepcopy(x)
                else:
                    x = mutate(population=population_num, x=x, mut_rate=mutate_rate, eta=math.ceil(mutate_distri_index * 1.5))
                    x = select(x_prev=x_prev, x_next=x, fitness_prev=fitness, elite_rate=elite_rate,
                               diver_rate=diver_rate, api=api, dataset=dataset, max_iter_time=max_iter_time,
                               nb201_or_meta=nb201_or_meta, test_dataset=None, meta_surrogate_unnoised_model=None,
                               nasbench201=None, fitness_restorer=None)
                    x_prev = copy.deepcopy(x)
            else:
                x = mutate(population=population_num, x=x, mut_rate=min(mutate_rate * 0.5, 0.1), eta=mutate_distri_index)
                x_prev = copy.deepcopy(x)
            
            uniq_rate = compute_uniqueness(arch_op_matrices=x)
            final_uniq_rate = uniq_rate
            avg_acc_trace.append(avg_acc)
            max_acc_trace.append(max_acc)
            valid_rate_trace.append(valid_rate)
            uniq_rate_trace.append(uniq_rate)
            bar.set_postfix({"max_acc": f"{max_acc:.2f}", "avg_acc": f"{avg_acc:.2f}", "valid_rate": f"{valid_rate:.2f}", "uniq_rate": f"{uniq_rate:.2f}"})

    except TimeoutException:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n>>> Programme exceeded time limit of {max_iter_time} seconds. Terminating...")
        last_max_acc = max_acc_trace[-1] if max_acc_trace else 0.0
        x = sanitize_genes(x)
        return last_max_acc, duration, final_uniq_rate, x

    if plot_results:
        plot_denoise(save_dir=save_dir, avg_acc_trace=avg_acc_trace, max_acc_trace=max_acc_trace,
                     valid_rate_trace=valid_rate_trace, uniq_rate_trace=uniq_rate_trace, seed=seed, dataset=dataset)
    
    end_time = time.time()
    if verbose:
        print(f"\n{'='*60}")
        print(f"[EvoDiff] Evolution completed")
        print(f"  - Final max accuracy: {max_acc_trace[-1]:.4f}")
        print(f"  - Total time: {end_time - start_time:.2f}s")
        print(f"  - Final uniqueness: {uniq_rate_trace[-1]:.4f}")
        print(f"{'='*60}\n")
    
    x = sanitize_genes(x)
    return max_acc_trace[-1], end_time - start_time, uniq_rate_trace[-1], x


def evo_diff_meta(
    dataset,
    api,
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
    nb201_or_meta,
    max_iter_time,
    use_consistency_generator=True,
    teacher_model=None,
    lambda_kl=0.5,
    lambda_ce=1.0,
    lambda_consistency=0.5,
    consistency_type='mse',
    perturb_scale=0.02,
    tournament_size=5,
    history_max_len=10,
    verbose=True,
    print_freq=10,
):
    assert nb201_or_meta == "meta", "nb201_or_meta should be meta, but got {}".format(nb201_or_meta)
    is_single_step = (num_step == 1)
    if verbose:
        print(f"\n{'='*60}")
        print(f"[EvoDiff Meta] Starting evolution process")
        print(f"  - Dataset: {dataset}")
        print(f"  - Total steps: {num_step}")
        print(f"  - Population size: {population_num}")
        print(f"  - Using ConsistencyGenerator: {use_consistency_generator}")
        print(f"  - Single-step mode: {is_single_step}")
        
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
    
    x = torch.randn(population_num, geno_shape[0] * geno_shape[1])
    x_prev = copy.deepcopy(x)
    avg_acc_trace = []
    max_acc_trace = []
    valid_rate_trace = []
    uniq_rate_trace = []
    history_x = [x.clone()]
    history_fitness = []
    scheduler = DDIMSchedulerCosine(num_step=num_step)
    mapping_fn = Energy(temperature=temperature)
    
    # 加载Meta模型
    nasbench201 = torch.load("meta_acc_predictor/data/nasbench201.pt")
    graph_config = load_graph_config(graph_data_name="nasbench201", nvt=7, data_path="meta_acc_predictor/data/nasbench201.pt")
    meta_surrogate_unnoised_model = MetaSurrogateUnnoisedModel(nvt=7, hs=512, nz=56, num_sample=20, graph_config=graph_config)
    meta_surrogate_unnoised_model = load_model(model=meta_surrogate_unnoised_model, ckpt_path="meta_acc_predictor/unnoised_checkpoint.pth.tar")
    fitness_restorer = FitnessRestorer(dataset_name=dataset, num_sample=20, seed=seed)
    
    start_time = time.time()
    iter_start_time = time.time()
    bar = tqdm.tqdm(scheduler, ncols=120)
    final_uniq_rate = 0.0
    
    try:
        for t, alpha in bar:
            if time.time() - iter_start_time > max_iter_time:
                raise TimeoutException
            iter_start_time = time.time()
            
            # 计算适应度（meta）
            accurancy, fitness, valid_rate = meta_arch_fitness(
                operation_matrix=x, api=api, dataset=dataset, test_dataset=None,
                meta_surrogate_unnoised_model=meta_surrogate_unnoised_model, nasbench201=nasbench201,
                fitness_restorer=fitness_restorer
            )
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
            
            # 生成器：meta 模式下传递所有必要的额外参数
            if use_consistency_generator:
                generator = ConsistencyGenerator(
                    x=x, fitness=fitness, alpha=alpha, density='uniform', h=0.1,
                    elite_strategy=False, teacher_model=teacher_model, lambda_kl=lambda_kl,
                    lambda_ce=lambda_ce, lambda_consistency=lambda_consistency,
                    consistency_type=consistency_type, perturb_scale=perturb_scale,
                    arch_fitness_fn=meta_arch_fitness, api=api, dataset=dataset,
                    tournament_size=tournament_size, history_max_len=history_max_len,
                    history_x=history_x, history_fitness=history_fitness,
                    verbose=verbose and (t == 0), print_freq=print_freq, is_single_step=is_single_step,
                    test_dataset=None, meta_surrogate_unnoised_model=meta_surrogate_unnoised_model,
                    nasbench201=nasbench201, fitness_restorer=fitness_restorer
                )
            else:
                generator = BayesianGenerator(
                    x=x, fitness=fitness, alpha=alpha, density='uniform', h=0.1,
                    elite_strategy=False, verbose=verbose and (t == 0)
                )
            
            x = generator.generate(x=x, noise=noise_scale, elite_rate=elite_rate)
            
            # Corrector
            if not is_single_step:
                if t != num_step - 1:
                    x = mutate(population=population_num, x=x, mut_rate=mutate_rate, eta=mutate_distri_index)
                    x = select(x_prev=x_prev, x_next=x, fitness_prev=fitness, elite_rate=elite_rate,
                               diver_rate=diver_rate, api=api, dataset=dataset, max_iter_time=max_iter_time,
                               nb201_or_meta=nb201_or_meta, test_dataset=None, meta_surrogate_unnoised_model=meta_surrogate_unnoised_model,
                               nasbench201=nasbench201, fitness_restorer=fitness_restorer)
                    x_prev = copy.deepcopy(x)
                else:
                    x = mutate(population=population_num, x=x, mut_rate=mutate_rate, eta=math.ceil(mutate_distri_index * 1.5))
                    x = select(x_prev=x_prev, x_next=x, fitness_prev=fitness, elite_rate=elite_rate,
                               diver_rate=diver_rate, api=api, dataset=dataset, max_iter_time=max_iter_time,
                               nb201_or_meta=nb201_or_meta, test_dataset=None, meta_surrogate_unnoised_model=meta_surrogate_unnoised_model,
                               nasbench201=nasbench201, fitness_restorer=fitness_restorer)
                    x_prev = copy.deepcopy(x)
            else:
                x = mutate(population=population_num, x=x, mut_rate=min(mutate_rate * 0.5, 0.1), eta=mutate_distri_index)
                x_prev = copy.deepcopy(x)
            
            uniq_rate = compute_uniqueness(arch_op_matrices=x)
            final_uniq_rate = uniq_rate
            avg_acc_trace.append(avg_acc)
            max_acc_trace.append(max_acc)
            valid_rate_trace.append(valid_rate)
            uniq_rate_trace.append(uniq_rate)
            bar.set_postfix({"max_pred_acc": f"{max_acc:.2f}", "avg_pred_acc": f"{avg_acc:.2f}", "valid_rate": f"{valid_rate:.2f}", "uniq_rate": f"{uniq_rate:.2f}"})

    except TimeoutException:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n>>> Programme exceeded time limit of {max_iter_time} seconds. Terminating...")
        last_max_acc = max_acc_trace[-1] if max_acc_trace else 0.0
        x = sanitize_genes(x)
        return last_max_acc, duration, final_uniq_rate, x

    end_time = time.time()
    if plot_results:
        plot_denoise(save_dir=save_dir, avg_acc_trace=avg_acc_trace, max_acc_trace=max_acc_trace,
                     valid_rate_trace=valid_rate_trace, uniq_rate_trace=uniq_rate_trace, seed=seed, dataset=dataset)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[EvoDiff Meta] Evolution completed")
        print(f"  - Final max predicted accuracy: {max_acc_trace[-1]:.4f}")
        print(f"  - Total time: {end_time - start_time:.2f}s")
        print(f"  - Final uniqueness: {uniq_rate_trace[-1]:.4f}")
        print(f"{'='*60}\n")
    
    x = sanitize_genes(x)
    return max_acc_trace[-1], end_time - start_time, uniq_rate_trace[-1], x

'''
#Aircraft和Pets版本
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
from utils.nb201_fitness import arch_fitness
from utils.meta_fitness import meta_arch_fitness
from utils.meta_d2a import FitnessRestorer
from utils.corrector import TimeoutException
from utils.analyse import compute_uniqueness
from utils.meta_d2a import MetaSurrogateUnnoisedModel, load_graph_config, load_model

# ========================== 新增：极简基因消毒函数 ==========================
def sanitize_genes(x):
    """
    不修改 nb201_fitness.py，直接修改基因矩阵 x。
    原理：将 x reshape 后，强制把对应 'input'(索引0) 和 'output'(索引1) 的值设为极小，
    但不是 -inf，避免数值问题。
    """
    batch_size = x.shape[0]
    # Reshape 为 (Batch, 8, 7)
    x_reshaped = x.view(batch_size, 8, 7)
    
    # 核心操作：
    # 我们不设为 -inf，而是把前两列的值设为当前行最小值减 100
    # 这样既保证了 argmax 不会选它们，又避免了数值不稳定
    min_vals = x_reshaped.min(dim=2, keepdim=True)[0]
    x_reshaped[:, :, 0] = min_vals[:, :, 0] - 100.0  # 屏蔽 input
    x_reshaped[:, :, 1] = min_vals[:, :, 0] - 100.0  # 屏蔽 output
    
    # 恢复形状
    return x_reshaped.view(batch_size, -1)
# ===========================================================================

def evo_diff(
    dataset,
    api,
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
    nb201_or_meta,
    max_iter_time,
    use_consistency_generator=True,
    teacher_model=None,
    lambda_kl=0.5,
    lambda_ce=1.0,
    lambda_consistency=0.5,
    consistency_type='mse',
    perturb_scale=0.01,
    tournament_size=5,
    history_max_len=10,
    verbose=True,
    print_freq=10,
):
    start_time = time.time()
    assert nb201_or_meta == "nb201", "nb201_or_meta should be nb201, but got {}".format(
        nb201_or_meta
    )
    is_single_step = (num_step == 1)
    if verbose:
        print(f"\n{'='*60}")
        print(f"[EvoDiff] Starting evolution process")
        print(f"  - Dataset: {dataset}")
        print(f"  - Total steps: {num_step}")
        print(f"  - Population size: {population_num}")
        print(f"  - Using ConsistencyGenerator: {use_consistency_generator}")
        print(f"  - Single-step mode: {is_single_step}")
        
        # ========================== 新增：ConsistencyGenerator 参数详细打印 ==========================
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
        # ============================================================================================
        
        print(f"{'='*60}\n")
    
    x = torch.randn(population_num, geno_shape[0] * geno_shape[1])
    x_prev = copy.deepcopy(x)
    avg_acc_trace = []
    max_acc_trace = []
    valid_rate_trace = []
    uniq_rate_trace = []
    history_x = [x.clone()]
    history_fitness = []
    scheduler = DDIMSchedulerCosine(num_step=num_step)
    mapping_fn = Energy(temperature=temperature)
    iter_start_time = time.time()
    bar = tqdm.tqdm(scheduler, ncols=120)
    final_uniq_rate = 0.0
    
    try:
        for t, alpha in bar:
            if time.time() - iter_start_time > max_iter_time:
                raise TimeoutException
            iter_start_time = time.time()
            
            # 计算适应度
            accurancy, fitness, valid_rate = arch_fitness(
                operation_matrix=x, api=api, dataset=dataset
            )
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
            
            # 生成器
            if use_consistency_generator:
                generator = ConsistencyGenerator(
                    x=x, fitness=fitness, alpha=alpha, density='uniform', h=0.1,
                    elite_strategy=False, teacher_model=teacher_model, lambda_kl=lambda_kl,
                    lambda_ce=lambda_ce, lambda_consistency=lambda_consistency,
                    consistency_type=consistency_type, perturb_scale=perturb_scale,
                    arch_fitness_fn=arch_fitness, api=api, dataset=dataset,
                    tournament_size=tournament_size, history_max_len=history_max_len,
                    history_x=history_x, history_fitness=history_fitness,
                    verbose=verbose and (t == 0), print_freq=print_freq, is_single_step=is_single_step,
                    test_dataset=None,meta_surrogate_unnoised_model=None,
                    nasbench201=None,fitness_restorer=None
                )
            else:
                generator = BayesianGenerator(
                    x=x, fitness=fitness, alpha=alpha, density='uniform', h=0.1,
                    elite_strategy=False, verbose=verbose and (t == 0)
                )
            
            x = generator.generate(x=x, noise=noise_scale, elite_rate=elite_rate)
            
            # Corrector
            if not is_single_step:
                if t != num_step - 1:
                    x = mutate(population=population_num, x=x, mut_rate=mutate_rate, eta=mutate_distri_index)
                    x = select(x_prev=x_prev, x_next=x, fitness_prev=fitness, elite_rate=elite_rate,
                               diver_rate=diver_rate, api=api, dataset=dataset, max_iter_time=max_iter_time,
                               nb201_or_meta=nb201_or_meta, test_dataset=None, meta_surrogate_unnoised_model=None,
                               nasbench201=None, fitness_restorer=None)
                    x_prev = copy.deepcopy(x)
                else:
                    x = mutate(population=population_num, x=x, mut_rate=mutate_rate, eta=math.ceil(mutate_distri_index * 1.5))
                    x = select(x_prev=x_prev, x_next=x, fitness_prev=fitness, elite_rate=elite_rate,
                               diver_rate=diver_rate, api=api, dataset=dataset, max_iter_time=max_iter_time,
                               nb201_or_meta=nb201_or_meta, test_dataset=None, meta_surrogate_unnoised_model=None,
                               nasbench201=None, fitness_restorer=None)
                    x_prev = copy.deepcopy(x)
            else:
                x = mutate(population=population_num, x=x, mut_rate=min(mutate_rate * 0.5, 0.1), eta=mutate_distri_index)
                x_prev = copy.deepcopy(x)
            
            uniq_rate = compute_uniqueness(arch_op_matrices=x)
            final_uniq_rate = uniq_rate
            avg_acc_trace.append(avg_acc)
            max_acc_trace.append(max_acc)
            valid_rate_trace.append(valid_rate)
            uniq_rate_trace.append(uniq_rate)
            bar.set_postfix({"max_acc": f"{max_acc:.2f}", "avg_acc": f"{avg_acc:.2f}", "valid_rate": f"{valid_rate:.2f}", "uniq_rate": f"{uniq_rate:.2f}"})

    except TimeoutException:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n>>> Programme exceeded time limit of {max_iter_time} seconds. Terminating...")
        last_max_acc = max_acc_trace[-1] if max_acc_trace else 0.0
        # ========================== 修复：超时返回前也消毒 ==========================
        x = sanitize_genes(x)
        return last_max_acc, duration, final_uniq_rate, x

    if plot_results:
        plot_denoise(save_dir=save_dir, avg_acc_trace=avg_acc_trace, max_acc_trace=max_acc_trace,
                     valid_rate_trace=valid_rate_trace, uniq_rate_trace=uniq_rate_trace, seed=seed, dataset=dataset)
    
    end_time = time.time()
    if verbose:
        print(f"\n{'='*60}")
        print(f"[EvoDiff] Evolution completed")
        print(f"  - Final max accuracy: {max_acc_trace[-1]:.4f}")
        print(f"  - Total time: {end_time - start_time:.2f}s")
        print(f"  - Final uniqueness: {uniq_rate_trace[-1]:.4f}")
        print(f"{'='*60}\n")
    
    # ========================== 修复：正常返回前消毒 ==========================
    x = sanitize_genes(x)
    return max_acc_trace[-1], end_time - start_time, uniq_rate_trace[-1], x

def evo_diff_meta(
    dataset,
    api,
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
    nb201_or_meta,
    max_iter_time,
    use_consistency_generator=True,
    teacher_model=None,
    lambda_kl=0.5,
    lambda_ce=1.0,
    lambda_consistency=0.5,
    consistency_type='mse',
    perturb_scale=0.01,
    tournament_size=5,
    history_max_len=10,
    verbose=True,
    print_freq=10,
):
    assert nb201_or_meta == "meta", "nb201_or_meta should be meta, but got {}".format(nb201_or_meta)
    is_single_step = (num_step == 1)
    if verbose:
        print(f"\n{'='*60}")
        print(f"[EvoDiff Meta] Starting evolution process")
        print(f"  - Dataset: {dataset}")
        print(f"  - Total steps: {num_step}")
        print(f"  - Population size: {population_num}")
        print(f"  - Using ConsistencyGenerator: {use_consistency_generator}")
        print(f"  - Single-step mode: {is_single_step}")
        
        # ========================== 新增：ConsistencyGenerator 参数详细打印 (Meta版) ==========================
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
        # ============================================================================================================
        
        print(f"{'='*60}\n")
    
    x = torch.randn(population_num, geno_shape[0] * geno_shape[1])
    x_prev = copy.deepcopy(x)
    avg_acc_trace = []
    max_acc_trace = []
    valid_rate_trace = []
    uniq_rate_trace = []
    history_x = [x.clone()]
    history_fitness = []
    scheduler = DDIMSchedulerCosine(num_step=num_step)
    mapping_fn = Energy(temperature=temperature)
    
    # 加载Meta模型 (保持原样)
    nasbench201 = torch.load("meta_acc_predictor/data/nasbench201.pt")
    graph_config = load_graph_config(graph_data_name="nasbench201", nvt=7, data_path="meta_acc_predictor/data/nasbench201.pt")
    meta_surrogate_unnoised_model = MetaSurrogateUnnoisedModel(nvt=7, hs=512, nz=56, num_sample=20, graph_config=graph_config)
    meta_surrogate_unnoised_model = load_model(model=meta_surrogate_unnoised_model, ckpt_path="meta_acc_predictor/unnoised_checkpoint.pth.tar")
    fitness_restorer = FitnessRestorer(dataset_name=dataset, num_sample=20, seed=seed)
    
    start_time = time.time()
    iter_start_time = time.time()
    bar = tqdm.tqdm(scheduler, ncols=120)
    final_uniq_rate = 0.0
    
    try:
        for t, alpha in bar:
            if time.time() - iter_start_time > max_iter_time:
                raise TimeoutException
            iter_start_time = time.time()
            
            # 计算适应度
            accurancy, fitness, valid_rate = meta_arch_fitness(
                operation_matrix=x, api=api, dataset=dataset, test_dataset=None,
                meta_surrogate_unnoised_model=meta_surrogate_unnoised_model, nasbench201=nasbench201,
                fitness_restorer=fitness_restorer
            )
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
            
            # 生成器
            if use_consistency_generator:
                generator = ConsistencyGenerator(
                    x=x, fitness=fitness, alpha=alpha, density='uniform', h=0.1,
                    elite_strategy=False, teacher_model=teacher_model, lambda_kl=lambda_kl,
                    lambda_ce=lambda_ce, lambda_consistency=lambda_consistency,
                    consistency_type=consistency_type, perturb_scale=perturb_scale,
                    arch_fitness_fn=meta_arch_fitness, api=api, dataset=dataset,
                    tournament_size=tournament_size, history_max_len=history_max_len,
                    history_x=history_x, history_fitness=history_fitness,
                    verbose=verbose and (t == 0), print_freq=print_freq, is_single_step=is_single_step,
                    test_dataset=None,meta_surrogate_unnoised_model=meta_surrogate_unnoised_model,
                    nasbench201=nasbench201,fitness_restorer=fitness_restorer
                )
            else:
                generator = BayesianGenerator(
                    x=x, fitness=fitness, alpha=alpha, density='uniform', h=0.1,
                    elite_strategy=False, verbose=verbose and (t == 0)
                )
            
            x = generator.generate(x=x, noise=noise_scale, elite_rate=elite_rate)
            
            # Corrector
            if not is_single_step:
                if t != num_step - 1:
                    x = mutate(population=population_num, x=x, mut_rate=mutate_rate, eta=mutate_distri_index)
                    x = select(x_prev=x_prev, x_next=x, fitness_prev=fitness, elite_rate=elite_rate,
                               diver_rate=diver_rate, api=api, dataset=dataset, max_iter_time=max_iter_time,
                               nb201_or_meta=nb201_or_meta, test_dataset=None, meta_surrogate_unnoised_model=meta_surrogate_unnoised_model,
                               nasbench201=nasbench201, fitness_restorer=fitness_restorer)
                    x_prev = copy.deepcopy(x)
                else:
                    x = mutate(population=population_num, x=x, mut_rate=mutate_rate, eta=math.ceil(mutate_distri_index * 1.5))
                    x = select(x_prev=x_prev, x_next=x, fitness_prev=fitness, elite_rate=elite_rate,
                               diver_rate=diver_rate, api=api, dataset=dataset, max_iter_time=max_iter_time,
                               nb201_or_meta=nb201_or_meta, test_dataset=None, meta_surrogate_unnoised_model=meta_surrogate_unnoised_model,
                               nasbench201=nasbench201, fitness_restorer=fitness_restorer)
                    x_prev = copy.deepcopy(x)
            else:
                x = mutate(population=population_num, x=x, mut_rate=min(mutate_rate * 0.5, 0.1), eta=mutate_distri_index)
                x_prev = copy.deepcopy(x)
            
            uniq_rate = compute_uniqueness(arch_op_matrices=x)
            final_uniq_rate = uniq_rate
            avg_acc_trace.append(avg_acc)
            max_acc_trace.append(max_acc)
            valid_rate_trace.append(valid_rate)
            uniq_rate_trace.append(uniq_rate)
            bar.set_postfix({"max_pred_acc": f"{max_acc:.2f}", "avg_pred_acc": f"{avg_acc:.2f}", "valid_rate": f"{valid_rate:.2f}", "uniq_rate": f"{uniq_rate:.2f}"})

    except TimeoutException:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n>>> Programme exceeded time limit of {max_iter_time} seconds. Terminating...")
        last_max_acc = max_acc_trace[-1] if max_acc_trace else 0.0
        # ========================== 修复：Meta超时返回前也消毒 ==========================
        x = sanitize_genes(x)
        return last_max_acc, duration, final_uniq_rate, x

    end_time = time.time()
    if plot_results:
        plot_denoise(save_dir=save_dir, avg_acc_trace=avg_acc_trace, max_acc_trace=max_acc_trace,
                     valid_rate_trace=valid_rate_trace, uniq_rate_trace=uniq_rate_trace, seed=seed, dataset=dataset)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[EvoDiff Meta] Evolution completed")
        print(f"  - Final max predicted accuracy: {max_acc_trace[-1]:.4f}")
        print(f"  - Total time: {end_time - start_time:.2f}s")
        print(f"  - Final uniqueness: {uniq_rate_trace[-1]:.4f}")
        print(f"{'='*60}\n")
    
    # ========================== 修复：Meta正常返回前消毒 ==========================
    x = sanitize_genes(x)
    return max_acc_trace[-1], end_time - start_time, uniq_rate_trace[-1], x

'''