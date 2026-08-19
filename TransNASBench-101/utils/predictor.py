import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.corrector import normalize

# ====================== 基础 DDIM 函数 ======================
def ddim_step(xt, x0, alphas: tuple, noise: float = None):
    alphat, alphatp = alphas
    sigma = ddpm_sigma(alphat, alphatp) * noise if noise is not None else 0.0
    eps = (xt - (alphat ** 0.5) * x0) / ((1.0 - alphat) ** 0.5 + 1e-10)
    if sigma is None:
        sigma = ddpm_sigma(alphat, alphatp)
    x_next = (alphatp ** 0.5) * x0 + ((1 - alphatp - sigma ** 2) ** 0.5) * eps + sigma * torch.randn_like(x0)
    return x_next

def ddpm_sigma(alphat, alphatp):
    return ((1 - alphatp) / (1 - alphat) * (1 - alphat / alphatp)) ** 0.5 if alphat > 0 else 0.0

# ====================== 贝叶斯估计器 ======================
class BayesianEstimator:
    def __init__(self, x: torch.tensor, fitness: torch.tensor, alpha, density='uniform', h=0.1):
        self.x = x
        self.fitness = fitness
        self.alpha = alpha
        self.density_method = density
        self.h = h
        if density not in ['uniform']:
            raise NotImplementedError(f'Density estimator {density} is not implemented.')

    def append(self, estimator):
        self.x = torch.cat([self.x, estimator.x], dim=0)
        self.fitness = torch.cat([self.fitness, estimator.fitness], dim=0)

    def density(self, x):
        if self.density_method == 'uniform':
            return torch.ones(x.shape[0], device=x.device) / x.shape[0]

    @staticmethod
    def norm(x):
        if x.shape[-1] == 1:
            return torch.abs(x).squeeze(-1)
        else:
            return torch.norm(x, dim=-1)

    def gaussian_prob(self, x, mu, sigma):
        dist = self.norm(x - mu)
        return torch.exp(-(dist ** 2) / (2 * sigma ** 2))

    def _estimate(self, x_t, p_x_t):
        mu = self.x * (self.alpha ** 0.5)
        sigma = (1 - self.alpha) ** 0.5
        p_diffusion = self.gaussian_prob(x_t, mu, sigma)
        prob = (self.fitness + 1e-9) * (p_diffusion + 1e-9) / (p_x_t + 1e-9)
        z = torch.sum(prob)
        origin = torch.sum(prob.unsqueeze(1) * self.x, dim=0) / (z + 1e-9)
        return origin

    def estimate(self, x_t):
        p_x_t = self.density(x_t)
        origin = torch.vmap(self._estimate, (0, 0))(x_t, p_x_t)
        return origin

    def __call__(self, x_t):
        return self.estimate(x_t)

    def __repr__(self):
        return f'<BayesianEstimator {len(self.x)} samples>'

# ====================== 基线贝叶斯生成器 ======================
class BayesianGenerator:
    def __init__(self, x, fitness, alpha, density='uniform', h=0.1, elite_strategy=False, verbose=True):
        self.x = x
        self.fitness = fitness
        self.elite_strategy = elite_strategy
        self.alpha, self.alpha_past = alpha
        self.estimator = BayesianEstimator(self.x, self.fitness, self.alpha, density=density, h=h)
        if verbose:
            print(f"\n[Generator Info] Using BayesianGenerator (baseline)")

    def generate(self, x, noise, elite_rate, return_x0=False):
        x0_est = self.estimator(x)
        x_next = ddim_step(xt=x, x0=x0_est, alphas=(self.alpha, self.alpha_past), noise=noise)
        x_next = normalize(x_next)
        if return_x0:
            return x_next, x0_est
        else:
            return x_next

    def __call__(self, noise=1.0, return_x0=False):
        return self.generate(noise=noise, return_x0=return_x0)


# ====================== 一致性生成器（包含 SATS + EMA + 教师扰动） ======================
class ConsistencyGenerator:
    """
    完整 CDENAG 生成器，包含 SATS、教师扰动和 EMA 混合，适配 TransNASBench-101。
    """
    def __init__(self, x, fitness, alpha, density='uniform', h=0.1, elite_strategy=False,
                 teacher_model=None, lambda_kl=0.5, lambda_ce=1.0, lambda_consistency=1.0,
                 consistency_type='mse', perturb_scale=0.1, arch_fitness_fn=None, api=None,
                 task=None, search_space=None,
                 tournament_size=5, history_max_len=100, history_x=None, history_fitness=None,
                 verbose=True, print_freq=10, is_single_step=False,
                 stability_perturb_num=5, stability_perturb_scale=1e-3,
                 **fitness_kwargs):      # 接收额外参数（如适应度函数可能需要）
        # 基础参数
        self.x = x
        self.fitness = fitness
        self.elite_strategy = elite_strategy
        self.alpha, self.alpha_past = alpha
        self.density = density
        self.h = h

        # 一致性参数
        self.teacher_model = teacher_model.eval() if teacher_model is not None else None
        self.lambda_kl = lambda_kl
        self.lambda_ce = lambda_ce
        self.lambda_consistency = lambda_consistency
        self.consistency_type = consistency_type
        self.perturb_scale = perturb_scale

        # TransNASBench-101 适配
        self.arch_fitness_fn = arch_fitness_fn
        self.api = api
        self.task = task
        self.search_space = search_space
        self.fitness_kwargs = fitness_kwargs

        # 历史数据与锦标赛选择
        self.tournament_size = tournament_size
        self.history_max_len = history_max_len
        self.history_x = history_x if history_x is not None else [x.clone()]
        self.history_fitness = history_fitness if history_fitness is not None else [fitness.clone()]

        self.is_single_step = is_single_step
        self.verbose = verbose
        self.print_freq = print_freq
        self.call_count = 0
        self.best_teacher_fitness = -float('inf')

        # SATS 参数
        self.stability_perturb_num = stability_perturb_num
        self.stability_perturb_scale = stability_perturb_scale

        # 贝叶斯估计器
        self.estimator = BayesianEstimator(self.x, self.fitness, self.alpha, density=density, h=h)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[Generator Info] Using ConsistencyGenerator (EMA + Perturb + SATS) for TransNAS-Bench-101")
            print(f"  - Task: {self.task}, Search space: {self.search_space}")
            print(f"  - Lambda consistency: {self.lambda_consistency}")
            print(f"  - Tournament size: {self.tournament_size}")
            print(f"  - Perturb scale: {self.perturb_scale}")
            print(f"  - Stability perturbations: {self.stability_perturb_num}, scale: {self.stability_perturb_scale}")
            print(f"  - History length (current): {len(self.history_x)}")
            print(f"  - Single-step mode: {self.is_single_step}")
            print(f"{'='*60}\n")

    def _compute_stability_single(self, x_single):
        """
        对单个架构计算稳定性得分（适配 TransNASBench-101 适应度函数）。
        arch_fitness_fn 需传入 operation_matrix, api, task, search_space，返回 (accuracy, fitness) 或更多。
        """
        if self.arch_fitness_fn is None or self.stability_perturb_num <= 0:
            return 1.0

        perturbed_fitness = []
        for _ in range(self.stability_perturb_num):
            x_pert = x_single + torch.randn_like(x_single) * self.stability_perturb_scale
            x_pert = normalize(x_pert)
            # TransNASBench-101 调用: arch_fitness(operation_matrix=x_pert, api=api, task=task, search_space=search_space)
            out = self.arch_fitness_fn(operation_matrix=x_pert, api=self.api, task=self.task, search_space=self.search_space)
            # 获取适应度值（通常是第二个返回值）
            if isinstance(out, tuple):
                fit_val = out[1]   # fitness 在第二位
            else:
                fit_val = out
            perturbed_fitness.append(fit_val.item() if hasattr(fit_val, 'item') else fit_val)

        fits = torch.tensor(perturbed_fitness, device=x_single.device)
        mean_fit = fits.mean()
        std_fit = fits.std()
        if mean_fit <= 0:
            return 0.0
        cv = std_fit / (mean_fit + 1e-9)
        return 1.0 / (1.0 + cv)

    def tournament_select(self):
        """
        SATS: 稳定性感知锦标赛选择。
        综合得分 = 归一化适应度 × 稳定性得分，选最高者作为教师。
        """
        all_history_x = torch.cat(self.history_x, dim=0)
        all_history_fitness = torch.cat(self.history_fitness, dim=0)
        num_total = all_history_x.shape[0]
        if num_total == 0:
            raise ValueError("History population is empty for tournament selection")

        k = min(self.tournament_size, num_total)
        if num_total <= self.tournament_size:
            selected_indices = torch.arange(num_total, device=all_history_x.device)
        else:
            selected_indices = torch.randint(0, num_total, (k,), device=all_history_x.device)

        candidates_x = all_history_x[selected_indices]
        candidates_fitness = all_history_fitness[selected_indices]

        # 计算稳定性得分
        if self.arch_fitness_fn is not None and self.stability_perturb_num > 0:
            stability_scores = torch.zeros(k, device=candidates_x.device)
            for i in range(k):
                stability_scores[i] = self._compute_stability_single(candidates_x[i:i+1])
        else:
            stability_scores = torch.ones(k, device=candidates_x.device)

        # 归一化适应度
        f_min, f_max = candidates_fitness.min(), candidates_fitness.max()
        if f_max - f_min > 1e-9:
            norm_fitness = (candidates_fitness - f_min) / (f_max - f_min)
        else:
            norm_fitness = torch.ones_like(candidates_fitness)

        # 综合得分
        comprehensive = norm_fitness * stability_scores
        best_local_idx = torch.argmax(comprehensive)
        best_idx = selected_indices[best_local_idx]

        if self.verbose and self.call_count % self.print_freq == 0:
            best_fit = all_history_fitness[best_idx].item()
            best_stab = stability_scores[best_local_idx].item()
            best_score = comprehensive[best_local_idx].item()
            print(f"\n  [SATS] Step {self.call_count}")
            print(f"    - Candidates: {k}/{num_total}, Task: {self.task}")
            print(f"    - Teacher fitness: {best_fit:.4f}, stability: {best_stab:.4f}, score: {best_score:.4f}")
            if best_fit > self.best_teacher_fitness:
                self.best_teacher_fitness = best_fit
                print(f"    [NEW BEST] Global best fitness updated!")

        return all_history_x[best_idx:best_idx+1], all_history_fitness[best_idx:best_idx+1]

    def _ddim_step(self, x_t, x0_pred, noise=1.0):
        if self.is_single_step:
            return normalize(x0_pred)
        else:
            return ddim_step(xt=x_t, x0=x0_pred, alphas=(self.alpha, self.alpha_past), noise=noise)

    def generate(self, x, noise, elite_rate, return_x0=False):
        self.call_count += 1

        # 1. 贝叶斯估计
        x0_est = self.estimator(x)

        # 2. 一致性引导判别
        apply_guidance = (
            len(self.history_x) > 1 and
            self.lambda_consistency > 0 and
            not self.is_single_step
        )

        if apply_guidance:
            # 3. 锦标赛选择教师
            teacher_x, _ = self.tournament_select()

            # ---- 教师扰动 ----
            if self.perturb_scale > 0:
                perturb_noise = torch.randn_like(teacher_x) * self.perturb_scale
                teacher_x = teacher_x + perturb_noise
                teacher_x = normalize(teacher_x)
                if self.verbose and self.call_count % self.print_freq == 0:
                    print(f"    [Perturb] Teacher perturbed with scale {self.perturb_scale:.3f}")

            teacher_x_expanded = teacher_x.expand_as(x0_est)

            # 4. EMA 混合
            x0_est = (1.0 - self.lambda_consistency) * x0_est + self.lambda_consistency * teacher_x_expanded
            x0_est = normalize(x0_est)

            if self.verbose and self.call_count % self.print_freq == 0:
                print(f"    [EMA Mix] lambda_consistency={self.lambda_consistency:.3f} applied (history_len={len(self.history_x)})")
        else:
            if self.verbose and self.call_count == 1 and len(self.history_x) <= 1:
                print(f"    [Warmup] First generation: collecting initial history, skipping guidance")

        # 5. DDIM 采样
        x_next = self._ddim_step(x_t=x, x0_pred=x0_est, noise=noise)
        x_next = normalize(x_next)

        if return_x0:
            return x_next, x0_est
        else:
            return x_next

    def __call__(self, noise=1.0, return_x0=False):
        return self.generate(noise=noise, return_x0=return_x0)


'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.corrector import normalize

# ====================== 基础 DDIM 函数（保持原有逻辑） ======================
def ddim_step(xt, x0, alphas: tuple, noise: float = None):
    alphat, alphatp = alphas
    sigma = ddpm_sigma(alphat, alphatp) * noise if noise is not None else 0.0
    eps = (xt - (alphat ** 0.5) * x0) / ((1.0 - alphat) ** 0.5 + 1e-10)
    if sigma is None:
        sigma = ddpm_sigma(alphat, alphatp)
    x_next = (alphatp ** 0.5) * x0 + ((1 - alphatp - sigma ** 2) ** 0.5) * eps + sigma * torch.randn_like(x0)
    return x_next

def ddpm_sigma(alphat, alphatp):
    return ((1 - alphatp) / (1 - alphat) * (1 - alphat / alphatp)) ** 0.5 if alphat > 0 else 0.0

# ====================== 贝叶斯估计器（保持原有逻辑） ======================
class BayesianEstimator:
    def __init__(self, x: torch.tensor, fitness: torch.tensor, alpha, density='uniform', h=0.1):
        self.x = x
        self.fitness = fitness
        self.alpha = alpha
        self.density_method = density
        self.h = h
        if not density in ['uniform']:
            raise NotImplementedError(f'Density estimator {density} is not implemented.')
    
    def append(self, estimator):
        self.x = torch.cat([self.x, estimator.x], dim=0)
        self.fitness = torch.cat([self.fitness, estimator.fitness], dim=0)
    
    def density(self, x):
        if self.density_method == 'uniform':
            return torch.ones(x.shape[0]) / x.shape[0]
    
    @staticmethod
    def norm(x):
        if x.shape[-1] == 1:
            return torch.abs(x).squeeze(-1)
        else:
            return torch.norm(x, dim=-1)
    
    def gaussian_prob(self, x, mu, sigma):
        dist = self.norm(x - mu)
        return torch.exp(-(dist ** 2) / (2 * sigma ** 2))
    
    def _estimate(self, x_t, p_x_t):
        mu = self.x * (self.alpha ** 0.5)
        sigma = (1 - self.alpha) ** 0.5
        p_diffusion = self.gaussian_prob(x_t, mu, sigma)
        prob = (self.fitness + 1e-9) * (p_diffusion + 1e-9) / (p_x_t + 1e-9)
        z = torch.sum(prob)
        origin = torch.sum(prob.unsqueeze(1) * self.x, dim=0) / (z + 1e-9)
        return origin
    
    def estimate(self, x_t):
        p_x_t = self.density(x_t)
        origin = torch.vmap(self._estimate, (0, 0))(x_t, p_x_t)
        return origin
    
    def __call__(self, x_t):
        return self.estimate(x_t)
    
    def __repr__(self):
        return f'<BayesianEstimator {len(self.x)} samples>'

# ====================== 基线贝叶斯生成器（保持原有逻辑） ======================
class BayesianGenerator:
    def __init__(self, x, fitness, alpha, density='uniform', h=0.1, elite_strategy=False, verbose=True):
        self.x = x
        self.fitness = fitness
        self.elite_strategy = elite_strategy
        self.alpha, self.alpha_past = alpha
        self.estimator = BayesianEstimator(self.x, self.fitness, self.alpha, density=density, h=h)
        if verbose:
            print(f"\n[Generator Info] Using BayesianGenerator (baseline)")
    
    def generate(self, x, noise, elite_rate, return_x0=False):
        x0_est = self.estimator(x)
        x_next = ddim_step(xt=x, x0=x0_est, alphas=(self.alpha, self.alpha_past), noise=noise)
        x_next = normalize(x_next)
        if return_x0:
            return x_next, x0_est
        else:
            return x_next
    
    def __call__(self, noise=1.0, return_x0=False):
        return self.generate(noise=noise, return_x0=return_x0)

# ====================== 新增：复用 NAS-Bench-201 的 ConsistencyGenerator ======================
class ConsistencyGenerator:
    """
    完全复用 NAS-Bench-201 的 ConsistencyGenerator，仅适配 TransNASBench-101 的 arch_fitness 调用
    """
    def __init__(self, x, fitness, alpha, density='uniform', h=0.1, elite_strategy=False,
                 teacher_model=None, lambda_kl=0.5, lambda_ce=1.0, lambda_consistency=1.0,
                 consistency_type='mse', perturb_scale=0.1, arch_fitness_fn=None, api=None, task=None, search_space=None,
                 tournament_size=5, history_max_len=100, history_x=None, history_fitness=None,
                 verbose=True, print_freq=10, is_single_step=False):
        
        # 基础参数
        self.x = x
        self.fitness = fitness
        self.elite_strategy = elite_strategy
        self.alpha, self.alpha_past = alpha
        self.density = density
        self.h = h
        
        # Consistency 参数
        self.teacher_model = teacher_model.eval() if teacher_model is not None else None
        self.lambda_kl = lambda_kl
        self.lambda_ce = lambda_ce
        self.lambda_consistency = lambda_consistency
        self.consistency_type = consistency_type
        self.perturb_scale = perturb_scale
        
        # TransNASBench-101 适配参数（替换原 NAS-Bench-201 的 dataset/api）
        self.arch_fitness_fn = arch_fitness_fn
        self.api = api
        self.task = task
        self.search_space = search_space
        
        # 锦标赛选择参数
        self.tournament_size = tournament_size
        self.history_max_len = history_max_len
        self.history_x = history_x if history_x is not None else [x.clone()]
        self.history_fitness = history_fitness if history_fitness is not None else [fitness.clone()]
        
        self.is_single_step = is_single_step
        self.verbose = verbose
        self.print_freq = print_freq
        self.call_count = 0
        self.best_teacher_fitness = -float('inf')
        
        self.mse_loss = nn.MSELoss()
        self.estimator = BayesianEstimator(self.x, self.fitness, self.alpha, density=density, h=h)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[Generator Info] Using ConsistencyGenerator (TransNASBench-101)")
            print(f"  - Tournament size: {self.tournament_size}")
            print(f"  - Lambda consistency: {self.lambda_consistency}")
            print(f"  - Single-step mode: {self.is_single_step}")
            print(f"{'='*60}\n")
    
    def tournament_select(self):
        """锦标赛选择：直接返回历史适应度最高的样本（复用 NAS-Bench-201 逻辑）"""
        all_history_x = torch.cat(self.history_x, dim=0)
        all_history_fitness = torch.cat(self.history_fitness, dim=0)
        
        num_total = all_history_x.shape[0]
        if num_total == 0:
            raise ValueError("History population is empty")
        
        best_idx = torch.argmax(all_history_fitness)
        
        if self.verbose and self.call_count % self.print_freq == 0:
            current_best_fit = all_history_fitness[best_idx].item()
            print(f"\n  [Elite Select] Step {self.call_count}")
            print(f"    - History size: {num_total}")
            print(f"    - Teacher fitness: {current_best_fit:.4f}")
            if current_best_fit > self.best_teacher_fitness:
                self.best_teacher_fitness = current_best_fit
                print(f"    [NEW BEST] Elite teacher updated!")
        
        return all_history_x[best_idx:best_idx+1], all_history_fitness[best_idx:best_idx+1]
    
    def generate(self, x, noise, elite_rate, return_x0=False):
        self.call_count += 1
        
        # 1. 贝叶斯估计
        x0_est = self.estimator(x)
        
        # 2. 判断是否应用引导
        apply_guidance = (self.call_count > 1) and (self.lambda_consistency > 0) and (not self.is_single_step)
        
        if apply_guidance:
            teacher_x, _ = self.tournament_select()
            
            # 无梯度 EMA 融合（复用 NAS-Bench-201 逻辑）
            momentum = self.lambda_consistency * 0.5 
            teacher_x_expanded = teacher_x.expand_as(x0_est)
            x0_est = x0_est * (1.0 - momentum) + teacher_x_expanded * momentum
            x0_est = normalize(x0_est)
            
            if self.verbose and self.call_count % self.print_freq == 0:
                print(f"    [EMA Mix] Applied momentum: {momentum:.3f}")
        else:
            if self.verbose and self.call_count == 1:
                print(f"    [Warmup] Step 1: Collecting history...")
        
        # 3. DDIM 采样
        x_next = ddim_step(xt=x, x0=x0_est, alphas=(self.alpha, self.alpha_past), noise=noise)
        x_next = normalize(x_next)
        
        if return_x0:
            return x_next, x0_est
        else:
            return x_next
    
    def __call__(self, noise=1.0, return_x0=False):
        return self.generate(noise=noise, return_x0=return_x0)
'''



'''
import torch
from utils.corrector import normalize

def ddim_step(xt, x0, alphas: tuple, noise: float = None):
    """One step of the DDIM algorithm.

    Args:
    - xt: torch.Tensor, shape (n, d), the current samples.
    - x0: torch.Tensor, shape (n, d), the estimated origin.
    - alphas: tuple of two floats, alpha_{t} and alpha_{t-1}.

    Returns:
    - x_next: torch.Tensor, shape (n, d), the next samples.
    """
    alphat, alphatp = alphas
    # 计算噪声强度sigma
    sigma = ddpm_sigma(alphat, alphatp) * noise
    # 执行一步DDIM的反向去噪迭代
    eps = (xt - (alphat ** 0.5) * x0) / (1.0 - alphat) ** 0.5
    if sigma is None:
        sigma = ddpm_sigma(alphat, alphatp)
    x_next = (alphatp ** 0.5) * x0 + ((1 - alphatp - sigma ** 2) ** 0.5) * eps + sigma * torch.randn_like(x0)
    return x_next


def ddpm_sigma(alphat, alphatp):
    """Compute the default sigma for the DDPM algorithm."""
    return ((1 - alphatp) / (1 - alphat) * (1 - alphat / alphatp)) ** 0.5


class BayesianEstimator:
    """Bayesian Estimator of the origin points, based on current samples and fitness values."""
    def __init__(self, x: torch.tensor, fitness: torch.tensor, alpha, density='uniform', h=0.1):
        self.x = x
        self.fitness = fitness
        self.alpha = alpha
        self.density_method = density
        self.h = h
        if not density in ['uniform']:
            raise NotImplementedError(f'Density estimator {density} is not implemented.')

    def append(self, estimator):
        # 将另一个估计器的样本数据和适应度值拼接到当前估计器中
        self.x = torch.cat([self.x, estimator.x], dim=0)
        self.fitness = torch.cat([self.fitness, estimator.fitness], dim=0)
    
    def density(self, x):
        # 根据不同的概率分布，计算概率密度值
        if self.density_method == 'uniform':
            return torch.ones(x.shape[0]) / x.shape[0]
    
    @staticmethod
    def norm(x):
        # 计算向量的范数
        if x.shape[-1] == 1:
            # for some reason, torch.norm become very slow when dim=1, so we use torch.abs instead
            return torch.abs(x).squeeze(-1)
        else:
            return torch.norm(x, dim=-1)

    def gaussian_prob(self, x, mu, sigma):
        # 计算高斯分布的概率密度值
        dist = self.norm(x - mu)
        return torch.exp(-(dist ** 2) / (2 * sigma ** 2))

    def _estimate(self, x_t, p_x_t):
        # diffusion proability, P = N(x_t; \sqrt{α_t}x,\sqrt{1-α_t})
        mu = self.x * (self.alpha ** 0.5)   # 均值
        sigma = (1 - self.alpha) ** 0.5     # 标准差
        p_diffusion = self.gaussian_prob(x_t, mu, sigma)
        # 通过概率和适应度值估计原始样本（+1e-9是为了防止为0出错）
        prob = (self.fitness + 1e-9) * (p_diffusion + 1e-9) / (p_x_t + 1e-9)
        z = torch.sum(prob)
        origin = torch.sum(prob.unsqueeze(1) * self.x, dim=0) / (z + 1e-9)
        return origin

    def estimate(self, x_t):
        p_x_t = self.density(x_t)   # 计算概率密度值
        origin = torch.vmap(self._estimate, (0, 0))(x_t, p_x_t) #进行向量化
        return origin

    def __call__(self, x_t):
        return self.estimate(x_t)

    def __repr__(self):
        return f'<BayesianEstimator {len(self.x)} samples>'


class BayesianGenerator:
    """Bayesian Generator for the DDIM algorithm."""
    def __init__(self, x, fitness, alpha, density='uniform', h=0.1, elite_strategy=False):
        self.x = x
        self.fitness = fitness
        self.elite_strategy = elite_strategy
        self.alpha, self.alpha_past = alpha
        self.estimator = BayesianEstimator(self.x, self.fitness, self.alpha, density=density, h=h)
    
    def generate(self, x, noise, elite_rate, return_x0=False):
        # 通过当前时间步的样本和适应度值，估计初始点的样本
        x0_est = self.estimator(x)
        # 执行一次DDIM的反向去噪迭代
        x_next = ddim_step(xt=x, x0=x0_est, alphas=(self.alpha, self.alpha_past), noise=noise)
        # 正则化
        x_next = normalize(x_next)
        if return_x0:
            return x_next, x0_est
        else:
            return x_next

    def __call__(self, noise=1.0, return_x0=False):
        return self.generate(noise=noise, return_x0=return_x0)

'''