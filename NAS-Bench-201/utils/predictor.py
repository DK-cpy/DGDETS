import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.corrector import normalize

# ====================== 基础DDIM函数 ======================
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


# ====================== 一致性生成器（EMA混合版 + 稳定性感知锦标赛） ======================
class ConsistencyGenerator:
    """
    通过 EMA 混合引入历史最佳架构的引导。
    关键改动：
    1. 稳定性感知锦标赛选择（SATS）：综合 归一化适应度 × 稳定性得分 选择教师。
    2. 教师扰动：教师架构在混合前先加上标准差为 perturb_scale 的高斯噪声，
       再归一化，以鼓励探索、避免过早收敛。
    """
    def __init__(self, x, fitness, alpha, density='uniform', h=0.1, elite_strategy=False,
                 teacher_model=None, lambda_kl=0.5, lambda_ce=1.0, lambda_consistency=1.0,
                 consistency_type='mse', perturb_scale=0.1, arch_fitness_fn=None, api=None, dataset=None,
                 tournament_size=5, history_max_len=100, history_x=None, history_fitness=None,
                 verbose=True, print_freq=10, is_single_step=False,
                 stability_perturb_num=5, stability_perturb_scale=1e-3,
                 **fitness_kwargs):      # 接收适应度函数额外参数（如 meta 流程所需）
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

        # NAS适配
        self.arch_fitness_fn = arch_fitness_fn
        self.api = api
        self.dataset = dataset
        self.fitness_kwargs = fitness_kwargs   # 存储额外参数（如 meta_surrogate_unnoised_model 等）

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

        # 损失函数（未使用）
        self.ce_loss = nn.CrossEntropyLoss() if teacher_model is not None else None
        self.kl_loss = nn.KLDivLoss(reduction='batchmean') if teacher_model is not None else None
        self.mse_loss = nn.MSELoss()

        # 贝叶斯估计器
        self.estimator = BayesianEstimator(self.x, self.fitness, self.alpha, density=density, h=h)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[Generator Info] Using ConsistencyGenerator (EMA + Perturb + SATS)")
            print(f"  - Lambda consistency: {self.lambda_consistency}")
            print(f"  - Tournament size: {self.tournament_size}")
            print(f"  - Perturb scale: {self.perturb_scale}")
            print(f"  - Stability perturbations: {self.stability_perturb_num}, scale: {self.stability_perturb_scale}")
            print(f"  - History length (current): {len(self.history_x)}")
            print(f"  - Single-step mode: {self.is_single_step}")
            print(f"{'='*60}\n")

    def _compute_stability_single(self, x_single):
        """
        对单个架构计算稳定性得分。
        施加 stability_perturb_num 次微小扰动，计算扰动后适应度的变异系数 (CV)，
        得分 = 1 / (1 + CV)，越稳定得分越接近 1。
        """
        if self.arch_fitness_fn is None or self.stability_perturb_num <= 0:
            return 1.0  # 无法评估时默认满分，即不惩罚

        perturbed_fitness = []
        for _ in range(self.stability_perturb_num):
            x_pert = x_single + torch.randn_like(x_single) * self.stability_perturb_scale
            x_pert = normalize(x_pert)
            _, fit, _ = self.arch_fitness_fn(
                operation_matrix=x_pert,
                api=self.api,
                dataset=self.dataset,
                **self.fitness_kwargs   # 传递额外参数（如 meta 流程所需的 test_dataset 等）
            )
            perturbed_fitness.append(fit.item())

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
            if self.verbose and self.call_count % self.print_freq == 0:
                print(f"  [SATS] History size ({num_total}) <= tournament size, using all individuals")
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

        # 归一化适应度，避免量纲影响综合得分
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
            print(f"    - Candidates: {k}/{num_total}")
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

# ====================== 基础DDIM函数 ======================
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

# ====================== 一致性生成器（EMA混合版） ======================
class ConsistencyGenerator:
    """
    通过 EMA 混合引入历史最佳架构的引导。
    lambda_consistency 控制向教师架构的混合比例。
    利用 history_x 长度自动跳过第一步。
    锦标赛选择：直接从历史中选取 fitness 最高的个体作为教师（原始版本逻辑）。
    """
    def __init__(self, x, fitness, alpha, density='uniform', h=0.1, elite_strategy=False,
                 teacher_model=None, lambda_kl=0.5, lambda_ce=1.0, lambda_consistency=1.0,
                 consistency_type='mse', perturb_scale=0.1, arch_fitness_fn=None, api=None, dataset=None,
                 tournament_size=5, history_max_len=100, history_x=None, history_fitness=None,
                 verbose=True, print_freq=10, is_single_step=False,
                 stability_perturb_num=5, stability_perturb_scale=1e-3):
        # 基础参数
        self.x = x
        self.fitness = fitness
        self.elite_strategy = elite_strategy
        self.alpha, self.alpha_past = alpha
        self.density = density
        self.h = h

        # 一致性参数（teacher_model、kl、ce 当前未使用）
        self.teacher_model = teacher_model.eval() if teacher_model is not None else None
        self.lambda_kl = lambda_kl
        self.lambda_ce = lambda_ce
        self.lambda_consistency = lambda_consistency
        self.consistency_type = consistency_type
        self.perturb_scale = perturb_scale

        # NAS适配
        self.arch_fitness_fn = arch_fitness_fn
        self.api = api
        self.dataset = dataset

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

        # 损失函数（未使用）
        self.ce_loss = nn.CrossEntropyLoss() if teacher_model is not None else None
        self.kl_loss = nn.KLDivLoss(reduction='batchmean') if teacher_model is not None else None
        self.mse_loss = nn.MSELoss()

        # 贝叶斯估计器
        self.estimator = BayesianEstimator(self.x, self.fitness, self.alpha, density=density, h=h)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[Generator Info] Using ConsistencyGenerator (EMA Mix Version)")
            print(f"  - Lambda consistency: {self.lambda_consistency}")
            print(f"  - Tournament size: {self.tournament_size}")
            print(f"  - History length (current): {len(self.history_x)}")
            print(f"  - Single-step mode: {self.is_single_step}")
            print(f"{'='*60}\n")

    def tournament_select(self):
        """直接从历史中选取 fitness 最高的个体作为教师（原始逻辑）"""
        all_history_x = torch.cat(self.history_x, dim=0)
        all_history_fitness = torch.cat(self.history_fitness, dim=0)
        best_idx = torch.argmax(all_history_fitness)

        if self.verbose and self.call_count % self.print_freq == 0:
            current_best_fit = all_history_fitness[best_idx].item()
            if current_best_fit > self.best_teacher_fitness:
                self.best_teacher_fitness = current_best_fit
                print(f"\n  [Tournament] New elite teacher fitness: {current_best_fit:.4f}")

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

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.corrector import normalize

# ====================== 基础DDIM函数（完全兼容原生逻辑，无修改） ======================
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

# ====================== 贝叶斯估计器（完全兼容原生逻辑，无修改） ======================
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

# ====================== 基线贝叶斯生成器（完全兼容原生逻辑，无修改） ======================
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

# ====================== 零干预通用版ConsistencyGenerator ======================
class ConsistencyGenerator:
    """
    全搜索空间通用、零人为干预的Consistency Generator
    核心逻辑：仅靠「一致性稳定性」过滤预测器虚假高分，不添加任何人为先验、不限制搜索空间、不干预算子选择
    完全兼容原有所有调用接口，替换后直接运行
    """
    def __init__(self, x, fitness, alpha, density='uniform', h=0.1, elite_strategy=False,
                 teacher_model=None, lambda_kl=0.5, lambda_ce=1.0, lambda_consistency=1.0,
                 consistency_type='mse', perturb_scale=0.1, arch_fitness_fn=None, api=None, dataset=None,
                 tournament_size=5, history_max_len=100, history_x=None, history_fitness=None,
                 verbose=True, print_freq=10, is_single_step=False,
                 # 仅保留通用超参数，无任何搜索空间相关配置
                 stability_perturb_num=5,    # 稳定性测试的扰动次数
                 stability_perturb_scale=1e-3 # 稳定性测试的扰动尺度
                 ):
        # 基础兼容参数（完全保留原有逻辑）
        self.x = x
        self.fitness = fitness
        self.elite_strategy = elite_strategy
        self.alpha, self.alpha_past = alpha
        self.density = density
        self.h = h
        
        # Consistency核心参数
        self.teacher_model = teacher_model.eval() if teacher_model is not None else None
        self.lambda_kl = lambda_kl
        self.lambda_ce = lambda_ce
        self.lambda_consistency = lambda_consistency
        self.consistency_type = consistency_type
        self.perturb_scale = perturb_scale
        
        # 通用NAS适配参数（无任何搜索空间定制）
        self.arch_fitness_fn = arch_fitness_fn
        self.api = api
        self.dataset = dataset
        
        # 锦标赛选择参数
        self.tournament_size = tournament_size
        self.history_max_len = history_max_len
        self.history_x = history_x if history_x is not None else [x.clone()]
        self.history_fitness = history_fitness if history_fitness is not None else [fitness.clone()]
        
        # 单步模式标志
        self.is_single_step = is_single_step
        
        # 打印控制
        self.verbose = verbose
        self.print_freq = print_freq
        self.call_count = 0
        self.best_teacher_fitness = -float('inf')
        
        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss() if teacher_model is not None else None
        self.kl_loss = nn.KLDivLoss(reduction='batchmean') if teacher_model is not None else None
        self.mse_loss = nn.MSELoss()
        
        # 贝叶斯估计器
        self.estimator = BayesianEstimator(self.x, self.fitness, self.alpha, density=density, h=h)
        
        # ====================== 一致性稳定性核心参数（通用、无人工先验） ======================
        self.stability_perturb_num = stability_perturb_num
        self.stability_perturb_scale = stability_perturb_scale
        # ========================================================================================
        
        # 初始化打印
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[Generator Info] Using ConsistencyGenerator (Zero-Prior General Version)")
            print(f"  - Tournament size: {self.tournament_size}")
            print(f"  - History max length: {self.history_max_len}")
            print(f"  - Lambda consistency: {self.lambda_consistency}")
            print(f"  - Consistency type: {self.consistency_type}")
            print(f"  - Single-step mode: {self.is_single_step}")
            print(f"{'='*60}\n")

    def _compute_stability_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        【核心通用函数】计算样本的一致性稳定性得分
        逻辑：对样本加多次微小扰动，计算预测fitness的变异系数，变异系数越小，稳定性越高
        完全无人工先验，不依赖任何搜索空间、算子信息，通用所有NAS场景
        """
        batch_size = x.shape[0]
        stability_scores = torch.ones(batch_size, device=x.device)
        
        # 只有传入了fitness计算函数，才做稳定性计算，否则返回全1
        if self.arch_fitness_fn is None:
            return stability_scores
        
        for i in range(batch_size):
            x_single = x[i:i+1]
            # 存储多次扰动后的fitness
            perturbed_fitness_list = []
            
            for _ in range(self.stability_perturb_num):
                # 加微小高斯扰动
                x_pert = x_single + torch.randn_like(x_single) * self.stability_perturb_scale
                x_pert = normalize(x_pert)
                # 计算扰动后的fitness
                _, pert_fitness, _ = self.arch_fitness_fn(
                    operation_matrix=x_pert,
                    api=self.api,
                    dataset=self.dataset
                )
                perturbed_fitness_list.append(pert_fitness.item())
            
            # 计算fitness的变异系数（标准差/均值），越小越稳定
            pert_fitness_tensor = torch.tensor(perturbed_fitness_list, device=x.device)
            fitness_mean = pert_fitness_tensor.mean()
            fitness_std = pert_fitness_tensor.std()
            
            # 避免除零，均值为负的样本直接给0分
            if fitness_mean <= 0:
                stability_scores[i] = 0.0
            else:
                cv = fitness_std / (fitness_mean + 1e-9)
                # 稳定性得分 = 1 / (1 + cv)，映射到0-1之间，越稳定得分越高
                stability_scores[i] = 1.0 / (1.0 + cv)
        
        return stability_scores

    def tournament_select(self):
        """
        【核心修正】锦标赛选择：基于「fitness × 稳定性得分」综合排序
        自动过滤预测器虚假高分样本，无任何人工先验，全搜索空间通用
        """
        # 合并历史数据
        all_history_x = torch.cat(self.history_x, dim=0)
        all_history_fitness = torch.cat(self.history_fitness, dim=0)
        
        num_total = all_history_x.shape[0]
        if num_total == 0:
            raise ValueError("History population is empty")
        
        # 计算所有历史样本的稳定性得分
        stability_scores = self._compute_stability_score(all_history_x)
        # 综合得分 = 归一化fitness × 稳定性得分
        normalized_fitness = (all_history_fitness - all_history_fitness.min()) / (all_history_fitness.max() - all_history_fitness.min() + 1e-9)
        comprehensive_score = normalized_fitness * stability_scores
        
        # 锦标赛抽样
        if num_total <= self.tournament_size:
            selected_indices = torch.arange(num_total, device=all_history_x.device)
        else:
            selected_indices = torch.randint(0, num_total, (self.tournament_size,), device=all_history_x.device)
        
        # 从抽样样本中，选综合得分最高的精英教师
        selected_score = comprehensive_score[selected_indices]
        best_idx_in_selected = torch.argmax(selected_score)
        best_idx = selected_indices[best_idx_in_selected]
        
        # 打印信息
        if self.verbose and self.call_count % self.print_freq == 0:
            current_best_fit = all_history_fitness[best_idx].item()
            current_best_stability = stability_scores[best_idx].item()
            current_best_score = comprehensive_score[best_idx].item()
            print(f"\n  [Stability-Aware Tournament] Step {self.call_count}")
            print(f"    - History pool size: {num_total}")
            print(f"    - Best teacher fitness: {current_best_fit:.4f}")
            print(f"    - Best teacher stability score: {current_best_stability:.4f}")
            print(f"    - Best teacher comprehensive score: {current_best_score:.4f}")
            if current_best_fit > self.best_teacher_fitness:
                self.best_teacher_fitness = current_best_fit
                print(f"    [NEW BEST] Global elite teacher updated!")
        
        return all_history_x[best_idx:best_idx+1], all_history_fitness[best_idx:best_idx+1]

    def _compute_consistency_loss(self, x0_est, teacher_x, teacher_fitness):
        """
        纯一致性损失，无任何人工约束、无算子惩罚
        仅引导生成器向稳定的精英架构学习，完全不干预搜索方向
        """
        batch_size = x0_est.shape[0]
        
        # 基础一致性损失
        perturbed_x0 = x0_est + torch.randn_like(x0_est) * self.perturb_scale
        perturbed_x0 = normalize(perturbed_x0)
        teacher_x_expanded = teacher_x.expand_as(x0_est)
        
        if self.consistency_type == 'mse':
            loss_orig = self.mse_loss(x0_est, teacher_x_expanded)
            loss_pert = self.mse_loss(perturbed_x0, teacher_x_expanded)
            total_loss = loss_orig + loss_pert
        elif self.consistency_type == 'kl':
            # 自动适配任意基因形状，通用所有搜索空间
            x_dim = x0_est.shape[-1]
            orig_probs = F.log_softmax(x0_est.view(batch_size, -1, x_dim), dim=-1).view(batch_size, -1)
            pert_probs = F.log_softmax(perturbed_x0.view(batch_size, -1, x_dim), dim=-1).view(batch_size, -1)
            teacher_probs = F.softmax(teacher_x_expanded.view(batch_size, -1, x_dim), dim=-1).view(batch_size, -1)
            loss_orig = self.kl_loss(orig_probs, teacher_probs)
            loss_pert = self.kl_loss(pert_probs, teacher_probs)
            total_loss = loss_orig + loss_pert
        else:
            raise ValueError(f"Unsupported consistency type: {self.consistency_type}")
        
        if self.verbose and self.call_count % self.print_freq == 0:
            print(f"    - Consistency loss: {total_loss.item():.6f}")
        
        return total_loss

    def _ddim_step(self, x_t, x0_pred, noise=1.0):
        """兼容单步/多步模式的DDIM采样，无修改"""
        if self.is_single_step:
            return normalize(x0_pred)
        else:
            return ddim_step(xt=x_t, x0=x0_pred, alphas=(self.alpha, self.alpha_past), noise=noise)

    def generate(self, x, noise, elite_rate, return_x0=False):
        self.call_count += 1
        
        # 1. 贝叶斯估计原始点
        x0_est = self.estimator(x)
        
        # 2. 热身机制：第0步不引导，只收集历史数据（适配少步数场景）
        apply_guidance = (self.call_count > 1) and (self.lambda_consistency > 0) and (not self.is_single_step)
        
        if apply_guidance:
            # 3. 锦标赛选择稳定的精英教师
            teacher_x, teacher_fitness = self.tournament_select()
            
            # 4. 一致性损失优化，修正x0_est
            x0_opt = x0_est.detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([x0_opt], lr=1e-3)
            
            # 少量迭代优化，平衡效果和效率
            opt_steps = 3
            for _ in range(opt_steps):
                optimizer.zero_grad()
                loss = self.lambda_consistency * self._compute_consistency_loss(x0_opt, teacher_x, teacher_fitness)
                loss.backward()
                optimizer.step()
            
            # 更新x0_est为优化后的值
            x0_est = x0_opt.detach()
            x0_est = normalize(x0_est)
        
        else:
            if self.verbose and self.call_count == 1:
                print(f"    [Warmup] Step 1: Collecting architecture history for stability check...")

        # 5. DDIM采样
        x_next = self._ddim_step(x_t=x, x0_pred=x0_est, noise=noise)
        x_next = normalize(x_next)
        
        if return_x0:
            return x_next, x0_est
        else:
            return x_next

    def __call__(self, noise=1.0, return_x0=False):
        return self.generate(noise=noise, return_x0=return_x0)
'''

'''
#精英锦标赛选择
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.corrector import normalize

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
        
        # 基础基因消毒
        batch_size = x_next.shape[0]
        x_reshaped = x_next.view(batch_size, 8, 7)
        x_reshaped[:, :, 0] = -100.0
        x_reshaped[:, :, 1] = -100.0
        x_next = x_reshaped.view(batch_size, -1)
        
        x_next = normalize(x_next)
        if return_x0:
            return x_next, x0_est
        else:
            return x_next

    def __call__(self, noise=1.0, return_x0=False):
        return self.generate(noise=noise, return_x0=return_x0)

class ConsistencyGenerator:
    """
    专为少步数 (num_step <= 10) 优化的 Consistency Generator
    - 精英锦标赛选择
    - 无梯度 EMA 特征融合
    - 自动热身跳过第0步
    """
    def __init__(self, x, fitness, alpha, density='uniform', h=0.1, elite_strategy=False,
                 teacher_model=None, lambda_kl=0.5, lambda_ce=1.0, lambda_consistency=1.0,
                 consistency_type='mse', perturb_scale=0.1, arch_fitness_fn=None, api=None, dataset=None,
                 tournament_size=5, history_max_len=100, history_x=None, history_fitness=None,
                 verbose=True, print_freq=10, is_single_step=False):
        
        # 基础参数
        self.x = x
        self.fitness = fitness
        self.elite_strategy = elite_strategy
        self.alpha, self.alpha_past = alpha
        self.density = density
        self.h = h
        
        # Consistency 参数 (虽然不用反向传播，但保留接口用于控制 EMA 强度)
        self.teacher_model = teacher_model.eval() if teacher_model is not None else None
        self.lambda_consistency = lambda_consistency
        self.consistency_type = consistency_type
        self.perturb_scale = perturb_scale
        
        # NAS 适配
        self.arch_fitness_fn = arch_fitness_fn
        self.api = api
        self.dataset = dataset
        
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
            print(f"[Generator Info] Using ConsistencyGenerator (Elite EMA Version)")
            print(f"  - Selection: Pure Elite (Best in History)")
            print(f"  - Guidance: EMA Feature Mixing (lambda={self.lambda_consistency})")
            print(f"  - Single-step mode: {self.is_single_step}")
            print(f"{'='*60}\n")

    def tournament_select(self):
        """
        精英选择：直接返回历史适应度最高的样本
        """
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
        # 策略：call_count > 1 (即跳过第0步) 且 权重 > 0
        apply_guidance = (self.call_count > 1) and (self.lambda_consistency > 0) and (not self.is_single_step)
        
        if apply_guidance:
            teacher_x, _ = self.tournament_select()
            
            # ========================== 核心逻辑：无梯度 EMA 融合 ==========================
            # 计算动量系数：将 lambda_consistency (通常0-1) 映射到一个合理的混合比例
            # lambda=1.0 -> momentum=0.5 (对半混合)
            # lambda=0.5 -> momentum=0.25
            momentum = self.lambda_consistency * 0.5 
            
            # 扩展教师维度以匹配 batch size
            teacher_x_expanded = teacher_x.expand_as(x0_est)
            
            # 执行特征空间的移动平均
            # x0_est_new = x0_est * (1 - m) + teacher * m
            x0_est = x0_est * (1.0 - momentum) + teacher_x_expanded * momentum
            
            # 重新归一化，保证数值稳定
            x0_est = normalize(x0_est)
            
            if self.verbose and self.call_count % self.print_freq == 0:
                print(f"    [EMA Mix] Applied momentum: {momentum:.3f}")
            # ===============================================================================
        else:
            if self.verbose and self.call_count == 1:
                print(f"    [Warmup] Step 1/2: Skipping guidance to collect history.")

        # 3. DDIM 采样
        x_next = ddim_step(xt=x, x0=x0_est, alphas=(self.alpha, self.alpha_past), noise=noise)
        
        # 4. 基因消毒 (最终保障)
        batch_size = x_next.shape[0]
        x_reshaped = x_next.view(batch_size, 8, 7)
        
        # 屏蔽 input/output
        x_reshaped[:, :, 0] = -100.0
        x_reshaped[:, :, 1] = -100.0
        
        # (可选) 稍微提升卷积权重，防止全是 skip_connect
        # 如果你发现 FLOPs 还是太低，取消下面这两行的注释
        # x_reshaped[:, :, 4] += 1.0 # nor_conv_1x1
        # x_reshaped[:, :, 5] += 1.0 # nor_conv_3x3
        
        x_next = x_reshaped.view(batch_size, -1)
        x_next = normalize(x_next)
        
        if return_x0:
            return x_next, x0_est
        else:
            return x_next

    def __call__(self, noise=1.0, return_x0=False):
        return self.generate(noise=noise, return_x0=return_x0)

'''
'''
#已经可以跑通的版本，并上传到了本地的文件夹
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.corrector import normalize

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

class ConsistencyGenerator:
    """Consistency Generator with Tournament Selection for NAS (supports num_step=1)"""
    def __init__(self, x, fitness, alpha, density='kde', h=0.1, elite_strategy=False,
                 teacher_model=None, lambda_kl=0.5, lambda_ce=1.0, lambda_consistency=1.0,
                 consistency_type='mse', perturb_scale=0.1, arch_fitness_fn=None, api=None, dataset=None,
                 tournament_size=5, history_max_len=100, history_x=None, history_fitness=None,
                 verbose=True, print_freq=10, is_single_step=False):
        # 兼容原有BayesianGenerator参数
        self.x = x
        self.fitness = fitness
        self.elite_strategy = elite_strategy
        self.alpha, self.alpha_past = alpha
        self.density = density
        self.h = h
        
        # 一致性生成器特有参数
        self.teacher_model = teacher_model.eval() if teacher_model is not None else None
        self.lambda_kl = lambda_kl
        self.lambda_ce = lambda_ce
        self.lambda_consistency = lambda_consistency
        self.consistency_type = consistency_type
        self.perturb_scale = perturb_scale
        
        # NAS适配参数
        self.arch_fitness_fn = arch_fitness_fn
        self.api = api
        self.dataset = dataset
        
        # 锦标赛选择参数
        self.tournament_size = tournament_size
        self.history_max_len = history_max_len
        self.history_x = history_x if history_x is not None else [x.clone()]
        self.history_fitness = history_fitness if history_fitness is not None else [fitness.clone()]
        
        # 单步模式标志
        self.is_single_step = is_single_step
        
        # 打印控制
        self.verbose = verbose
        self.print_freq = print_freq
        self.call_count = 0
        self.best_teacher_fitness = -float('inf')
        
        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss() if teacher_model is not None else None
        self.kl_loss = nn.KLDivLoss(reduction='batchmean') if teacher_model is not None else None
        self.mse_loss = nn.MSELoss()
        
        # 贝叶斯估计器
        self.estimator = BayesianEstimator(self.x, self.fitness, self.alpha, density=density, h=h)
        
        # 初始化打印
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[Generator Info] Using ConsistencyGenerator with Tournament Selection")
            print(f"  - Tournament size: {self.tournament_size}")
            print(f"  - History max length: {self.history_max_len}")
            print(f"  - Consistency type: {self.consistency_type}")
            print(f"  - Lambda consistency: {self.lambda_consistency}")
            print(f"  - Perturb scale: {self.perturb_scale}")
            print(f"  - Single-step mode: {self.is_single_step}")
            print(f"{'='*60}\n")

    def tournament_select(self):
        """锦标赛选择：从历史所有架构中随机选k个，返回适应度最高的样本"""
        # 合并历史所有架构和适应度
        all_history_x = torch.cat(self.history_x, dim=0)
        all_history_fitness = torch.cat(self.history_fitness, dim=0)
        
        num_total = all_history_x.shape[0]
        if num_total == 0:
            raise ValueError("History population is empty for tournament selection")
        
        # 随机抽取锦标赛样本
        if num_total <= self.tournament_size:
            selected_indices = torch.arange(num_total, device=all_history_x.device)
            if self.verbose and self.call_count % self.print_freq == 0:
                print(f"  [Tournament] History size ({num_total}) <= tournament size, using all samples")
        else:
            selected_indices = torch.randint(0, num_total, (self.tournament_size,), device=all_history_x.device)
        
        # 选择适应度最高的样本
        selected_x = all_history_x[selected_indices]
        selected_fitness = all_history_fitness[selected_indices]
        best_idx = torch.argmax(selected_fitness)
        
        # 打印锦标赛选择信息
        if self.verbose and self.call_count % self.print_freq == 0:
            print(f"\n  [Tournament Selection] Step {self.call_count}")
            print(f"    - Total history samples: {num_total}")
            print(f"    - Tournament candidates: {self.tournament_size}")
            print(f"    - Candidate fitness range: [{selected_fitness.min():.4f}, {selected_fitness.max():.4f}]")
            print(f"    - Selected teacher fitness: {selected_fitness[best_idx]:.4f}")
            if selected_fitness[best_idx] > self.best_teacher_fitness:
                self.best_teacher_fitness = selected_fitness[best_idx]
                print(f"    [NEW BEST] Teacher fitness improved! New best: {self.best_teacher_fitness:.4f}")
        
        return selected_x[best_idx:best_idx+1], selected_fitness[best_idx:best_idx+1]

    def _compute_consistency_loss(self, x0_est, teacher_x, teacher_fitness):
        """一致性损失：引导当前样本向锦标赛选出的教师样本靠近"""
        # 对当前样本添加扰动
        perturbed_x0 = x0_est + torch.randn_like(x0_est) * self.perturb_scale
        perturbed_x0 = normalize(perturbed_x0)
        
        # 扩展教师样本维度以匹配当前样本
        teacher_x_expanded = teacher_x.expand_as(x0_est)
        
        # 计算一致性损失
        if self.consistency_type == 'mse':
            loss_orig = self.mse_loss(x0_est, teacher_x_expanded)
            loss_pert = self.mse_loss(perturbed_x0, teacher_x_expanded)
            total_loss = loss_orig + loss_pert
        elif self.consistency_type == 'kl':
            orig_probs = F.log_softmax(x0_est, dim=1)
            pert_probs = F.log_softmax(perturbed_x0, dim=1)
            teacher_probs = F.softmax(teacher_x_expanded, dim=1)
            loss_orig = self.kl_loss(orig_probs, teacher_probs)
            loss_pert = self.kl_loss(pert_probs, teacher_probs)
            total_loss = loss_orig + loss_pert
        else:
            raise ValueError(f"Unsupported consistency type: {self.consistency_type}")
        
        # 打印损失信息
        if self.verbose and self.call_count % self.print_freq == 0:
            print(f"    - Consistency loss: {total_loss.item():.6f}")
        
        return total_loss

    def _ddim_step(self, x_t, x0_pred, noise=1.0):
        """单步DDIM采样，支持单步模式"""
        if self.is_single_step:
            # 单步模式：直接返回x0_pred作为最终结果
            return normalize(x0_pred)
        else:
            # 多步模式：正常DDIM采样
            return ddim_step(xt=x_t, x0=x0_pred, alphas=(self.alpha, self.alpha_past), noise=noise)

    def generate(self, x, noise, elite_rate, return_x0=False):
        self.call_count += 1
        
        # 1. 贝叶斯估计当前样本的原始点
        x0_est = self.estimator(x)
        
        # 2. 锦标赛选择教师样本（仅当历史足够时）
        if len(self.history_fitness) > 0 and len(self.history_fitness[0]) > 0:
            teacher_x, teacher_fitness = self.tournament_select()
        else:
            # 历史为空时，使用当前样本的均值作为教师
            teacher_x = x.mean(dim=0, keepdim=True)
            teacher_fitness = self.fitness.mean(dim=0, keepdim=True)
        
        # 3. 计算一致性损失
        total_loss = torch.tensor(0.0, device=x.device)
        if self.arch_fitness_fn is not None and not self.is_single_step:
            consistency_loss = self._compute_consistency_loss(x0_est, teacher_x, teacher_fitness)
            total_loss += self.lambda_consistency * consistency_loss
        
        # 4. DDIM单步采样（或单步模式直接返回）
        x_next = self._ddim_step(x_t=x, x0_pred=x0_est, noise=noise)
        x_next = normalize(x_next)
        
        # 打印生成结果
        if self.verbose and (self.call_count % self.print_freq == 0 or self.is_single_step):
            if self.is_single_step:
                print(f"  [Single-step] Directly returning denoised architecture parameters")
            else:
                print(f"    - Generated new samples, norm: {torch.norm(x_next, dim=1).mean():.4f}")
                print(f"  [Tournament] Working - guiding search towards high-fitness architectures\n")
        
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

class ConsistencyGenerator:
    def __init__(self, x, fitness, alpha, teacher_model,
                 density='kde', h=0.1,
                 lambda_kl=0.5, lambda_ce=1.0, lambda_consistency=1.0,
                 consistency_type='mse', perturb_scale=0.1):
        super().__init__(x, fitness, (alpha, alpha), density, h)  # 固定单步扩散参数α一致
        self.fitness = fitness
        self.alpha = alpha
        self.teacher_model = teacher_model.eval()
        self.lambda_kl = lambda_kl
        self.lambda_ce = lambda_ce
        self.lambda_consistency = lambda_consistency
        self.consistency_type = consistency_type
        self.perturb_scale = perturb_scale

        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.model = None
        self.estimator = BayesianEstimator(x, fitness, alpha, density, h)

    def generate(self, inputs, targets, noise=1.0):
        """生成新的架构参数"""
        # 1. 贝叶斯估计获取架构参数分布
        x0_est = self.estimator(self.x)  # [N, arch_params]

        # 2. 学生模型前向传播
        student_logits = self._model_forward(x0_est, inputs)

        # 3. 交叉熵损失（硬标签损失）
        ce = self.ce_loss(student_logits, targets)

        # 4. 教师模型软标签蒸馏（软标签损失）
        with torch.no_grad():
            teacher_logits = self.teacher_model(inputs)
            teacher_probs = F.softmax(teacher_logits, dim=1)
        student_probs = F.log_softmax(student_logits, dim=1)
        kl = self.kl_loss(student_probs, teacher_probs)

        # 5. 一致性损失
        consistency_loss = self._compute_consistency_loss(x0_est, inputs)

        # 7. 综合损失函数
        total_loss = (self.lambda_ce * ce +
                      self.lambda_kl * kl +
                      self.lambda_consistency * consistency_loss)
        total_loss.backward()  # 反向传播更新生成器参数

        # 8. 单步DDIM生成（一步扩散过程）
        x_next = self._ddim_step(self.x, x0_est, self.alpha, noise=noise)
        return x_next.detach()  # 返回新生成的架构参数

    def _compute_consistency_loss(self, arch_params, inputs):
        """计算一致性损失"""
        # 对架构参数添加随机扰动
        perturbed_params = arch_params + torch.randn_like(arch_params) * self.perturb_scale

        # 获取原始参数和扰动参数的模型输出
        with torch.no_grad():
            original_outputs = self._model_forward(arch_params, inputs)
            perturbed_outputs = self._model_forward(perturbed_params, inputs)

        # 计算一致性损失
        if self.consistency_type == 'mse':
            return F.mse_loss(original_outputs, perturbed_outputs)
        elif self.consistency_type == 'kl':
            original_probs = F.softmax(original_outputs, dim=1)
            perturbed_log_probs = F.log_softmax(perturbed_outputs, dim=1)
            return self.kl_loss(perturbed_log_probs, original_probs)
        else:
            raise ValueError(f"Unsupported consistency type: {self.consistency_type}")

    def _ddim_step(self, x_t, x0_pred, alpha, noise=1.0):
        """单步DDIM采样过程"""
        alpha_t, alpha_prev = alpha
        sigma_t = 0  # 确定性采样

        # 计算噪声预测
        eps_pred = (x_t - x0_pred * alpha_t.sqrt()) / ((1 - alpha_t).sqrt() + 1e-10)

        # 一步DDIM更新
        if alpha_prev > 0:
            x_prev = (alpha_prev.sqrt() * x0_pred +
                      (1 - alpha_prev - sigma_t ** 2).sqrt() * eps_pred +
                      sigma_t * torch.randn_like(x_t) * noise)
        else:
            x_prev = x0_pred  # 当alpha_prev为0时，直接返回x0_pred

        return x_prev

    def _model_forward(self, arch_params, inputs):
        """将架构参数映射到模型并前向传播"""
        # 假设arch_params是一个扁平化的张量，需要重塑为架构矩阵
        # 这里需要根据实际架构表示进行调整
        batch_size = arch_params.shape[0]
        arch_matrix = arch_params.view(batch_size, -1)  # 重塑为[batch_size, num_params]

        # 将架构参数应用到模型
        self.model.copy_arch_parameters(arch_matrix)

        # 前向传播
        return self.model(inputs)

'''