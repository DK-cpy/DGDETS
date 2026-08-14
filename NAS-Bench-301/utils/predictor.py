# ==================== predictor.py (NAS-Bench-301 修改版) ====================
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.corrector import normalize

# ---------------------------- 基础 DDIM 函数 ----------------------------
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

# ---------------------------- 贝叶斯估计器 ----------------------------
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

# ---------------------------- 基线贝叶斯生成器 ----------------------------
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

# ---------------------------- 一致性生成器（含 SATS + EMA） ----------------------------
class ConsistencyGenerator:
    """
    完全对齐 NB201 的 CDENAG 生成器：
    1. 稳定性感知锦标赛选择 (SATS)：综合 归一化适应度 × 稳定性得分 选择教师。
    2. 教师扰动 + EMA 混合。
    """
    def __init__(self, x, fitness, alpha, density='uniform', h=0.1, elite_strategy=False,
                 teacher_model=None, lambda_kl=0.5, lambda_ce=1.0, lambda_consistency=1.0,
                 consistency_type='mse', perturb_scale=0.1, arch_fitness_fn=None, api=None,
                 tournament_size=5, history_max_len=100, history_x=None, history_fitness=None,
                 verbose=True, print_freq=10, is_single_step=False,
                 stability_perturb_num=5, stability_perturb_scale=1e-3,
                 **fitness_kwargs):      # 接收额外参数（如 meta 流程所需，这里为兼容保留）
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

        # NAS 适配（NB301 使用 arch_fitness_fn 和 api）
        self.arch_fitness_fn = arch_fitness_fn
        self.api = api
        self.fitness_kwargs = fitness_kwargs   # 未使用，保留接口

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
            print(f"[Generator Info] Using ConsistencyGenerator (EMA + Perturb + SATS) for NB301")
            print(f"  - Lambda consistency: {self.lambda_consistency}")
            print(f"  - Tournament size: {self.tournament_size}")
            print(f"  - Perturb scale: {self.perturb_scale}")
            print(f"  - Stability perturbations: {self.stability_perturb_num}, scale: {self.stability_perturb_scale}")
            print(f"  - History length (current): {len(self.history_x)}")
            print(f"  - Single-step mode: {self.is_single_step}")
            print(f"{'='*60}\n")

    def _compute_stability_single(self, x_single):
        """
        对单个架构计算稳定性得分（适配 NB301 适应度函数）。
        NB301 的 arch_fitness_fn 应接受 adj_matrix 和 nb_api，返回 (accuracy, fitness) 或更多。
        """
        if self.arch_fitness_fn is None or self.stability_perturb_num <= 0:
            return 1.0

        perturbed_fitness = []
        for _ in range(self.stability_perturb_num):
            x_pert = x_single + torch.randn_like(x_single) * self.stability_perturb_scale
            x_pert = normalize(x_pert)
            # NB301 调用： arch_fitness(adj_matrix=x_pert, nb_api=self.api)
            out = self.arch_fitness_fn(adj_matrix=x_pert, nb_api=self.api)
            # out 通常是 (accuracy, fitness) 或者 (accuracy, fitness, ...) ，取 fitness 并转为标量
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
        """ SATS: 稳定性感知锦标赛选择 """
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

        # 稳定性得分
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
            teacher_x, _ = self.tournament_select()
            # 教师扰动
            if self.perturb_scale > 0:
                perturb_noise = torch.randn_like(teacher_x) * self.perturb_scale
                teacher_x = teacher_x + perturb_noise
                teacher_x = normalize(teacher_x)
            teacher_x_expanded = teacher_x.expand_as(x0_est)
            # EMA 混合
            x0_est = (1.0 - self.lambda_consistency) * x0_est + self.lambda_consistency * teacher_x_expanded
            x0_est = normalize(x0_est)
            if self.verbose and self.call_count % self.print_freq == 0:
                print(f"    [EMA Mix] lambda_consistency={self.lambda_consistency:.3f} applied")
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


def ddim_step(xt, x0, alphas: tuple, noise: float = None):
    """DDIM采样单步实现（兼容NAS-Bench-301，数值稳定性优化）
    Args:
        xt: 当前步样本 (n, d)
        x0: 估计的原始样本 (n, d)
        alphas: (alpha_t, alpha_{t-1}) 扩散系数
        noise: 噪声系数
    Returns:
        x_next: 下一步样本 (n, d)
    """
    alphat, alphatp = alphas
    sigma = ddpm_sigma(alphat, alphatp) * noise if noise is not None else 0.0
    # 数值稳定性优化：添加极小值避免除零
    eps = (xt - (alphat ** 0.5) * x0) / ((1.0 - alphat) ** 0.5 + 1e-10)
    if sigma is None:
        sigma = ddpm_sigma(alphat, alphatp)
    x_next = (
        (alphatp ** 0.5) * x0
        + ((1 - alphatp - sigma ** 2) ** 0.5) * eps
        + sigma * torch.randn_like(x0)
    )
    return x_next


def ddpm_sigma(alphat, alphatp):
    """计算DDPM默认sigma（适配NAS-Bench-301分布）"""
    if alphat <= 0:
        return 0.0
    return ((1 - alphatp) / (1 - alphat) * (1 - alphat / alphatp)) ** 0.5


class ConsistencyEstimator:
    """NAS-Bench-301专用一致性估计器
    核心改进：
    1. 移除vmap依赖，改用显式循环适配数据依赖控制流
    2. 适配NAS-Bench-301的分布特性调整一致性阈值
    3. 增强数值稳定性，避免空掩码/除零问题
    """
    def __init__(self, x: torch.tensor, fitness: torch.tensor, alpha, tau=0.05, density='uniform', h=0.1):
        self.x = x
        self.fitness = fitness
        self.alpha = alpha
        self.tau = tau  # NAS-Bench-301一致性筛选阈值
        self.density_method = density
        self.h = h
        if not density in ['uniform']:
            raise NotImplementedError(f'Density estimator {density} is not implemented.')

    def append(self, estimator):
        """合并多个估计器的样本和fitness"""
        self.x = torch.cat([self.x, estimator.x], dim=0)
        self.fitness = torch.cat([self.fitness, estimator.fitness], dim=0)
    
    def density(self, x):
        """均匀密度估计（兼容原BayesianEstimator接口）"""
        if self.density_method == 'uniform':
            return torch.ones(x.shape[0], device=x.device) / x.shape[0]
    
    @staticmethod
    def norm(x):
        """通用归一化函数（适配1D/多维特征）"""
        if x.shape[-1] == 1:
            return torch.abs(x).squeeze(-1)
        else:
            return torch.norm(x, dim=-1)

    def gaussian_prob(self, x, mu, sigma):
        """高斯概率计算（兼容原BayesianEstimator逻辑）"""
        dist = self.norm(x - mu)
        return torch.exp(-(dist ** 2) / (2 * sigma ** 2))

    def consistency_filter(self, x_t, p_x_t):
        """
        基于一致性的样本筛选核心逻辑
        适配NAS-Bench-301：融合贝叶斯估计+一致性筛选
        """
        # 扩散分布参数（兼容原贝叶斯逻辑）
        mu = self.x * (self.alpha ** 0.5)
        sigma = (1 - self.alpha) ** 0.5
        
        # 1. 贝叶斯概率计算
        p_diffusion = self.gaussian_prob(x_t, mu, sigma)
        prob = (self.fitness + 1e-9) * (p_diffusion + 1e-9) / (p_x_t + 1e-9)
        
        # 2. 一致性筛选：基于概率分布的高置信样本筛选
        prob_threshold = torch.quantile(prob, 1 - self.tau)  # 取top-tau的高概率样本
        mask = prob >= prob_threshold
        
        # 兜底：无满足条件样本时使用全部样本（避免空掩码）
        if not mask.any():
            mask = torch.ones_like(mask, dtype=torch.bool)
        
        # 3. 加权融合高一致性样本
        prob_filtered = prob[mask]
        z = torch.sum(prob_filtered)
        origin = torch.sum(prob_filtered.unsqueeze(1) * self.x[mask], dim=0) / (z + 1e-9)
        return origin

    def estimate(self, x_t):
        """批量估计x0（替换vmap为显式循环，修复数据依赖控制流问题）"""
        batch_size = x_t.shape[0]
        x0_est = torch.zeros_like(x_t)
        p_x_t = self.density(x_t)
        
        for i in range(batch_size):
            x0_est[i] = self.consistency_filter(x_t[i], p_x_t[i])
        return x0_est

    def __call__(self, x_t):
        return self.estimate(x_t)

    def __repr__(self):
        return f'<ConsistencyEstimator (NAS-Bench-301) {len(self.x)} samples, tau={self.tau}>'


class BayesianGenerator:
    """NAS-Bench-301基线贝叶斯生成器（保持原接口兼容）"""
    def __init__(self, x, fitness, alpha, density='uniform', h=0.1, elite_strategy=False, verbose=True):
        self.x = x
        self.fitness = fitness
        self.elite_strategy = elite_strategy
        self.alpha, self.alpha_past = alpha
        self.estimator = ConsistencyEstimator(
            self.x, self.fitness, self.alpha, density=density, h=h
        )
        if verbose:
            print(f"\n[Generator Info] Using BayesianGenerator (NAS-Bench-301 baseline)")
    
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
    """
    NAS-Bench-301专用一致性生成器（零干预通用版）
    核心特性：
    1. 兼容NAS-Bench-201的调用接口，直接替换即可运行
    2. 移除vmap依赖，适配NAS-Bench-301的数据依赖控制流
    3. 融合一致性筛选+贝叶斯估计，无人工先验/搜索空间限制
    4. 支持锦标赛选择、EMA特征融合（适配少步数场景）
    """
    def __init__(self, x, fitness, alpha, density='uniform', h=0.1, elite_strategy=False,
                 teacher_model=None, lambda_kl=0.5, lambda_ce=1.0, lambda_consistency=1.0,
                 consistency_type='mse', perturb_scale=0.1, arch_fitness_fn=None, api=None, dataset=None,
                 tournament_size=5, history_max_len=100, history_x=None, history_fitness=None,
                 verbose=True, print_freq=10, is_single_step=False,
                 tau=0.05,  # NAS-Bench-301一致性阈值
                 stability_perturb_num=5,    # 稳定性测试扰动次数
                 stability_perturb_scale=1e-3 # 稳定性测试扰动尺度
                 ):
        # 基础兼容参数
        self.x = x
        self.fitness = fitness
        self.elite_strategy = elite_strategy
        self.alpha, self.alpha_past = alpha
        self.density = density
        self.h = h
        self.tau = tau  # NAS-Bench-301专属一致性阈值
        
        # Consistency核心参数
        self.teacher_model = teacher_model.eval() if teacher_model is not None else None
        self.lambda_kl = lambda_kl
        self.lambda_ce = lambda_ce
        self.lambda_consistency = lambda_consistency
        self.consistency_type = consistency_type
        self.perturb_scale = perturb_scale
        
        # NAS适配参数（通用，无搜索空间限制）
        self.arch_fitness_fn = arch_fitness_fn
        self.api = api
        self.dataset = dataset
        
        # 锦标赛选择参数
        self.tournament_size = tournament_size
        self.history_max_len = history_max_len
        self.history_x = history_x if history_x is not None else [x.clone()]
        self.history_fitness = history_fitness if history_fitness is not None else [fitness.clone()]
        
        # 模式控制
        self.is_single_step = is_single_step
        self.verbose = verbose
        self.print_freq = print_freq
        self.call_count = 0
        self.best_teacher_fitness = -float('inf')
        
        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss() if teacher_model is not None else None
        self.kl_loss = nn.KLDivLoss(reduction='batchmean') if teacher_model is not None else None
        self.mse_loss = nn.MSELoss()
        
        # 一致性估计器（NAS-Bench-301专用）
        self.estimator = ConsistencyEstimator(
            self.x, self.fitness, self.alpha, tau=tau, density=density, h=h
        )
        
        # 稳定性核心参数
        self.stability_perturb_num = stability_perturb_num
        self.stability_perturb_scale = stability_perturb_scale
        
        # 初始化打印
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[Generator Info] Using ConsistencyGenerator (NAS-Bench-301)")
            print(f"  - Tournament size: {self.tournament_size}")
            print(f"  - History max length: {self.history_max_len}")
            print(f"  - Lambda consistency: {self.lambda_consistency}")
            print(f"  - Consistency type: {self.consistency_type}")
            print(f"  - Consistency tau: {self.tau}")
            print(f"  - Single-step mode: {self.is_single_step}")
            print(f"{'='*60}\n")

    def _compute_stability_score(self, x: torch.Tensor) -> torch.Tensor:
        """计算样本稳定性得分（NAS-Bench-301通用，无人工先验）"""
        batch_size = x.shape[0]
        stability_scores = torch.ones(batch_size, device=x.device)
        
        if self.arch_fitness_fn is None:
            return stability_scores
        
        for i in range(batch_size):
            x_single = x[i:i+1]
            perturbed_fitness_list = []
            
            # 多次扰动计算fitness变异系数
            for _ in range(self.stability_perturb_num):
                x_pert = x_single + torch.randn_like(x_single) * self.stability_perturb_scale
                x_pert = normalize(x_pert)
                _, pert_fitness, _ = self.arch_fitness_fn(
                    operation_matrix=x_pert,
                    api=self.api,
                    dataset=self.dataset
                )
                perturbed_fitness_list.append(pert_fitness.item())
            
            # 计算变异系数（越小越稳定）
            pert_fitness_tensor = torch.tensor(perturbed_fitness_list, device=x.device)
            fitness_mean = pert_fitness_tensor.mean()
            fitness_std = pert_fitness_tensor.std()
            
            if fitness_mean <= 0:
                stability_scores[i] = 0.0
            else:
                cv = fitness_std / (fitness_mean + 1e-9)
                stability_scores[i] = 1.0 / (1.0 + cv)
        
        return stability_scores

    def tournament_select(self):
        """锦标赛选择：基于fitness×稳定性得分综合排序（NAS-Bench-301适配）"""
        all_history_x = torch.cat(self.history_x, dim=0)
        all_history_fitness = torch.cat(self.history_fitness, dim=0)
        
        num_total = all_history_x.shape[0]
        if num_total == 0:
            raise ValueError("History population is empty for tournament selection")
        
        # 计算稳定性得分
        stability_scores = self._compute_stability_score(all_history_x)
        # 综合得分 = 归一化fitness × 稳定性得分
        normalized_fitness = (all_history_fitness - all_history_fitness.min()) / (all_history_fitness.max() - all_history_fitness.min() + 1e-9)
        comprehensive_score = normalized_fitness * stability_scores
        
        # 锦标赛抽样
        if num_total <= self.tournament_size:
            selected_indices = torch.arange(num_total, device=all_history_x.device)
        else:
            selected_indices = torch.randint(0, num_total, (self.tournament_size,), device=all_history_x.device)
        
        # 选择综合得分最高的样本
        selected_score = comprehensive_score[selected_indices]
        best_idx_in_selected = torch.argmax(selected_score)
        best_idx = selected_indices[best_idx_in_selected]
        
        # 打印日志
        if self.verbose and self.call_count % self.print_freq == 0:
            current_best_fit = all_history_fitness[best_idx].item()
            current_best_stability = stability_scores[best_idx].item()
            current_best_score = comprehensive_score[best_idx].item()
            print(f"\n  [Stability-Aware Tournament] Step {self.call_count} (NAS-Bench-301)")
            print(f"    - History pool size: {num_total}")
            print(f"    - Best teacher fitness: {current_best_fit:.4f}")
            print(f"    - Best teacher stability score: {current_best_stability:.4f}")
            print(f"    - Best teacher comprehensive score: {current_best_score:.4f}")
            if current_best_fit > self.best_teacher_fitness:
                self.best_teacher_fitness = current_best_fit
                print(f"    [NEW BEST] Global elite teacher updated!")
        
        return all_history_x[best_idx:best_idx+1], all_history_fitness[best_idx:best_idx+1]

    def _compute_consistency_loss(self, x0_est, teacher_x, teacher_fitness):
        """纯一致性损失（无人工约束，适配NAS-Bench-301）"""
        batch_size = x0_est.shape[0]
        perturbed_x0 = x0_est + torch.randn_like(x0_est) * self.perturb_scale
        perturbed_x0 = normalize(perturbed_x0)
        teacher_x_expanded = teacher_x.expand_as(x0_est)
        
        if self.consistency_type == 'mse':
            loss_orig = self.mse_loss(x0_est, teacher_x_expanded)
            loss_pert = self.mse_loss(perturbed_x0, teacher_x_expanded)
            total_loss = loss_orig + loss_pert
        elif self.consistency_type == 'kl':
            # 自动适配NAS-Bench-301的特征维度
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
            print(f"    - Consistency loss (NAS-Bench-301): {total_loss.item():.6f}")
        
        return total_loss

    def _ddim_step(self, x_t, x0_pred, noise=1.0):
        """兼容单步/多步模式的DDIM采样（NAS-Bench-301优化）"""
        if self.is_single_step:
            return normalize(x0_pred)
        else:
            return ddim_step(xt=x_t, x0=x0_pred, alphas=(self.alpha, self.alpha_past), noise=noise)

    def generate(self, x, noise, elite_rate, return_x0=False):
        self.call_count += 1
        
        # 1. 一致性贝叶斯估计x0
        x0_est = self.estimator(x)
        
        # 2. 热身机制：跳过第一步收集历史数据
        apply_guidance = (self.call_count > 1) and (self.lambda_consistency > 0) and (not self.is_single_step)
        
        if apply_guidance:
            # 3. 锦标赛选择稳定精英样本
            teacher_x, teacher_fitness = self.tournament_select()
            
            # 4. 一致性损失优化x0
            x0_opt = x0_est.detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([x0_opt], lr=1e-3)
            
            # 少量迭代平衡效果与效率
            opt_steps = 3
            for _ in range(opt_steps):
                optimizer.zero_grad()
                loss = self._compute_consistency_loss(x0_opt, teacher_x, teacher_fitness)
                loss.backward()
                optimizer.step()
            
            x0_est = x0_opt.detach()
            x0_est = normalize(x0_est)
        
        else:
            if self.verbose and self.call_count == 1:
                print(f"    [Warmup] Step 1: Collecting architecture history for stability check (NAS-Bench-301)...")

        # 5. DDIM采样生成下一步样本
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
    # sigma
    sigma = ddpm_sigma(alphat, alphatp) * noise
    # DDIM
    eps = (xt - (alphat**0.5) * x0) / (1.0 - alphat) ** 0.5
    if sigma is None:
        sigma = ddpm_sigma(alphat, alphatp)
    x_next = (
        (alphatp**0.5) * x0
        + ((1 - alphatp - sigma**2) ** 0.5) * eps
        + sigma * torch.randn_like(x0)
    )
    return x_next


def ddpm_sigma(alphat, alphatp):
    """Compute the default sigma for the DDPM algorithm."""
    return ((1 - alphatp) / (1 - alphat) * (1 - alphat / alphatp)) ** 0.5


class BayesianEstimator:
    """Bayesian Estimator of the origin points, based on current samples and fitness values."""

    def __init__(
        self, x: torch.tensor, fitness: torch.tensor, alpha, density="uniform", h=0.1
    ):
        self.x = x
        self.fitness = fitness
        self.alpha = alpha
        self.density_method = density
        self.h = h
        if not density in ["uniform"]:
            raise NotImplementedError(
                f"Density estimator {density} is not implemented."
            )

    def append(self, estimator):
        #
        self.x = torch.cat([self.x, estimator.x], dim=0)
        self.fitness = torch.cat([self.fitness, estimator.fitness], dim=0)

    def density(self, x):
        # ，
        if self.density_method == "uniform":
            return torch.ones(x.shape[0]) / x.shape[0]

    @staticmethod
    def norm(x):
        #
        if x.shape[-1] == 1:
            # for some reason, torch.norm become very slow when dim=1, so we use torch.abs instead
            return torch.abs(x).squeeze(-1)
        else:
            return torch.norm(x, dim=-1)

    def gaussian_prob(self, x, mu, sigma):
        #
        dist = self.norm(x - mu)
        return torch.exp(-(dist**2) / (2 * sigma**2))

    def _estimate(self, x_t, p_x_t):
        # diffusion proability, P = N(x_t; \sqrt{α_t}x,\sqrt{1-α_t})
        mu = self.x * (self.alpha**0.5)  #
        sigma = (1 - self.alpha) ** 0.5  #
        p_diffusion = self.gaussian_prob(x_t, mu, sigma)
        # （+1e-90）
        prob = (self.fitness + 1e-9) * (p_diffusion + 1e-9) / (p_x_t + 1e-9)
        z = torch.sum(prob)
        origin = torch.sum(prob.unsqueeze(1) * self.x, dim=0) / (z + 1e-9)
        return origin

    def estimate(self, x_t):
        p_x_t = self.density(x_t)  #
        origin = torch.vmap(self._estimate, (0, 0))(x_t, p_x_t)  #
        return origin

    def __call__(self, x_t):
        return self.estimate(x_t)

    def __repr__(self):
        return f"<BayesianEstimator {len(self.x)} samples>"


class BayesianGenerator:
    """Bayesian Generator for the DDIM algorithm."""

    def __init__(
        self, x, fitness, alpha, density="uniform", h=0.1, elite_strategy=False
    ):
        self.x = x
        self.fitness = fitness
        self.elite_strategy = elite_strategy
        self.alpha, self.alpha_past = alpha
        self.estimator = BayesianEstimator(
            self.x, self.fitness, self.alpha, density=density, h=h
        )

    def generate(self, x, noise, elite_rate, return_x0=False):
        # ，
        x0_est = self.estimator(x)
        # DDIM
        x_next = ddim_step(
            xt=x, x0=x0_est, alphas=(self.alpha, self.alpha_past), noise=noise
        )
        #
        x_next = normalize(x_next)
        if return_x0:
            return x_next, x0_est
        else:
            return x_next

    def __call__(self, noise=1.0, return_x0=False):
        return self.generate(noise=noise, return_x0=return_x0)
'''