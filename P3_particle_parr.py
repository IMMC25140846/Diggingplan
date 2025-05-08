import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional
# 导入适应度计算接口
from P3_fitness_calculator import calculate_fitness
import yaml
# 导入并行处理模块
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
# 导入日志模块
#使用中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 从particle_parr.py中移除setup_logger函数，改为导入
from logger_setup import setup_logger

logger = setup_logger()

# 定义用于并行计算适应度的函数
def evaluate_fitness(particle_position, cfg,id):
    """并行计算适应度的函数"""
    return calculate_fitness(particle_position, cfg,id)

class Particle:
    """粒子类，代表搜索空间中的一个候选解"""
    
    def __init__(self, dim: int, bounds: List[Tuple[float, float]], initial_position: Optional[np.ndarray] = None, integer_dims: List[int] = None):
        """
        初始化一个粒子
        
        参数:
            dim: 问题的维度
            bounds: 每个维度的取值范围，格式为[(min_1, max_1), (min_2, max_2), ...]
            initial_position: 可选的初始位置，如果提供则使用此位置，否则随机初始化
            integer_dims: 需要取整数值的维度列表，例如[0, 1]表示第0维和第1维需要是整数
        """
        # 存储哪些维度需要是整数
        self.integer_dims = [] if integer_dims is None else integer_dims
        
        # 初始化位置，如果提供了初始位置则使用，否则随机初始化
        if initial_position is not None:
            self.position = initial_position.copy()
        else:
            self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
            
        # 将指定维度转为整数
        for dim_idx in self.integer_dims:
            self.position[dim_idx] = int(self.position[dim_idx])
            
        # 随机初始化速度（位置范围的10%作为速度范围）
        self.velocity = np.array([np.random.uniform(-0.1*(high-low), 0.1*(high-low)) for low, high in bounds])
        # 粒子的当前适应度值
        self.fitness = float('inf')  # 初始化为无穷大（假设是最小化问题）
        # 粒子历史最佳位置
        self.best_position = self.position.copy()
        # 确保best_position中的整数维度也是整数类型
        for dim_idx in self.integer_dims:
            self.best_position[dim_idx] = int(self.best_position[dim_idx])
        # 粒子历史最佳适应度
        self.best_fitness = float('inf')
        
        # 用于剪枝的属性
        self.poor_performance_count = 0  # 连续表现不佳的次数
        

class ParticleSwarmOptimization:
    """粒子群优化算法"""
    
    def __init__(self, 
                 dim: int, 
                 bounds: List[Tuple[float, float]], 
                 num_particles: int = 30,
                 max_iter: int = 100,
                 w: float = 0.7,  # 惯性权重
                 c1: float = 1.5,  # 认知系数
                 c2: float = 1.5,  # 社会系数
                 minimize: bool = True,
                 initial_values=None,
                 integer_dims=None,
                 cfg=None,
                 early_stop: bool = False,   # 是否启用早停机制
                 early_stop_iter: int = 20,  # 连续多少次迭代无改善时停止
                 early_stop_tol: float = 1e-6,  # 改善的最小阈值
                 dynamic_params: bool = False,  # 是否启用参数动态调整
                 w_strategy: str = "linear",  # 惯性权重调整策略：linear, nonlinear, chaotic
                 w_min: float = 0.4,         # 惯性权重最小值
                 w_max: float = 0.9,         # 惯性权重最大值
                 c_strategy: str = "constant",  # 学习因子调整策略：constant, adaptive
                 pruning: bool = False,      # 是否启用剪枝
                 pruning_start: int = 20,    # 从第几次迭代开始剪枝
                 pruning_threshold: float = 0.5,  # 剪枝阈值（相对于全局最佳的差距比例）
                 pruning_count: int = 10,    # 连续多少次表现不佳时剪枝
                 min_particles: int = 5,     # 保留的最小粒子数量
                 parallel: bool = True,      # 是否启用并行计算
                 n_processes: int = None):   # 并行进程数量，None表示使用CPU核心数
        """
        初始化PSO算法
        
        参数:
            dim: 搜索空间的维度
            bounds: 每个维度的取值范围
            num_particles: 粒子数量
            max_iter: 最大迭代次数
            w: 惯性权重
            c1: 认知系数（个体学习因子）
            c2: 社会系数（群体学习因子）
            minimize: 是否为最小化问题（True为最小化，False为最大化）
            initial_values: 每个维度的初始值，如果提供，第一个粒子会使用这些值初始化
            integer_dims: 需要取整数值的维度列表，例如[0, 1]表示第0维和第1维需要是整数
            early_stop: 是否启用早停机制
            early_stop_iter: 早停条件，连续多少次迭代无改善时停止
            early_stop_tol: 早停条件，改善小于此值视为无改善
            dynamic_params: 是否启用参数动态调整
            w_strategy: 惯性权重调整策略
            w_min: 惯性权重最小值（用于线性和非线性策略）
            w_max: 惯性权重最大值（用于线性和非线性策略）
            c_strategy: 学习因子调整策略
            pruning: 是否启用剪枝功能
            pruning_start: 从第几次迭代开始剪枝
            pruning_threshold: 剪枝阈值（相对于全局最佳的差距比例）
            pruning_count: 连续多少次表现不佳时剪枝
            min_particles: 保留的最小粒子数量
            parallel: 是否启用并行计算
            n_processes: 并行进程数量，None表示使用CPU核心数
        """
        # 移除了objective_func参数，将从外部接口获取适应度
        self.dim = dim
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.minimize = minimize
        self.integer_dims = [] if integer_dims is None else integer_dims
        self.cfg = cfg
        
        # 早停相关参数
        self.early_stop = early_stop
        self.early_stop_iter = early_stop_iter
        self.early_stop_tol = early_stop_tol
        
        # 参数动态调整相关参数
        self.dynamic_params = dynamic_params
        self.w_strategy = w_strategy
        self.w_min = w_min
        self.w_max = w_max
        self.c_strategy = c_strategy
        
        # 如果启用动态参数，初始化惯性权重为最大值
        if self.dynamic_params and self.w_strategy in ["linear", "nonlinear"]:
            self.w = self.w_max
            
        # 混沌映射参数（用于chaotic策略）
        self.chaotic_z = 0.7  # 初始值在(0,1)之间
        
        # 初始化粒子群
        self.particles = []
        
        # 如果提供了初始值，第一个粒子使用初始值，其余随机初始化
        if initial_values is not None:
            initial_position = np.array(initial_values)
            # 确保初始位置在边界内
            for i in range(dim):
                low, high = bounds[i]
                initial_position[i] = max(low, min(high, initial_position[i]))
                # 如果是整数维度，则取整
                if i in self.integer_dims:
                    initial_position[i] = int(initial_position[i])
            
            # 第一个粒子使用初始值
            self.particles.append(Particle(dim, bounds, initial_position, self.integer_dims))
            
            # 其余粒子随机初始化
            for _ in range(1, num_particles):
                self.particles.append(Particle(dim, bounds, None, self.integer_dims))
        else:
            # 所有粒子随机初始化
            self.particles = [Particle(dim, bounds, None, self.integer_dims) for _ in range(num_particles)]
        
        # 初始化全局最佳
        self.global_best_position = None
        self.global_best_fitness = float('inf') if minimize else float('-inf')
        
        # 记录每次迭代的最佳适应度，用于后续分析
        self.fitness_history = []
        
        # 剪枝相关参数
        self.pruning = pruning
        self.pruning_start = pruning_start
        self.pruning_threshold = pruning_threshold
        self.pruning_count = pruning_count
        self.min_particles = min_particles
        
        # 并行计算相关参数
        self.parallel = parallel
        self.n_processes = n_processes if n_processes is not None else multiprocessing.cpu_count()
        
    def update_params(self, iter_num):
        """
        根据当前迭代次数更新算法参数
        
        参数:
            iter_num: 当前迭代次数
        """
        if not self.dynamic_params:
            return
            
        # 更新惯性权重
        if self.w_strategy == "linear":
            # 线性递减惯性权重
            self.w = self.w_max - (self.w_max - self.w_min) * iter_num / self.max_iter
        elif self.w_strategy == "nonlinear":
            # 非线性递减惯性权重（二次递减）
            self.w = self.w_min + (self.w_max - self.w_min) * ((self.max_iter - iter_num) / self.max_iter) ** 2
        elif self.w_strategy == "chaotic":
            # 使用Logistic映射生成混沌序列
            self.chaotic_z = 4 * self.chaotic_z * (1 - self.chaotic_z)
            self.w = self.w_min + (self.w_max - self.w_min) * self.chaotic_z
        
        # 更新学习因子
        if self.c_strategy == "adaptive":
            # 自适应调整学习因子
            progress_ratio = iter_num / self.max_iter
            # 随着迭代进行，认知系数减小，社会系数增大
            self.c1 = 2.5 - 2 * progress_ratio
            self.c2 = 0.5 + 2 * progress_ratio
    
    def prune_particles(self, iter_num):
        """
        剪枝表现不佳的粒子
        
        参数:
            iter_num: 当前迭代次数
        """
        if not self.pruning or iter_num < self.pruning_start or len(self.particles) <= self.min_particles:
            return
        
        # 计算适应度差距阈值
        if self.minimize:
            # 最小化问题
            base_fitness = self.global_best_fitness
            threshold = base_fitness * (1 + self.pruning_threshold)
            
            # 检查并更新每个粒子的表现计数
            for particle in self.particles:
                if particle.fitness > threshold:
                    particle.poor_performance_count += 1
                else:
                    particle.poor_performance_count = 0
        else:
            # 最大化问题
            base_fitness = self.global_best_fitness
            threshold = base_fitness * (1 - self.pruning_threshold)
            
            # 检查并更新每个粒子的表现计数
            for particle in self.particles:
                if particle.fitness < threshold:
                    particle.poor_performance_count += 1
                else:
                    particle.poor_performance_count = 0
        
        # 移除表现不佳的粒子
        if len(self.particles) > self.min_particles:
            # 筛选出需要保留的粒子
            good_particles = [p for p in self.particles if p.poor_performance_count < self.pruning_count]
            
            # 确保至少保留min_particles个粒子
            if len(good_particles) >= self.min_particles:
                pruned_count = len(self.particles) - len(good_particles)
                self.particles = good_particles
                if pruned_count > 0:
                    return pruned_count  # 返回剪枝的粒子数量
        
        return 0
    
    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        执行粒子群优化
        
        参数:
            verbose: 是否打印迭代过程信息
            
        返回:
            Tuple[np.ndarray, float]: (最优解位置, 最优适应度值)
        """
        # 初始化：评估所有粒子的初始适应度
        if self.parallel:
            # 并行评估初始适应度
            with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                # 提交所有粒子的适应度计算任务
                futures = [executor.submit(evaluate_fitness, particle.position, self.cfg,id) 
                          for id,particle in enumerate(self.particles)]
                
                # 收集结果
                for i, future in enumerate(as_completed(futures)):
                    self.particles[i].fitness = future.result()
                    
                    # 更新粒子的个体最佳
                    self.particles[i].best_fitness = self.particles[i].fitness
                    self.particles[i].best_position = self.particles[i].position.copy()
                    # 确保best_position中的整数维度是整数类型
                    for dim_idx in self.integer_dims:
                        self.particles[i].best_position[dim_idx] = int(self.particles[i].best_position[dim_idx])
                    
                    # 更新全局最佳
                    if (self.minimize and self.particles[i].fitness < self.global_best_fitness) or \
                       (not self.minimize and self.particles[i].fitness > self.global_best_fitness):
                        self.global_best_fitness = self.particles[i].fitness
                        self.global_best_position = self.particles[i].position.copy()
                        # 确保global_best_position中的整数维度是整数类型
                        for dim_idx in self.integer_dims:
                            self.global_best_position[dim_idx] = int(self.global_best_position[dim_idx])
        else:
            # 串行评估初始适应度（保持原有代码）
            for particle in self.particles:
                # 使用外部接口计算适应度
                particle.fitness = calculate_fitness(particle.position, self.cfg)
                
                # 更新粒子的个体最佳
                particle.best_fitness = particle.fitness
                particle.best_position = particle.position.copy()
                # 确保best_position中的整数维度是整数类型
                for dim_idx in self.integer_dims:
                    particle.best_position[dim_idx] = int(particle.best_position[dim_idx])
                
                # 更新全局最佳
                if (self.minimize and particle.fitness < self.global_best_fitness) or \
                   (not self.minimize and particle.fitness > self.global_best_fitness):
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()
                    # 确保global_best_position中的整数维度是整数类型
                    for dim_idx in self.integer_dims:
                        self.global_best_position[dim_idx] = int(self.global_best_position[dim_idx])
        
        # 早停相关变量
        no_improve_count = 0
        last_best_fitness = self.global_best_fitness
        
        # 主循环
        for iter_num in range(self.max_iter):
            # 动态更新算法参数
            self.update_params(iter_num)
            
            # 更新粒子的速度和位置
            for particle in self.particles:
                # 速度更新公式
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                
                cognitive_velocity = self.c1 * r1 * (particle.best_position - particle.position)
                social_velocity = self.c2 * r2 * (self.global_best_position - particle.position)
                
                particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity
                
                # 位置更新
                particle.position = particle.position + particle.velocity
                
                # 整数维度取整
                for dim_idx in self.integer_dims:
                    particle.position[dim_idx] = int(particle.position[dim_idx])
                
                # 边界处理
                for i in range(self.dim):
                    low, high = self.bounds[i]
                    if particle.position[i] < low:
                        particle.position[i] = low
                        particle.velocity[i] *= -0.5  # 反弹
                    elif particle.position[i] > high:
                        particle.position[i] = high
                        particle.velocity[i] *= -0.5  # 反弹
                    
                    # 确保整数维度在边界处理后仍然是整数
                    if i in self.integer_dims:
                        particle.position[i] = int(particle.position[i])
            
            # 并行评估所有粒子的适应度
            if self.parallel:
                positions = [particle.position for particle in self.particles]
                
                with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                    # 提交所有粒子的适应度计算任务
                    futures = [executor.submit(evaluate_fitness, pos, self.cfg,id) for id,pos in enumerate(positions)]
                    
                    # 收集结果并更新粒子适应度
                    for i, future in enumerate(as_completed(futures)):
                        fitness = future.result()
                        self.particles[i].fitness = fitness
                        
                        # 更新粒子的个体最佳
                        if (self.minimize and fitness < self.particles[i].best_fitness) or \
                           (not self.minimize and fitness > self.particles[i].best_fitness):
                            self.particles[i].best_fitness = fitness
                            self.particles[i].best_position = self.particles[i].position.copy()
                            # 确保best_position中的整数维度是整数类型
                            for dim_idx in self.integer_dims:
                                self.particles[i].best_position[dim_idx] = int(self.particles[i].best_position[dim_idx])
                        
                        # 更新全局最佳
                        if (self.minimize and fitness < self.global_best_fitness) or \
                           (not self.minimize and fitness > self.global_best_fitness):
                            self.global_best_fitness = fitness
                            self.global_best_position = self.particles[i].position.copy()
                            # 确保global_best_position中的整数维度是整数类型
                            for dim_idx in self.integer_dims:
                                self.global_best_position[dim_idx] = int(self.global_best_position[dim_idx])
            else:
                # 串行评估适应度（保持原有代码）
                for particle in self.particles:
                    # 评估新位置，使用外部接口计算适应度
                    particle.fitness = calculate_fitness(particle.position, self.cfg)
                    
                    # 更新粒子的个体最佳
                    if (self.minimize and particle.fitness < particle.best_fitness) or \
                       (not self.minimize and particle.fitness > particle.best_fitness):
                        particle.best_fitness = particle.fitness
                        particle.best_position = particle.position.copy()
                        # 确保best_position中的整数维度是整数类型
                        for dim_idx in self.integer_dims:
                            particle.best_position[dim_idx] = int(particle.best_position[dim_idx])
                    
                    # 更新全局最佳
                    if (self.minimize and particle.fitness < self.global_best_fitness) or \
                       (not self.minimize and particle.fitness > self.global_best_fitness):
                        self.global_best_fitness = particle.fitness
                        self.global_best_position = particle.position.copy()
                        # 确保global_best_position中的整数维度是整数类型
                        for dim_idx in self.integer_dims:
                            self.global_best_position[dim_idx] = int(self.global_best_position[dim_idx])
            
            # 记录本次迭代的最佳适应度
            self.fitness_history.append(self.global_best_fitness)
            
            # 剪枝表现不佳的粒子
            pruned_count = self.prune_particles(iter_num)
            
            # 检查是否需要早停
            if self.early_stop:
                # 计算改善程度
                if self.minimize:
                    improvement = last_best_fitness - self.global_best_fitness
                else:
                    improvement = self.global_best_fitness - last_best_fitness
                
                # 判断是否有显著改善
                if improvement > self.early_stop_tol:
                    no_improve_count = 0  # 重置计数器
                    last_best_fitness = self.global_best_fitness
                else:
                    no_improve_count += 1
                
                # 达到早停条件
                if no_improve_count >= self.early_stop_iter:
                    if verbose:
                        logger.info(f"早停触发！连续 {self.early_stop_iter} 次迭代无显著改善，在第 {iter_num + 1} 次迭代停止")
                    break
            
            # 打印进度信息
            if verbose and (iter_num + 1) % 1 == 0:
                if self.dynamic_params:
                    logger.info(f"迭代 {iter_num + 1}/{self.max_iter}, 当前最佳适应度: {self.global_best_fitness}, 最佳位置: {self.global_best_position}, w={self.w:.4f}, c1={self.c1:.4f}, c2={self.c2:.4f}, 剩余粒子: {len(self.particles)}")
                else:
                    logger.info(f"迭代 {iter_num + 1}/{self.max_iter}, 当前最佳适应度: {self.global_best_fitness}, 最佳位置: {self.global_best_position}, 剩余粒子: {len(self.particles)}")
        
        if verbose:
            logger.info(f"优化完成！最佳位置: {self.global_best_position}, 最佳适应度: {self.global_best_fitness}")
        
        return self.global_best_position, self.global_best_fitness
    
    def plot_convergence(self) -> None:
        """绘制收敛曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.fitness_history) + 1), self.fitness_history, marker='o', linestyle='-', markersize=2)
        plt.title('PSO 算法收敛曲线')
        plt.xlabel('迭代次数')
        plt.ylabel('全局最佳适应度')
        plt.grid(True)
        plt.show()

def load_config(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)
    return cfg

# 示例使用
def start_simulation():
    # 设置日志
    
    
    cfg = load_config('config.yaml')
    logger.info("加载配置文件成功")
    
    # 定义问题参数
    dim = 3  
    bounds = [(cfg['vehicles_lower_bounds'], cfg['vehicles_upper_bounds']),
              (cfg['n_clusters_lower_bounds'], cfg['n_clusters_upper_bounds']),
              (cfg['height_threshold_lower_bounds'], cfg['height_threshold_upper_bounds'])]
    
    logger.info(f"问题维度: {dim}, 搜索范围: {bounds}")
    
    initial_values = [cfg['num_vehicles'], cfg['n_clusters'], cfg['height_threshold']]
    logger.info(f"初始值: {initial_values}")
    
    # 指定第0维和第1维（车辆数量和聚类数）为整数维度
    integer_dims = [0, 1]
    
    # 从配置文件获取早停参数，如果不存在则使用默认值
    early_stop = cfg.get('early_stop', False)
    early_stop_iter = cfg.get('early_stop_iter', 20)
    early_stop_tol = cfg.get('early_stop_tol', 1e-6)
    
    # 从配置文件获取动态参数设置，如果不存在则使用默认值
    dynamic_params = cfg.get('dynamic_params', False)
    w_strategy = cfg.get('w_strategy', 'linear')
    w_min = cfg.get('w_min', 0.4)
    w_max = cfg.get('w_max', 0.9)
    c_strategy = cfg.get('c_strategy', 'constant')
    
    # 从配置文件获取剪枝参数
    pruning = cfg.get('pruning', False)
    pruning_start = cfg.get('pruning_start', 20)
    pruning_threshold = cfg.get('pruning_threshold', 0.5)
    pruning_count = cfg.get('pruning_count', 10)
    min_particles = cfg.get('min_particles', 5)
    
    # 从配置文件获取并行处理参数
    parallel = cfg.get('parallel', True)
    n_processes = cfg.get('n_processes', None)
    
    logger.info("PSO 参数配置:")
    logger.info(f"粒子数: {cfg['num_particles']}, 最大迭代次数: {cfg['max_iter']}")
    logger.info(f"是否最小化问题: {cfg['minimize']}, w: {cfg['w']}, c1: {cfg['c1']}, c2: {cfg['c2']}")
    logger.info(f"早停: {early_stop}, 早停迭代: {early_stop_iter}, 早停阈值: {early_stop_tol}")
    logger.info(f"动态参数: {dynamic_params}, w策略: {w_strategy}, w范围: [{w_min}, {w_max}], c策略: {c_strategy}")
    logger.info(f"剪枝: {pruning}, 开始剪枝迭代: {pruning_start}, 剪枝阈值: {pruning_threshold}")
    logger.info(f"最小粒子数: {min_particles}, 并行计算: {parallel}, 进程数: {n_processes}")
    
    pso = ParticleSwarmOptimization(
        dim=dim,
        bounds=bounds,
        num_particles=cfg['num_particles'],
        max_iter=cfg['max_iter'],
        minimize=cfg['minimize'],
        w=cfg['w'],
        c1=cfg['c1'],
        c2=cfg['c2'],
        initial_values=initial_values,
        integer_dims=integer_dims,  # 添加整数维度参数
        cfg=cfg,
        early_stop=early_stop,
        early_stop_iter=early_stop_iter,
        early_stop_tol=early_stop_tol,
        dynamic_params=dynamic_params,
        w_strategy=w_strategy,
        w_min=w_min,
        w_max=w_max,
        c_strategy=c_strategy,
        pruning=pruning,
        pruning_start=pruning_start,
        pruning_threshold=pruning_threshold,
        pruning_count=pruning_count,
        min_particles=min_particles,
        parallel=parallel,
        n_processes=n_processes
    )
    
    logger.info("开始PSO优化...")
    
    # 运行优化
    best_position, best_fitness = pso.optimize(verbose=True)
    
    logger.info(f"优化结束. 最佳位置: {best_position}, 最佳适应度: {best_fitness}")
    
    # 绘制收敛曲线
    pso.plot_convergence()
    logger.info("收敛曲线已生成")
    
    return best_position, best_fitness


if __name__ == "__main__":
    try:
        best_position, best_fitness = start_simulation()
        logger.info(f"程序成功完成. 最终结果: 位置={best_position}, 适应度={best_fitness}")
    except Exception as e:
        logger.exception(f"程序执行出错: {str(e)}")
