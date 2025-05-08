import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging
from logger_setup import setup_logger

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 

logger = setup_logger()

class Ant:
    def __init__(self, num_vehicles, num_customers, vehicle_capacity, alpha, beta, rho, Q, times, prices,demands, prices_digging, times_digging, pheromone, time_weight=0.5, cost_weight=0.5,cluster_center=[0,0]):
        self.routes = [[] for _ in range(num_vehicles)]  # 每辆车的路径
        self.visited = set()                             # 已访问客户
        self.loads = [0] * num_vehicles                  # 每辆车的当前载重

        self.times = [0] * num_vehicles                  # 每辆车的行驶时间
        self.costs = [0] * num_vehicles                  # 每辆车的行驶费用

        self.total_time = 0                              # 总时间
        self.total_cost = 0                              # 总代价
        self.total_weighted = 0                          # 加权总目标值

        self.current_vehicle = 0                         # 当前使用的车辆
        self.num_vehicles = num_vehicles
        self.num_customers = num_customers
        self.vehicle_capacity = vehicle_capacity
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

        
        self.time_weight = time_weight                   # 时间权重
        self.cost_weight = cost_weight                   # 花费权重
        
        self.times_matrix = times
        self.times_digging = times_digging
        self.prices_matrix = prices
        self.prices_digging = prices_digging
       
        self.pheromone = pheromone
        self.demands = demands

        self.cluster_center = cluster_center

    def build_route(self):
        # 所有车辆从仓库出发
        for v in range(self.num_vehicles):
            self.routes[v] = [0]
        
        self.visited = set()
        self.loads = [0] * self.num_vehicles
        self.current_vehicle = 0
      
        while len(self.visited) < self.num_customers:
            next_node = self.select_next_node()
          
            if next_node == 0:  # 需要返回仓库补充
                self.routes[self.current_vehicle].append(0)
                self.loads[self.current_vehicle] = 0
                
                # 如果还有未使用的车辆，切换到新车辆
                if self.loads.count(0) > 1 and self.current_vehicle < self.num_vehicles - 1:
                    unused_vehicles = [v for v in range(self.num_vehicles) if len(self.routes[v]) == 1]
                    if unused_vehicles:
                        self.current_vehicle = unused_vehicles[0]
                
            else:
                self.routes[self.current_vehicle].append(next_node)
                self.visited.add(next_node)
                self.loads[self.current_vehicle] += self.demands[next_node-1]
        
        # 所有车辆最后返回仓库
        for v in range(self.num_vehicles):
            if self.routes[v][-1] != 0:
                self.routes[v].append(0)
        
        self.calculate_cost_and_time()
  
    def select_next_node(self):
        current_node = self.routes[self.current_vehicle][-1]
        
        # 找出尚未访问且当前车辆能载重的客户
        candidates = [j for j in range(1, self.num_customers+1) 
                     if j not in self.visited and 
                     self.demands[j-1] + self.loads[self.current_vehicle] <= self.vehicle_capacity]
      
        if not candidates:
            # 如果有未使用的车辆且还有未访问的客户，考虑使用新车辆
            if len(self.visited) < self.num_customers and self.current_vehicle < self.num_vehicles - 1:
                unused_vehicles = [v for v in range(self.num_vehicles) if len(self.routes[v]) == 1]
                if unused_vehicles:
                    self.current_vehicle = unused_vehicles[0]
                    return self.select_next_node()
            
            return 0  # 必须返回仓库
      
        # 计算转移概率 - 同时考虑时间和花费
        probabilities = []
        total = 0
        for j in candidates:
            pheromone_val = self.pheromone[current_node][j]
            # 考虑时间和花费的综合启发式信息
            time_heuristic = 1 / (self.times_matrix[current_node][j] + self.times_digging[j] + 1e-10)
            cost_heuristic = 1 / (self.prices_matrix[current_node][j] + self.prices_digging[j] + 1e-10)
            heuristic_val = self.time_weight * time_heuristic + self.cost_weight * cost_heuristic
            
            probabilities.append((pheromone_val**self.alpha) * (heuristic_val**self.beta))
            total += probabilities[-1]
      
        if total == 0:
            return np.random.choice(candidates)
          
        probabilities = [p/total for p in probabilities]
        return np.random.choice(candidates, p=probabilities)
  
    def calculate_cost_and_time(self):
        self.total_cost = 0
        for v in range(self.num_vehicles):
            self.times[v] = 0
            self.costs[v] = 0
            route = self.routes[v]
            for i in range(len(route)-1):
                self.times[v] += self.times_matrix[route[i]][route[i+1]] 
                self.times[v] += self.times_digging[route[i+1]]
                self.costs[v] += self.prices_matrix[route[i]][route[i+1]]
                self.costs[v] += self.prices_digging[route[i+1]]
            # 成本需要累加
            self.total_cost += self.costs[v]
        
        self.total_time = max(self.times) if self.times else 0
        self.total_weighted = self.time_weight * self.total_time + self.cost_weight * self.total_cost


def create_problem_data(num_customers, customer_coords, demands, v, price_hour, digging_v, area, cluster_center=[0,0]):
    """创建问题数据
    
    参数:
        num_customers: 客户数量（挖方区域数量）
        customer_coords: 客户坐标数组，形状为(num_customers, 2)
        demands: 每个客户的需求量（挖方高度）
        v: 车辆行驶速度，单位为米/小时
        price_hour: 每小时运营成本，单位为元/小时
        digging_v: 挖掘速度，单位为立方米/小时
        area: 每个挖方区域的面积，用于计算挖掘体积
        cluster_center: 装卸点（仓库）坐标，默认为[0,0]
    
    返回:
        customer_coords: 客户坐标
        demands: 客户需求量
        times: 行驶时间矩阵，表示从点i到点j所需的时间
        prices: 行驶成本矩阵，表示从点i到点j所需的成本
        times_digging: 挖掘时间数组，表示在每个点挖掘所需的时间
        prices_digging: 挖掘成本数组，表示在每个点挖掘所需的成本
    """
    
    # 计算时间矩阵 - 从点i到点j的行驶时间
    times = np.zeros((num_customers+1, num_customers+1))
    for i in range(num_customers+1):
        for j in range(num_customers+1):
            if i == j:
                times[i,j] = 0  # 相同点之间的时间为0
            else:
                # 索引0表示装卸点（仓库），其他索引需要-1来匹配customer_coords
                x1, y1 = (cluster_center[0], cluster_center[1]) if i == 0 else customer_coords[i-1]
                x2, y2 = (cluster_center[0], cluster_center[1]) if j == 0 else customer_coords[j-1]
                # 计算欧氏距离并除以速度得到时间
                times[i,j] = np.sqrt((x1-x2)**2 + (y1-y2)**2) / v

    # 计算行驶成本矩阵 - 时间乘以每小时成本
    prices = times * price_hour
    
    # 计算挖掘时间 - 体积(高度*面积)除以挖掘速度
    times_digging = np.insert((demands*area)/digging_v, 0, 0)  # 在索引0处插入0（仓库不需要挖掘）
    times_digging = np.maximum(times_digging, 1e-6)  # 将负值或零值替换为一个很小的正数，避免计算问题
    
    # 计算挖掘成本 - 挖掘时间乘以每小时成本
    prices_digging = np.insert(times_digging * price_hour, 0, 0)  # 在索引0处插入0
    prices_digging = np.maximum(prices_digging, 1e-6)  # 确保没有负值

    return customer_coords, demands, times, prices, times_digging, prices_digging


def start_simulation(max_iterations, num_ants, num_vehicles, num_customers, vehicle_capacity, 
                    alpha, beta, rho, Q, customer_coords, demands, times, prices, times_digging, prices_digging, 
                    time_weight=0.5, cost_weight=0.5, if_show=True,cluster_center=[0,0],id=0,cluster=0):
    """运行蚁群算法模拟，同时优化时间和花费
    
    参数:
        max_iterations: 最大迭代次数
        num_ants: 蚂蚁数量
        num_vehicles: 车辆数量
        num_customers: 客户数量（挖方区域数量）
        vehicle_capacity: 车辆容量
        alpha: 信息素重要程度参数
        beta: 启发式信息重要程度参数
        rho: 信息素蒸发系数
        Q: 信息素增强系数
        customer_coords: 客户坐标数组，形状为(num_customers, 2)
        demands: 客户需求量（挖方高度）
        times: 行驶时间矩阵，表示从点i到点j所需的时间
        prices: 行驶成本矩阵，表示从点i到点j所需的成本
        times_digging: 挖掘时间数组，表示在每个点挖掘所需的时间
        prices_digging: 挖掘成本数组，表示在每个点挖掘所需的成本
        time_weight: 时间权重，默认为0.5
        cost_weight: 成本权重，默认为0.5
        if_show: 是否显示可视化结果，默认为True
    
    返回:
        无直接返回值，但会打印迭代过程中的最佳解，并可选择性地可视化结果
    """
    # 初始化信息素矩阵
    pheromone = np.ones((num_customers+1, num_customers+1))
    
    # 蚁群算法主循环
    best_time = float('inf')
    best_cost = float('inf')
    best_weighted = float('inf')
    best_routes = None
    history_time = []
    history_cost = []
    history_weighted = []

    for iteration in range(max_iterations):
        ants = [Ant(num_vehicles, num_customers, vehicle_capacity, alpha, beta, rho, Q, 
                   times, prices, demands, prices_digging, times_digging, pheromone, 
                   time_weight, cost_weight,cluster_center) for _ in range(num_ants)]
    
        # 所有蚂蚁构建路径
        for ant in ants:
            ant.build_route()
            
            # 更新基于加权目标的最优解
            if ant.total_weighted < best_weighted:
                best_weighted = ant.total_weighted
                best_time = ant.total_time
                best_cost = ant.total_cost
                best_routes = [route.copy() for route in ant.routes]
    
        # 信息素挥发
        pheromone = pheromone * (1 - rho)   
    
        # 信息素更新（精英蚂蚁策略）
        for ant in ants:
            # 根据加权目标更新信息素
            for v in range(num_vehicles):
                route = ant.routes[v]
                for i in range(len(route)-1):
                    from_node = route[i]
                    to_node = route[i+1]
                    # 使用加权目标作为信息素增强的依据
                    pheromone[from_node][to_node] += Q / ant.total_weighted
    
        history_time.append(best_time)
        history_cost.append(best_cost)
        history_weighted.append(best_weighted)
        logger.info(f"粒子{id} 迭代 {iteration+1}, 最佳时间: {best_time:.2f}, 最佳代价: {best_cost:.2f}, 加权值: {best_weighted:.2f}")

    
    if if_show:
        visualize_results(history_time, history_cost, history_weighted, best_time, best_cost, best_weighted, best_routes, customer_coords, num_vehicles,cluster_center,id,cluster)
    
    return best_routes, best_time, best_cost, best_weighted


def visualize_results(history_time, history_cost, history_weighted, best_time, best_cost, best_weighted, best_routes, customer_coords, num_vehicles,cluster_center,id,cluster):
    """可视化结果
    
    参数:
        history_time: 列表，每次迭代的最佳时间记录
        history_cost: 列表，每次迭代的最佳花费记录
        history_weighted: 列表，每次迭代的最佳加权值记录
        best_time: 浮点数，最终最佳时间
        best_cost: 浮点数，最终最佳花费
        best_routes: 列表，最佳路径方案
        customer_coords: 数组，客户坐标
        num_vehicles: 整数，车辆数量
    """
    plt.figure(figsize=(16, 8))

    plt.subplot(3, 2, 1)
    plt.plot(history_time, '#f9f871')
    plt.title('时间优化过程')
    plt.xlabel('迭代次数')
    plt.ylabel('最佳时间')

   
    plt.subplot(3, 2, 3)
    plt.plot(history_cost, '#ffc735')
    plt.title('花费优化过程')
    plt.xlabel('迭代次数')
    plt.ylabel('最佳花费')
    
   
    plt.subplot(3, 2, 5)
    plt.plot(history_weighted, '#ff9671')
    plt.title('加权目标优化过程')
    plt.xlabel('迭代次数')
    plt.ylabel('加权值')

    
    plt.subplot(1, 2, 2)
    plt.scatter([cluster_center[0]], [cluster_center[1]], c='#FF6F91', s=100, label='仓库')
    plt.scatter(customer_coords[:,0], customer_coords[:,1], c='#845ec2', label='客户')

    
    colors = cm.viridis(np.linspace(0, 1, num_vehicles))

   
    for v in range(num_vehicles):
        if not best_routes[v] or len(best_routes[v]) <= 1:
            continue
        
        current_route = []
        for node in best_routes[v]:
            current_route.append((cluster_center[0],cluster_center[1]) if node == 0 else customer_coords[node-1])
        
        x, y = zip(*current_route)
        plt.plot(x, y, '--', color=colors[v], alpha=0.7, label=f'车辆 {v+1}')

    plt.title(f'最佳路径 (时间: {best_time:.2f}, 代价: {best_cost:.2f}, 加权: {best_weighted:.2f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'最佳路径_时间{best_time:.2f}_代价{best_cost:.2f}_加权{best_weighted:.2f}_聚类{cluster}.pdf', format='pdf', bbox_inches='tight')
    # plt.show()

    
    logger.info(f"\n粒子{id} 聚类{cluster}各车辆路径详情：")
    for v in range(num_vehicles):
        if best_routes[v]:
            logger.info(f"粒子{id} 车辆 {v+1} 路径: {best_routes[v]}")

def run_for_one_region(num_customers, vehicle_capacity, num_vehicles, max_iterations, num_ants, 
                      alpha, beta, rho, Q, customer_coords, demands_height, 
                      time_weight=0.5, cost_weight=0.5, v=10, price_hour=100, digging_v=5, area=16,if_show=True, cluster_center=[0,0],id=0,cluster=0):
    """
    运行蚁群算法，对一个区域进行优化，同时考虑时间和花费
    
    参数:
    - num_customers: 客户数量（需要服务的点位数量）
    - vehicle_capacity: 车辆容量（每辆车最大装载量）
    - num_vehicles: 车辆数量（可用于配送的车辆总数）
    - max_iterations: 最大迭代次数（算法运行的最大循环次数）
    - num_ants: 蚂蚁数量（参与寻路的蚂蚁个体数）
    - alpha: 信息素重要程度参数（控制信息素对蚂蚁决策的影响）
    - beta: 启发式信息重要程度参数（控制距离等启发式因素对决策的影响）
    - rho: 信息素蒸发系数（每次迭代后信息素衰减的比例）
    - Q: 信息素增加强度系数（蚂蚁完成路径后释放信息素的量）
    - customer_coords: 客户坐标列表（每个客户点的位置坐标）
    - demands_height: 需求高度列表（每个挖方点的高度值）
    - time_weight: 时间权重 (0-1之间，优化目标中时间因素的权重)
    - cost_weight: 花费权重 (0-1之间，优化目标中成本因素的权重)
    - v: 车辆行驶速度（单位：米/小时，影响行驶时间计算）
    - price_hour: 每小时运营成本（单位：元/小时，影响总成本计算）
    - digging_v: 挖掘速度（单位：立方米/小时，影响挖掘时间计算）
    - area: 挖掘面积（单位：平方米，用于计算每个点的挖掘体积）
    - if_show: 是否显示结果（控制是否展示优化过程和结果图表）
    - cluster_center: 聚类中心坐标（当前区域的中心点坐标，默认为原点[0,0]）
    """
    # 创建问题数据
    customer_coords, demands, times, prices, times_digging, prices_digging = create_problem_data(num_customers, customer_coords, demands_height, v, price_hour, digging_v, area, cluster_center)
    
    # 运行蚁群算法
    best_routes, best_time, best_cost, best_weighted = start_simulation(
        max_iterations, num_ants, num_vehicles, num_customers, 
        vehicle_capacity, alpha, beta, rho, Q, customer_coords, 
        demands, times, prices, times_digging, prices_digging,
        time_weight, cost_weight, if_show,cluster_center,id,cluster)
    
    return best_routes, best_time, best_cost, best_weighted


