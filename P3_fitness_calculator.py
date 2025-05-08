import numpy as np
from P3_best_dispatch_river import run_for_simulation_river
from P3_best_dispatch_road import run_for_simulation_road

def calculate_fitness(position,cfg,id):
    """
    计算给定位置的适应度值
    
    参数:
        position: 解空间中的位置向量
    
    返回:
        float: 适应度值
    """
    if cfg['problem_type'] == "river":
        results=run_for_simulation_river(n_clusters=position[1],height_threshold=position[2],if_show=cfg['if_show'],num_vehicles=position[0],split_mode=cfg['split_mode'],vehicle_capacity=cfg['vehicle_capacity'],max_iterations=cfg['max_iterations'],num_ants=cfg['num_ants'],alpha=cfg['alpha'],beta=cfg['beta'],rho=cfg['rho'],Q=cfg['Q'],time_weight=cfg['time_weight'],cost_weight=cfg['cost_weight'],v=cfg['v'],price_hour=cfg['price_hour'],digging_v=cfg['digging_v'],id=id)
    elif cfg['problem_type'] == "road":
        results=run_for_simulation_road(n_clusters=position[1],height_threshold=position[2],if_show=cfg['if_show'],num_vehicles=position[0],split_mode=cfg['split_mode'],vehicle_capacity=cfg['vehicle_capacity'],max_iterations=cfg['max_iterations'],num_ants=cfg['num_ants'],alpha=cfg['alpha'],beta=cfg['beta'],rho=cfg['rho'],Q=cfg['Q'],time_weight=cfg['time_weight'],cost_weight=cfg['cost_weight'],v=cfg['v'],price_hour=cfg['price_hour'],digging_v=cfg['digging_v'],id=id)
    
    # 计算适应度值
    costs = sum(results['cost'])  
    time  = max(results['time'])
    fitness = cfg['cost_weight']*costs + cfg['time_weight']*time

    return fitness