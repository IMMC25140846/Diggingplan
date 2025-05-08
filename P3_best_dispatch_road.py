from P2_road_earth import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import yaml
from P3_ant import run_for_one_region
import logging
from logger_setup import setup_logger

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

logger = setup_logger()

X_OFFSET=0
Y_OFFSET=0

def cluster_earthwork(earth_result, n_clusters=10, height_threshold=0.5):
    """
    对挖方区域数据进行聚类分析，确定最佳装卸点位置
    
    参数:
        earth_result: 三列数组，包含x坐标、y坐标和挖高
        n_clusters: 聚类数量，即希望得到的装卸点数量
        height_threshold: 高度阈值权重
    
    返回:
        cluster_centers: 聚类中心点坐标，可作为装卸点位置
        labels: 每个挖方区域所属的聚类
        earth_points: 挖方区域的中心点坐标和高度数据
    """
    
    centers_x = earth_result[:, 0] - X_OFFSET
    centers_y = earth_result[:, 1] - Y_OFFSET
    heights = earth_result[:, 2]
    
    X = np.column_stack((centers_x, centers_y, heights))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
   
    X_scaled[:, 2] = X_scaled[:, 2] * height_threshold
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    cluster_centers_scaled = kmeans.cluster_centers_
    cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
    
    earth_points = pd.DataFrame({
        'x': centers_x,
        'y': centers_y,
        'height': heights,
        'cluster': labels
    })

    
    
    return cluster_centers, labels, earth_points

def vis_earthwork(n_clusters=7, show_centers=True, height_threshold=0.5, if_show=True):
    """
    可视化挖方区域，将挖高设色和聚类结果分开显示
    """
    earth_result = get_road_earth()
    
    n_clusters=int(n_clusters)
    cluster_centers, labels, earth_points = cluster_earthwork(earth_result, n_clusters, height_threshold)

    if if_show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

        cmap = plt.cm.viridis
        norm = Normalize(vmin=min(earth_points['height']), vmax=max(earth_points['height']))
        
        cluster_colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
        
        for _, row in earth_points.iterrows():
            x, y = row['x'], row['y']
            square_vertices = [
                (x - 2.0, y - 2.0),
                (x + 2.0, y - 2.0),
                (x + 2.0, y + 2.0),
                (x - 2.0, y + 2.0)
            ]
            
            height = max(0, row['height'])
            fill_color = cmap(norm(height))
            
            polygon = plt.Polygon(square_vertices, closed=True, 
                               facecolor=fill_color, edgecolor='black', 
                               alpha=0.7, linewidth=0.5)
            ax1.add_patch(polygon)
        
        norm_colorbar = Normalize(vmin=0, vmax=max(earth_points['height']))
        cbar1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm_colorbar, cmap=cmap), ax=ax1)
        cbar1.set_label('挖方高度')
        
        ax1.set_title('挖方区域高度热图')
        ax1.set_xlabel('X 坐标', fontsize=18)
        ax1.set_ylabel('Y 坐标', fontsize=18)
        ax1.tick_params(axis='both', labelsize=18)
        ax1.axis('equal')
        
        for _, row in earth_points.iterrows():
            x, y = row['x'], row['y']
            cluster_id = int(row['cluster'])
            
            square_vertices = [
                (x - 2.0, y - 2.0),
                (x + 2.0, y - 2.0),
                (x + 2.0, y + 2.0),
                (x - 2.0, y + 2.0)
            ]
            
            fill_color = cluster_colors[cluster_id]
            
            polygon = plt.Polygon(square_vertices, closed=True, 
                               facecolor=fill_color, edgecolor='black', 
                               alpha=0.7, linewidth=0.5)
            ax2.add_patch(polygon)

        if show_centers:
            for i, center in enumerate(cluster_centers):
                ax2.scatter(center[0], center[1], s=200, c='red', marker='X', 
                            edgecolor='black', linewidth=2)
                ax2.annotate(f'装卸点 {i+1}', (center[0], center[1]), 
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=12, fontweight='bold')
        
        ax2.set_title('挖方区域聚类分析与装卸点位置')
        ax2.set_xlabel('X 坐标', fontsize=18)
        ax2.set_ylabel('Y 坐标', fontsize=18)

        ax2.tick_params(axis='both', labelsize=18)

        handles = [plt.Line2D([0], [0], color=cluster_colors[i], lw=4, 
                            label=f'聚类 {i+1}') for i in range(n_clusters)]
        ax2.legend(handles=handles, loc='lower right', ncol=3)

        ax2.axis('equal')

        plt.suptitle('挖方区域分析', fontsize=16)
        plt.savefig('挖方区域分析11.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
    
    return cluster_centers, earth_points

def analyze_loading_points(n_clusters=7,height_threshold=0.5,if_show=True):
    """分析装卸点的详细信息"""
    cluster_centers, earth_points = vis_earthwork(n_clusters=n_clusters, height_threshold=height_threshold,if_show=if_show)
    
    print("\n装卸点详细信息:")
    print("-" * 50)
    
    for i, center in enumerate(cluster_centers):
        cluster_data = earth_points[earth_points['cluster'] == i]
        
        total_height = cluster_data['height'].sum()
        avg_height = cluster_data['height'].mean()
        point_count = len(cluster_data)
        
        print(f"\n装卸点 {i+1}:")
        print(f"  坐标: X={center[0]:.2f}, Y={center[1]:.2f}")
        print(f"  覆盖挖方区域数量: {point_count}")
        print(f"  平均挖方高度: {avg_height:.2f}")
        print(f"  总体挖方高度: {total_height:.2f}")
    
    return cluster_centers, earth_points

def run_for_simulation_road(n_clusters, height_threshold, if_show, num_vehicles, split_mode, vehicle_capacity, max_iterations, 
                      num_ants, alpha, beta, rho, Q, time_weight, cost_weight, v, price_hour, digging_v,id):
    """
    运行蚁群算法，对所有挖方区域进行优化配送路径规划
    
    参数:
    - n_clusters: 聚类数量（装卸点数量）
    - height_threshold: 挖方高度阈值，用于筛选有效挖方点
    - if_show: 是否显示可视化结果
    - num_vehicles: 总车辆数量
    - split_mode: 车辆分配模式，可选"ratio"（按比例分配）或"random"（随机分配）
    - vehicle_capacity: 每辆车的装载容量
    - max_iterations: 蚁群算法最大迭代次数
    - num_ants: 蚂蚁数量
    - alpha: 信息素重要程度参数
    - beta: 启发式信息重要程度参数
    - rho: 信息素蒸发系数
    - Q: 信息素增加强度系数
    - time_weight: 时间权重（优化目标中时间因素的权重）
    - cost_weight: 成本权重（优化目标中成本因素的权重）
    - v: 车辆行驶速度（米/小时）
    - price_hour: 每小时运营成本（元/小时）
    - digging_v: 挖掘速度（立方米/小时）
    - id: 粒子编号
    
    返回:
    - 包含优化结果的字典，包括各区域的时间、成本、加权得分和路径
    """
    results={
        "time":[],
        "cost":[],
        "weighted":[],
        "routes":[]
    }
    cluster_centers, earth_points = vis_earthwork(n_clusters=n_clusters, height_threshold=height_threshold, if_show=if_show)
    total_vehicles = int(num_vehicles) 
    
    cluster_heights = []
    for i in range(len(cluster_centers)):
        cluster_data = earth_points[earth_points['cluster'] == i]
        total_height = cluster_data['height'].sum()
        cluster_heights.append(total_height)
    
    allocation_modes = {
        "随机分配": [],
        "按比例分配": []
    }
    
    remaining_vehicles = total_vehicles - len(cluster_centers)
    random_allocation = [1] * len(cluster_centers)
    
    if remaining_vehicles > 0:
        random_indices = np.random.choice(len(cluster_centers), remaining_vehicles, replace=True)
        for idx in random_indices:
            random_allocation[idx] += 1
    
    allocation_modes["随机分配"] = random_allocation
    
    proportional_allocation = [1] * len(cluster_centers)
    remaining_vehicles = total_vehicles - len(cluster_centers)
    
    if remaining_vehicles > 0 and sum(cluster_heights) > 0:
        height_proportions = [h / sum(cluster_heights) for h in cluster_heights]
        vehicle_distribution = [int(p * remaining_vehicles) for p in height_proportions]
        
        while sum(vehicle_distribution) < remaining_vehicles:
            errors = [(p * remaining_vehicles) - v for p, v in zip(height_proportions, vehicle_distribution)]
            idx = errors.index(max(errors))
            vehicle_distribution[idx] += 1
        
        for i in range(len(cluster_centers)):
            proportional_allocation[i] += vehicle_distribution[i]
    
    allocation_modes["按比例分配"] = proportional_allocation
    
    logger.info(f"粒子{id}车辆分配方案:")
    logger.info("-" * 50)
    for mode, allocation in allocation_modes.items():
        logger.info(f"\n{mode}:")
        for i, num in enumerate(allocation):
            logger.info(f" 粒子{id} 聚类 {i+1}: {num} 辆车")
    
    for i in range(len(cluster_centers)):
        cluster_data = earth_points[earth_points['cluster'] == i]
        
        if split_mode == "ratio":   
            vehicles_for_cluster = allocation_modes["按比例分配"][i]
        elif split_mode == "random":
            vehicles_for_cluster = allocation_modes["随机分配"][i]
        
        logger.info(f"\n正在为粒子{id} 聚类 {i+1} 运行算法 (分配 {vehicles_for_cluster} 辆车)...")
        best_routes, best_time, best_cost, best_weighted=run_for_one_region(num_customers=len(cluster_data), 
                           vehicle_capacity=vehicle_capacity, 
                           num_vehicles=vehicles_for_cluster,
                           max_iterations=max_iterations, 
                           num_ants=num_ants, 
                           alpha=alpha, 
                           beta=beta, 
                           rho=rho, 
                           Q=Q, 
                           customer_coords=cluster_data[['x', 'y']].values, 
                           demands_height=cluster_data['height'].values, 
                           time_weight=time_weight, 
                           cost_weight=cost_weight,
                           if_show=if_show, 
                           v=v, 
                           price_hour=price_hour,
                           digging_v=digging_v, 
                           cluster_center=cluster_centers[i],
                           id=id)
        results["time"].append(best_time)
        results["cost"].append(best_cost)
        results["weighted"].append(best_weighted)
        results["routes"].append(best_routes)

    return results

def load_config(cfg_path):
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg

if __name__ == "__main__":
    cfg = load_config('config.yaml')
    run_for_simulation_road(cfg)