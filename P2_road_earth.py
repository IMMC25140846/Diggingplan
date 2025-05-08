import json
import numpy as np
from scipy.interpolate import interp1d
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    


def get_road_earth():
    # 读取 JSON 文件
    with open('gradient_layers_data.json', 'r', encoding='utf-8') as json_file:
        layer_entities = json.load(json_file)

   
    axis_lines = layer_entities["横_轴线"]["lines"]
    axis_x_coordinates = [line["start"][0] for line in axis_lines]

   
    design_lines = layer_entities["横_设计线"]["lwpolylines"]
    ground_lines = layer_entities["横_原地线"]["lwpolylines"]

    
    reference_points = []
    start_index = 0  
    ans = 0  

    for axis_x in axis_x_coordinates:
        for i in range(start_index, len(design_lines)):  
            polyline = design_lines[i]
            point = polyline["points"][0]
            
            if abs(point[0] - axis_x) <= 0.05:
                ans += 1
                if ans == 2:  
                    reference_points.append(point)
                    start_index = i + 1  
                    ans = 0
                    break

    # 将参考点映射到真实世界坐标
    real_world_coordinates = []
    z_values = [
        188.325, 188.539, 188.753, 188.968, 189.182, 189.396, 189.61, 189.824,
        190.039, 190.253, 190.467, 190.681, 190.896, 191.11, 191.324, 191.538,
        191.752, 191.967, 192.181, 192.395, 192.609, 192.823, 193.038, 193.252,
        193.466, 193.68
    ]  # 真实世界的 z 坐标

    x_start = 520  
    x_increment = 20 

    for i, point in enumerate(reference_points):
        real_x = x_start + i * x_increment
        real_y = 0  
        real_z = z_values[i]  
        real_world_coordinates.append((real_x, real_y, real_z))

 
    An = [3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2]

  
    design_points_by_reference = {i: [] for i in range(len(reference_points))}
    ground_points_by_reference = {i: [] for i in range(len(reference_points))}

    for polyline in design_lines:
        for vertex in polyline["points"]:
            x = vertex[0]
            y = vertex[1]

            
            column = int((x - 4556) / 258)
            if column < 0 or column >= len(An):
                continue

            sum_indices = sum(An[:column + 1])
            possible_indices = range(sum_indices - An[column], sum_indices)

            for idx in possible_indices:
                if idx < 0 or idx >= len(reference_points):
                    continue

                ref_point = reference_points[idx]
                if y > ref_point[1] - 3:
                    design_points_by_reference[idx].append(vertex)
                    break

  
    for polyline in ground_lines:
        for vertex in polyline["points"]:
            x = vertex[0]
            y = vertex[1]

            column = int((x - 4556) / 258)
            if column < 0 or column >= len(An):
                continue

            sum_indices = sum(An[:column + 1])
            possible_indices = range(sum_indices - An[column], sum_indices)

            for idx in possible_indices:
                if idx < 0 or idx >= len(reference_points):
                    continue

                ref_point = reference_points[idx]
                if y > ref_point[1] - 3:
                    ground_points_by_reference[idx].append(vertex)
                    break

    
    results = {}


    for ref_idx, ref_point in enumerate(reference_points):
      
        design_points = design_points_by_reference[ref_idx]

        
        design_points = sorted(design_points, key=lambda p: (p[0], -p[1]))
        unique_design_points = []
        prev_x = None
        for point in design_points:
            if prev_x is None or point[0] != prev_x:
                unique_design_points.append(point)
                prev_x = point[0]

        design_x = [p[0] for p in unique_design_points]
        design_y = [p[1] for p in unique_design_points]
        design_interp = interp1d(design_x, design_y, kind='linear', fill_value="extrapolate")

        ground_points = ground_points_by_reference[ref_idx]

       
        ground_points = sorted(ground_points, key=lambda p: p[0])
        ground_x = [p[0] for p in ground_points]
        ground_y = [p[1] for p in ground_points]

        
        x_min, x_max = min(design_x), max(design_x)
        x_values = np.linspace(x_min, x_max, 100)

        
        delta_y = []
        for x in x_values:
   
            design_y_value = design_interp(x)

  
            closest_idx = np.argmin([abs(gx - x) for gx in ground_x])
            ground_y_value = ground_y[closest_idx]


            delta_y_value =  ground_y_value - design_y_value 

            x0, y0, z0 = real_world_coordinates[ref_idx]
            x_real = x0
            y_real = x - ref_point[0]


            delta_y.append({
                "x_real": x_real,
                "y": y_real,
                "delta_y": delta_y_value
            })

        results[f"参考点{ref_idx + 1}"] = delta_y

    all_points = []
    for ref_data in results.values():
        for point in ref_data:
            all_points.append((point["x_real"], point["y"], point["delta_y"]))

    all_points = np.array(all_points)
    x_real_values = all_points[:, 0]
    y_real_values = all_points[:, 1]
    delta_y_values = all_points[:, 2]

    x_min, x_max = x_real_values.min(), x_real_values.max()
    y_min, y_max = y_real_values.min(), y_real_values.max()

    x_step = 2.0
    y_step = 2.0


    x_grid = np.arange(x_min, x_max + x_step, x_step)
    y_grid = np.arange(y_min, y_max + y_step, y_step)


    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)



    delta_y_mesh = griddata(
        (x_real_values, y_real_values),  # 原始点的 (x, y)
        delta_y_values,                  # 原始点的 delta_y
        (x_mesh, y_mesh),                # 插值网格点
        method='linear'                  # 线性插值
    )


    original_grid_area = 1.0 * 1.0
    new_grid_area = x_step * y_step
    area_ratio = original_grid_area / new_grid_area
    

    delta_y_mesh = delta_y_mesh * area_ratio

    interpolated_results = []
    nan_count = 0
    valid_count = 0
    for i in range(x_mesh.shape[0]):
        for j in range(x_mesh.shape[1]):
            x_real = x_mesh[i, j]
            y_real = y_mesh[i, j]
            delta_y = delta_y_mesh[i, j]
            if not np.isnan(delta_y):  # 过滤掉插值结果中的 NaN 值
                interpolated_results.append([x_real, y_real, delta_y])
                valid_count += 1
            else:
                nan_count += 1
    interpolated_results = np.array(interpolated_results)

    total_excavation_volume = 0.0
    for i in range(delta_y_mesh.shape[0]): 
        for j in range(delta_y_mesh.shape[1]): 
            excavation_height = delta_y_mesh[i, j]
            
            if not np.isnan(excavation_height) and excavation_height > 0:
                cell_area = x_step * y_step
                cell_volume = excavation_height * cell_area
                total_excavation_volume += cell_volume

    print(f"\n估算的挖方总体积为: {total_excavation_volume:.3f} 立方米 ")


    return interpolated_results

if __name__ == "__main__":
    get_road_earth()
