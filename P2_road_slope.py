import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 等高线图绘制
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata

def get_road_slope():
    points={}

    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False    
    def calculate_distance(point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
    # 读取 JSON 文件
    with open('gradient_layers_data.json', 'r', encoding='utf-8') as json_file:
        layer_entities = json.load(json_file)

    axis_lines = layer_entities["横_轴线"]["lines"]
    axis_x_coordinates = [line["start"][0] for line in axis_lines]

    design_lines = layer_entities["横_设计线"]["lwpolylines"]

    reference_points = []
   
    start_index = 0  
    ans = 0  

    for axis_x in axis_x_coordinates:
        for i in range(start_index, len(design_lines)):  # 从 start_index 开始遍历
            polyline = design_lines[i]
            point = polyline["points"][0]
            
            if abs(point[0] - axis_x) <= 0.05:
                ans += 1
                if ans == 2:  # 只有出现两次才会记录
                    reference_points.append(point)
                    start_index = i + 1  # 下次从下一个 折线开始
                    ans = 0
                    break

    # 将参考点映射到真实世界坐标
    real_world_coordinates = []
    z_values = [
        188.325, 188.539, 188.753, 188.968, 189.182, 189.396, 189.61, 189.824,
        190.039, 190.253, 190.467, 190.681, 190.896, 191.11, 191.324, 191.538,
        191.752, 191.967, 192.181, 192.395, 192.609, 192.823, 193.038, 193.252,
        193.466, 193.68
    ]#读图一个一个列出来的z坐标

    x_start = 520  # x 坐标起始值
    x_increment = 20  # 每个点 x 坐标的增加的量

    for i, point in enumerate(reference_points):
        real_x = x_start + i * x_increment
        real_y = 0  
        real_z = z_values[i]  
        real_world_coordinates.append((real_x, real_y, real_z))

    An = [3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    converted_points = []

    for polyline in design_lines:
        for vertex in polyline["points"]:
            x = vertex[0]
            y = vertex[1]

            # 计算可能的参考点索引范围
            column = int((x - 4556) / 258)
            if column < 0 or column >= len(An):
                continue

            sum_indices = sum(An[:column + 1])
            possible_indices = range(sum_indices - An[column], sum_indices)

            # 找到对应的参考点
            for idx in possible_indices:
                if idx < 0 or idx >= len(reference_points):
                    continue

                ref_point = reference_points[idx]
                if y > ref_point[1] - 3:
                   
                    x0, y0, z0 = real_world_coordinates[idx]
                    x1 = x0
                    y1 = x - ref_point[0]
                    z1 = y - ref_point[1] + z0
                    converted_points.append((x1, y1, z1))
                    break

    # 提取三维坐标
    converted_x = [p[0] for p in converted_points]
    converted_y = [p[1] for p in converted_points]
    converted_z = [p[2] for p in converted_points]


    points["x"]=converted_x
    points["y"]=converted_y
    points["z"]=converted_z
    points["base_z"]=min(z_values)
    points["converted_points"]=converted_points


    return points

if __name__ == "__main__":
    points = get_road_slope()
    converted_x=points["x"]
    converted_y=points["y"]
    converted_z=points["z"]
    base_z=points["base_z"]
    converted_points=points["converted_points"]
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(converted_x, converted_y, converted_z, color='red', label='转化后的点', marker='o')

    ax.plot_trisurf(converted_x, converted_y, converted_z, color='blue', alpha=0.5, edgecolor='none')

    ax.set_xlabel('X 坐标 (真实世界)')
    ax.set_ylabel('Y 坐标 (真实世界)')
    ax.set_zlabel('Z 坐标 (真实世界)')
    ax.set_title('横_设计线顶点在真实世界坐标中的分布及拟合曲面')
    ax.legend()
    ax.grid(True)
    plt.show()

    points_2d = np.array([(p[0], p[1]) for p in converted_points])  
    tri = Delaunay(points_2d)


    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_trisurf(
        points_2d[:,0], 
        points_2d[:,1], 
        [p[2] for p in converted_points],
        triangles=tri.simplices,
        cmap='terrain', 
        alpha=0.7,
        edgecolor='none'
    )

    min_z = min([p[2] for p in converted_points])+5  
    y_min_indices = []
    y_max_indices = []
    y_coords = np.array([p[1] for p in converted_points])
    x_coords = np.array([p[0] for p in converted_points])
    unique_x = np.unique(x_coords)

    for x_val in unique_x:
        x_mask = (x_coords == x_val)
        if np.sum(x_mask) > 0:
            y_values = y_coords[x_mask]
            min_y_idx = np.where((x_coords == x_val) & (y_coords == np.min(y_values)))[0][0]
            max_y_idx = np.where((x_coords == x_val) & (y_coords == np.max(y_values)))[0][0]
            y_min_indices.append(min_y_idx)
            y_max_indices.append(max_y_idx)

    for idx_list, is_min in [(y_min_indices, True), (y_max_indices, False)]:
        for i in range(len(idx_list) - 1):
            idx1, idx2 = idx_list[i], idx_list[i + 1]
        
            p1 = converted_points[idx1]
            p2 = converted_points[idx2]
            
            
            p3 = (p2[0], p2[1], min_z)
            p4 = (p1[0], p1[1], min_z)
            
           
            segments = 50  
            
            
            z_min = min_z
            z_max_1 = p1[2]
            z_max_2 = p2[2]
            
            for j in range(segments):
                
                z1_bottom = z_min + (z_max_1 - z_min) * j / segments
                z1_top = z_min + (z_max_1 - z_min) * (j + 1) / segments
                
                z2_bottom = z_min + (z_max_2 - z_min) * j / segments
                z2_top = z_min + (z_max_2 - z_min) * (j + 1) / segments
                
                
                v1 = (p1[0], p1[1], z1_top)
                v2 = (p2[0], p2[1], z2_top)
                v3 = (p2[0], p2[1], z2_bottom)
                v4 = (p1[0], p1[1], z1_bottom)
                
               
                verts = [v1, v2, v3, v4]
                poly = Poly3DCollection([verts], alpha=0.7)
                
               
                avg_z = (z1_top + z2_top + z1_bottom + z2_bottom) / 4
                
                
                norm_z = (avg_z - min([p[2] for p in converted_points])) / (max([p[2] for p in converted_points]) - min([p[2] for p in converted_points]))
                color = plt.cm.terrain(norm_z)
                poly.set_color(color)
                ax.add_collection3d(poly)

    bottom_points = []
    for idx in y_min_indices:
        p = converted_points[idx]
        bottom_points.append((p[0], p[1], min_z))
    
    for idx in reversed(y_max_indices):
        p = converted_points[idx]
        bottom_points.append((p[0], p[1], min_z))
    
    if len(bottom_points) > 2:
        poly = Poly3DCollection([bottom_points], alpha=0.7)
        color = plt.cm.terrain(0.0)  
        poly.set_color(color)
        ax.add_collection3d(poly)

    ax.set_xlabel('X 坐标 (m)')
    ax.set_ylabel('Y 坐标 (m)')
    ax.set_zlabel('高程 (m)')
    ax.set_title('三维坡面模型与设计线框')
    fig.colorbar(surf, ax=ax, shrink=0.5, label='高程（米）')

    ax.view_init(elev=30, azim=-45)

    ax.set_zlim(min_z, max([p[2] for p in converted_points]) + 5)

    plt.tight_layout()
    plt.show()



    x = np.array(converted_x)
    y = np.array(converted_y)
    z = np.array(converted_z)  

    xi = np.linspace(min(x)-1, max(x)+1, 200)  
    yi = np.linspace(min(y)-1, max(y)+1, 200) 
    X, Y = np.meshgrid(xi, yi)

    Z = griddata((x, y), z, (X, Y), method='linear')

    plt.figure(figsize=(12, 8))


    levels = np.linspace(z.min()-0.5, z.max()+0.5, 15)  
    contour = plt.contourf(X, Y, Z, levels=levels, cmap=cm.terrain, extend='both')
    plt.colorbar(contour, label='高程 (m)', shrink=0.8)


    C = plt.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.5)
    plt.clabel(C, inline=True, fontsize=8, fmt='%.1f')

    plt.title('横_设计线高程等高线图')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.show()
