import json
import matplotlib.pyplot as plt
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import LinearNDInterpolator
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata
from matplotlib import cm

def calculate_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def calculate_x(y,point1, point2):
    return (y - point1[1]) * (point2[0] - point1[0]) / (point2[1] - point1[1]) + point1[0]

def calculate_z(y,point1, point2):
    return (y - point1[1]) * (point2[2] - point1[2]) / (point2[1] - point1[1]) + point1[2]

def find_x(y, points):
    if(points[0][1] > y):
        return points[0][0]
    for index,point in enumerate(points):
        if(point[1] > y):
            return calculate_x(y,points[index-1],point)
    return points[-1][0]

def find_z(y, points):
    if(points[0][1] > y):
        return points[0][2]
    for index,point in enumerate(points):
        if(point[1] > y):
            return calculate_z(y,points[index-1],point)
    return points[-1][2]

def find_closest_point(point, points,middle_lines):
    is_left = point[0] < find_x(point[1], middle_lines)
    new_points = [p for p in points if (p[0] > point[0] if is_left else p[0] < point[0])]
    #寻找距离最近的点，返回4维point坐标
    if len(new_points) == 0:
        return (point[0], point[1], 0, 0)
    closest_point = min(new_points, key=lambda p: calculate_distance(point, p))
    return (point[0], point[1], closest_point[2], closest_point[3])


def get_slope():

    # 读取 JSON 文件
    with open('dxf_layers_data.json', 'r', encoding='utf-8') as json_file:
        layer_entities = json.load(json_file)

    # 指定要遍历的图层名称
    specified_layers = ["开口线", "水下变坡线", "压顶线", "ZH-25-施工图"]

    # 初始化最小和最大坐标值
    x_min = float('inf')
    x_max = float('-inf')
    y_min = float('inf')
    y_max = float('-inf')

    # 提取指定图层所有线的起点和终点坐标，并更新最小和最大坐标值
    lines = []
    middle_lines=[]
    sidelines_1 = []
    sidelines_2 = []
    insertion_points = []
    for layer_name, layer in layer_entities.items():
        if layer_name == "文字标注":
            for label in layer['labels']:
                insertion_points.append([label['insertion_point'][0], label['insertion_point'][1], label['text'].split(":")[0], re.search(r'^[-+]?\d+\.?\d*', label['text'].split(":")[1]).group(0)])
        if layer_name == "90m-520m半径方案925":
            for label in layer['labels']:
                insertion_points.append([label['insertion_point'][0], label['insertion_point'][1], label['text'].split(":")[0], re.search(r'^[-+]?\d+\.?\d*', label['text'].split(":")[1]).group(0)])
        if layer_name == "8500~13000推荐方案":
            for label in layer['labels']:
                insertion_points.append([label['insertion_point'][0], label['insertion_point'][1], label['text'].split(":")[0], re.search(r'^[-+]?\d+\.?\d*', label['text'].split(":")[1]).group(0)])
        if layer_name in specified_layers:
            for line in layer['lines']:
                start = line['start']
                end = line['end']
                lines.append((start, end))
                if layer_name == "ZH-25-施工图":
                    middle_lines.append(((start[0]+end[0])/2,(start[1]+end[1])/2))
                    sidelines_1.append((start[0],start[1]))
                    sidelines_2.append((end[0],end[1]))
                x_min = min(x_min, start[0], end[0])
                x_max = max(x_max, start[0], end[0])
                y_min = min(y_min, start[1], end[1])
                y_max = max(y_max, start[1], end[1])

    # 提取指定图层所有多段线的数据，并更新最小和最大坐标值
    lwpolylines = {
        "开口线": [],
        "水下变坡线": [],
        "压顶线": [],
    }
    for layer_name, layer in layer_entities.items():
        if layer_name in lwpolylines.keys():
            for lwpolyline in layer.get('lwpolylines', []):
                points = lwpolyline['points']
                for point in points:
                    lwpolylines[layer_name].append((point[0], point[1]))
                    x_min = min(x_min, point[0])
                    x_max = max(x_max, point[0])
                    y_min = min(y_min, point[1])
                    y_max = max(y_max, point[1])


    middle_lines = sorted(middle_lines, key=lambda point: point[1]) # 按照y坐标排序
    sidelines_1 = sorted(sidelines_1, key=lambda point: point[1]) # 按照y坐标排序
    sidelines_2 = sorted(sidelines_2, key=lambda point: point[1]) # 按照y坐标排序

    #对每个图层的点分为左右两侧按y轴进行排序，分别存储左右点
    for layer_name, points in lwpolylines.items():
        left_points = []
        right_points = []
        for point in points:
            if point[0] < find_x(point[1], middle_lines):
                left_points.append(point)
            else:
                right_points.append(point)
        left_points.sort(key=lambda point: point[1])
        right_points.sort(key=lambda point: point[1])
        lwpolylines[layer_name] = [left_points, right_points]

    closest_point = {
        "开口线左侧": [],
        "开口线右侧": [],
        "水下变坡线左侧": [],
        "水下变坡线右侧": [],
        "压顶线左侧": [],
        "压顶线右侧": [],
    }
    for layer_name, points in lwpolylines.items():
        for point in points[0]:
            if layer_name == "开口线":
                closest_point[layer_name+"左侧"].append(find_closest_point(point, insertion_points,middle_lines))
            elif layer_name == "水下变坡线":
                closest_point[layer_name+"左侧"].append(find_closest_point(point, insertion_points,middle_lines))
            elif layer_name == "压顶线":
                closest_point[layer_name+"左侧"].append(find_closest_point(point, insertion_points,middle_lines))
        for point in points[1]:
            if layer_name == "开口线":
                closest_point[layer_name+"右侧"].append(find_closest_point(point, insertion_points,middle_lines))
            elif layer_name == "水下变坡线":
                closest_point[layer_name+"右侧"].append(find_closest_point(point, insertion_points,middle_lines))
            elif layer_name == "压顶线":
                closest_point[layer_name+"右侧"].append(find_closest_point(point, insertion_points,middle_lines))

    base_z = -7.61
    new_polylines = {
        "开口线左侧": [],
        "开口线右侧": [],
        "水下变坡线左侧": [],
        "水下变坡线右侧": [],
        "压顶线左侧": [],
        "压顶线右侧": [],
    }

    for layer_name, points in closest_point.items():
        if "水下变坡线" in layer_name or "压顶线" in layer_name:
            for point in points:
                x = find_x(point[1], sidelines_1 if point[0] < find_x(point[1], middle_lines) else sidelines_2)
                new_polylines[layer_name].append([point[0], point[1], base_z + abs(point[0] - x)/float(point[3]) if point[3] != 0 else base_z])

    for layer_name, points in closest_point.items():
        if layer_name == "开口线左侧":
            for point in points:
                x = find_x(point[1], new_polylines["水下变坡线左侧"])
                z = find_z(point[1], new_polylines["水下变坡线左侧"])
                if x != new_polylines["水下变坡线左侧"][0][0] and x != new_polylines["水下变坡线左侧"][-1][0]:
                    new_polylines[layer_name].append([point[0], point[1], z + abs(point[0] - x)/float(point[3]) if point[3] != 0 else z])
                else:
                    x = find_x(point[1], sidelines_1)
                    new_polylines[layer_name].append([point[0], point[1], base_z + abs(point[0] - x)/float(point[3]) if point[3] != 0 else base_z])
        elif layer_name == "开口线右侧":
            for point in points:
                x = find_x(point[1], new_polylines["水下变坡线右侧"])
                z = find_z(point[1], new_polylines["水下变坡线右侧"])
                if x != new_polylines["水下变坡线右侧"][0][0] and x != new_polylines["水下变坡线右侧"][-1][0]:
                    new_polylines[layer_name].append([point[0], point[1], z + abs(point[0] - x)/float(point[3]) if point[3] != 0 else z])
                else:
                    x = find_x(point[1], sidelines_2)
                    new_polylines[layer_name].append([point[0], point[1], base_z + abs(point[0] - x)/float(point[3]) if point[3] != 0 else base_z])


    results={}
    results["new_polylines"]=new_polylines
    results["middle_lines"]=middle_lines
    results["sidelines_1"]=sidelines_1
    results["sidelines_2"]=sidelines_2
    results["base_z"]=base_z
    results["x_min"]=x_min
    results["x_max"]=x_max
    results["y_min"]=y_min
    results["y_max"]=y_max
    return results

if __name__ == "__main__":
    results = get_slope()
    # 添加坐标偏移常量
    X_OFFSET = 486400
    Y_OFFSET = 2427000

    # 在绘图前调整坐标的函数
    def adjust_coordinates(x, y):
        return x - X_OFFSET, y - Y_OFFSET

    # 在绘制三维图之前调整所有点的坐标
    adjusted_new_polylines = {}
    for layer_name, points in results["new_polylines"].items():
        adjusted_points = []
        for point in points:
            adjusted_x, adjusted_y = adjust_coordinates(point[0], point[1])
            adjusted_points.append([adjusted_x, adjusted_y, point[2]])
        adjusted_new_polylines[layer_name] = adjusted_points

    adjusted_sidelines_1 = []
    for point in results["sidelines_1"]:
        adjusted_x, adjusted_y = adjust_coordinates(point[0], point[1])
        adjusted_sidelines_1.append((adjusted_x, adjusted_y))

    adjusted_sidelines_2 = []
    for point in results["sidelines_2"]:
        adjusted_x, adjusted_y = adjust_coordinates(point[0], point[1])
        adjusted_sidelines_2.append((adjusted_x, adjusted_y))

    adjusted_middle_lines = []
    for point in results["middle_lines"]:
        adjusted_x, adjusted_y = adjust_coordinates(point[0], point[1])
        adjusted_middle_lines.append((adjusted_x, adjusted_y))

    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False 
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 保存开口线左侧和开口线右侧的原始坐标到CSV文件
    import csv
    
    # 获取原始坐标（取消坐标变换）
    with open('边坡坐标.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['X', 'Y', 'Z'])  # 写入表头
        for layer_name, points in results["new_polylines"].items():
            for point in points:
                writer.writerow([ point[0], point[1], point[2]])  # 写入原始坐标
    
    # 三维线图
    for layer_name, points in adjusted_new_polylines.items():
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        zs = [point[2] for point in points]
        ax.plot(xs, ys, zs, label=layer_name)

    x = [point[0] for point in adjusted_sidelines_1]
    y = [point[1] for point in adjusted_sidelines_1]
    ax.plot(x, y, results["base_z"], label="航道线左侧")

    x = [point[0] for point in adjusted_sidelines_2]
    y = [point[1] for point in adjusted_sidelines_2]
    ax.plot(x, y, results["base_z"], label="航道线右侧")

    x = [point[0] for point in adjusted_middle_lines]
    y = [point[1] for point in adjusted_middle_lines]
    ax.plot(x, y, results["base_z"], label="航道中线")

    ax.legend()
    
    plt.savefig('三维线图.pdf', format='pdf', bbox_inches='tight')
    #plt.show()

   
    # 合并所有点
    all_points = []
    for layer_name, points in adjusted_new_polylines.items():
        all_points.extend([(p[0], p[1], p[2]) for p in points])
    for point in adjusted_sidelines_1:
        all_points.append((point[0], point[1], results["base_z"]))
    for point in adjusted_sidelines_2:
        all_points.append((point[0], point[1], results["base_z"]    ))
    for point in adjusted_middle_lines:
        all_points.append((point[0], point[1], results["base_z"]))

    
    points_2d = np.array([(p[0], p[1]) for p in all_points])
    tri = Delaunay(points_2d)

    # 绘制坡面
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(
        points_2d[:,0], points_2d[:,1], np.array([p[2] for p in all_points]),
        triangles=tri.simplices,
        cmap='terrain', alpha=0.8
    )

    
    colorbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='高程 (m)')

    
    for layer_name, points in adjusted_new_polylines.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        ax.plot(xs, ys, zs, color='black', linewidth=0.5)

    plt.title("三维坡面图")
   
    plt.savefig('三维坡面图.pdf', format='pdf', bbox_inches='tight')
    
    plt.figure(figsize=(10, 6))
    for layer_name, points in adjusted_new_polylines.items():
        # 提取XY坐标和高程Z值
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        z = [p[2] for p in points]
        scatter = plt.scatter(x, y, c=z, cmap='viridis', s=5, label=layer_name)

    # 添加航道线
    plt.plot([p[0] for p in adjusted_sidelines_1], [p[1] for p in adjusted_sidelines_1], 'k--', lw=1, label="航道边线")
    plt.plot([p[0] for p in adjusted_sidelines_2], [p[1] for p in adjusted_sidelines_2], 'k--', lw=1)
    plt.plot([p[0] for p in adjusted_middle_lines], [p[1] for p in adjusted_middle_lines], 'r-', lw=1, label="航道中线")

    plt.colorbar(scatter, label='高程 (m)')
    plt.title("二维平面投影（颜色表示高程）")
    plt.legend()
    

    adjusted_x_min = results["x_min"] - X_OFFSET
    adjusted_x_max = results["x_max"] - X_OFFSET
    adjusted_y_min = results["y_min"] - Y_OFFSET
    adjusted_y_max = results["y_max"] - Y_OFFSET
    plt.xlim(adjusted_x_min, adjusted_x_max)
    plt.ylim(adjusted_y_min, adjusted_y_max)
    
    plt.savefig('二维平面投影图.pdf', format='pdf', bbox_inches='tight')
    
    x = np.array([p[0] for p in all_points])
    y = np.array([p[1] for p in all_points])
    z = np.array([p[2] for p in all_points])

    
    xi = np.linspace(min(x)-1, max(x)+1, 200)  
    yi = np.linspace(min(y)-1, max(y)+1, 200)  
    X, Y = np.meshgrid(xi, yi)

    Z = griddata((x, y), z, (X, Y), method='linear')

    plt.figure(figsize=(12, 8))

    levels = np.linspace(z.min()-0.5, z.max()+0.5, 15)  # 根据实际高程范围设定
    contour = plt.contourf(X, Y, Z, levels=levels, cmap=cm.terrain, extend='both')
    plt.colorbar(contour, label='高程 (m)', shrink=0.8)
 
    C = plt.contour(X, Y, Z, levels=levels, colors='black', linewidths=1.5)
    plt.clabel(C, inline=True, fontsize=16, fmt='%.1f')

    # plt.title("高程等高线图")
    plt.xlabel('x (m)',fontsize=20)
    plt.ylabel('y (m)',fontsize=20)
    plt.grid(True, linestyle=':', alpha=0.3)
   
    plt.xlim(adjusted_x_min, adjusted_x_max)
    plt.ylim(adjusted_y_min, adjusted_y_max)
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    plt.tight_layout()
    
    plt.savefig('高程等高线图.pdf', format='pdf', bbox_inches='tight')
    # plt.show()

