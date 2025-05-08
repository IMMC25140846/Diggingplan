import json
import matplotlib.pyplot as plt
import re
from P1_river_slope import get_slope
import numpy as np
from scipy.spatial import Delaunay

X_OFFSET = 486400
Y_OFFSET = 2427000

def get_earthwork():
    # 读取 JSON 文件
    with open("dxf_layers_data.json", 'r', encoding='utf-8') as json_file:
        layer_entities = json.load(json_file)
    lwpolylines = []
    texts = {
        "insertion_point": [],
        "水位": [],
        "硬底差": [],
        "square_point":[],
        "z": [],
        "height": []
    }
    for layer_name, layer in layer_entities.items():
        if layer_name == "船位":
            for lwpolyline in layer["lwpolylines"]:
                lwpolylines.append(lwpolyline["points"])
        elif layer_name == "开挖船位文字标注":
            for mtext in layer["mtext"]:
                texts["insertion_point"].append(mtext["insertion_point"])
                text = mtext["text"]
                #匹配"水位"后和"m"前的浮点数
                match = re.search(r"水位([+-]?\d+(?:\.\d+)?)(m)", text)
                if  match:
                    texts["水位"].append(float(match.group(1)))
                else:
                    texts["水位"].append(0)
                #匹配"差"后和"m"前的浮点数
                match = re.search(r"差([+-]?\d+(?:\.\d+)?)(m)", text)
                if match:
                    texts["硬底差"].append(float(match.group(1)))
                else:
                    texts["硬底差"].append(0)
    base_z = 7.61
    square = []
    true_square = []
    for polyline in lwpolylines:
        x = [point[0] for point in polyline]
        y = [point[1] for point in polyline]
        square.append((min(x), min(y), max(x), max(y)))
        #画长方形需要5个点
        x.append(x[0])
        y.append(y[0])
        true_square.append([x, y])
    #根据insertion_point将texts中的浮点数和square匹配
    for i in range(len(texts["insertion_point"])):
        for j in range(len(square)):
            if texts["insertion_point"][i][0] >= square[j][0] and texts["insertion_point"][i][0] <= square[j][2] and texts["insertion_point"][i][1] >= square[j][1] and texts["insertion_point"][i][1] <= square[j][3]:
                if texts["水位"][i] > base_z:
                    texts["height"].append(base_z - texts["水位"][i] + texts["硬底差"][i])
                else:
                    texts["height"].append(base_z - texts["水位"][i] - texts["硬底差"][i])
                texts["square_point"].append(true_square[j])
                texts["z"].append(-base_z + texts["height"][-1])
                break
    return texts


if __name__ == "__main__":

  
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False  

    results = get_slope()
    
    def adjust_coordinates(x, y):
        return x - X_OFFSET, y - Y_OFFSET
    
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

    texts = get_earthwork()
    
    
    # fig = plt.figure(figsize=(15, 12))
    # ax = fig.add_subplot(111, projection='3d')
    
    # # 1. 绘制方格挖方区域
    # for i in range(len(texts["square_point"])):
    #     x = [texts["square_point"][i][0] - X_OFFSET, texts["square_point"][i][2] - X_OFFSET, 
    #         texts["square_point"][i][2] - X_OFFSET, texts["square_point"][i][0] - X_OFFSET, 
    #         texts["square_point"][i][0] - X_OFFSET]
    #     y = [texts["square_point"][i][1] - Y_OFFSET, texts["square_point"][i][1] - Y_OFFSET, 
    #         texts["square_point"][i][3] - Y_OFFSET, texts["square_point"][i][3] - Y_OFFSET, 
    #         texts["square_point"][i][1] - Y_OFFSET]
    #     z = [texts["z"][i]] * 5
    #     ax.plot(x, y, z, linewidth=1.5, color='blue')
    
    # # 2. 绘制三角剖分坡面
    
    # all_points = []
    # for layer_name, points in adjusted_new_polylines.items():
    #     all_points.extend([(p[0], p[1], p[2]) for p in points])
    # for point in adjusted_sidelines_1:
    #     all_points.append((point[0], point[1], results["base_z"]))
    # for point in adjusted_sidelines_2:
    #     all_points.append((point[0], point[1], results["base_z"]))
    # for point in adjusted_middle_lines:
    #     all_points.append((point[0], point[1], results["base_z"]))

    
    # points_2d = np.array([(p[0], p[1]) for p in all_points])
    # tri = Delaunay(points_2d)

   
    # surf = ax.plot_trisurf(
    #     points_2d[:,0], points_2d[:,1], np.array([p[2] for p in all_points]),
    #     triangles=tri.simplices,
    #     cmap='terrain', alpha=0.6
    # )
    # colorbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='高程 (m)')
    # # 3. 绘制三维线框
    # for layer_name, points in adjusted_new_polylines.items():
    #     xs = [point[0] for point in points]
    #     ys = [point[1] for point in points]
    #     zs = [point[2] for point in points]
    #     ax.plot(xs, ys, zs, label=layer_name, linewidth=1.5)
    # x = [point[0] for point in adjusted_sidelines_1]
    # y = [point[1] for point in adjusted_sidelines_1]
    # ax.plot(x, y, results["base_z"], label="航道线左侧", linewidth=1.5)

    # x = [point[0] for point in adjusted_sidelines_2]
    # y = [point[1] for point in adjusted_sidelines_2]
    # ax.plot(x, y, results["base_z"], label="航道线右侧", linewidth=1.5)

    # x = [point[0] for point in adjusted_middle_lines]
    # y = [point[1] for point in adjusted_middle_lines]
    # ax.plot(x, y, results["base_z"], label="航道中线", linewidth=1.5)

    # ax.legend(fontsize=10, loc='best')
    
    # ax.set_xlabel('X', fontsize=12)
    # ax.set_ylabel('Y', fontsize=12)
    # ax.set_zlabel('Z', fontsize=12)
    # plt.title('三维挖方综合示意图', fontsize=14)
    # ax.view_init(elev=30, azim=45)
    
    # plt.tight_layout()
    # plt.show()
    
    # plt.savefig("3d_plot_combined.png", dpi=300, bbox_inches='tight')

    volume = 0
    for i in range(len(texts["square_point"])):
        x_coords = texts["square_point"][i][0]
        y_coords = texts["square_point"][i][1]
        square_area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
        volume += square_area * texts["height"][i]
    print("挖方体积为：", volume)