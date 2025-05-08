import json
import matplotlib.pyplot as plt
import re

def extract_texts_from_dxf(dxf_json_file_path):
    # 读取 JSON 文件
    with open(dxf_json_file_path, 'r', encoding='utf-8') as json_file:
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

def calculate_volume(texts):
    volume = 0
    for i in range(len(texts["square_point"])):
        x = texts["square_point"][i][0][0:4]
        y = texts["square_point"][i][1][0:4]
        points = [(x[0], y[0]), (x[1], y[1]), (x[2], y[2]), (x[3], y[3])]        
        # 使用鞋带公式计算四边形面积
        sum1, sum2 = 0, 0
        n = len(points)
        for j in range(n):
            next_j = (j + 1) % n
            sum1 += points[j][0] * points[next_j][1]
            sum2 += points[j][1] * points[next_j][0]
        area = 0.5 * abs(sum1 - sum2)        
        # 累加体积（面积×高度）
        volume += area * texts["height"][i]
    return volume