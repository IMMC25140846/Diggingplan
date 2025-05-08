import ezdxf
import json

# 读取DXF文件
doc = ezdxf.readfile("航道开挖图.dxf")
msp = doc.modelspace()

# 创建一个字典来存储按图层分类的几何图形和标签
layer_entities = {}

# 遍历模型空间中的所有实体
for entity in msp:
    layer_name = entity.dxf.layer
    if layer_name not in layer_entities:
        layer_entities[layer_name] = {
            'lines': [],
            'points': [],
            'circles': [],
            'labels': [],
            'lwpolylines': [],
            'polylines': [],  # 添加常规多段线类型
            'mtext': []  #添加多行文字类型
        }

    # 根据实体类型，将实体添加到对应的字典分类中
    if isinstance(entity, ezdxf.entities.Line):
        layer_entities[layer_name]['lines'].append({
            'start': [entity.dxf.start[0], entity.dxf.start[1], entity.dxf.start[2]],
            'end': [entity.dxf.end[0], entity.dxf.end[1], entity.dxf.end[2]],
        })
    elif isinstance(entity, ezdxf.entities.Point):
        layer_entities[layer_name]['points'].append({
            'location': [entity.dxf.location[0], entity.dxf.location[1], entity.dxf.location[2]],
        })
    elif isinstance(entity, ezdxf.entities.Circle):
        layer_entities[layer_name]['circles'].append({
            'center': [entity.dxf.center[0], entity.dxf.center[1], entity.dxf.center[2]],
            'radius': entity.dxf.radius
        })
    elif isinstance(entity, ezdxf.entities.MText):  # 添加多行文字类型
        layer_entities[layer_name]['mtext'].append({
            'text': entity.plain_text(),
            'insertion_point': [entity.dxf.insert[0], entity.dxf.insert[1], entity.dxf.insert[2]]
        })
    elif isinstance(entity, ezdxf.entities.Text):
        layer_entities[layer_name]['labels'].append({
            'text': entity.dxf.text,
            'insertion_point': [entity.dxf.insert[0], entity.dxf.insert[1], entity.dxf.insert[2]]
        })
    elif isinstance(entity, ezdxf.entities.Polyline):
        points = []
        for vertex in entity.vertices():
            points.append([vertex.dxf.location[0], vertex.dxf.location[1], vertex.dxf.location[2]])
        layer_entities[layer_name]['polylines'].append({
            'points': points
        })
    elif isinstance(entity, ezdxf.entities.LWPolyline):
        points = []
        for point in entity:
            points.append([point[0], point[1]])
        layer_entities[layer_name]['lwpolylines'].append({
            'points': points
        })

# 将分层数据写入JSON文件
with open('dxf_layers_data.json', 'w', encoding='utf-8') as json_file:
    json.dump(layer_entities, json_file, ensure_ascii=False, indent=4)

print("几何图形和标签信息已按图层保存到'dxf_layers_data.json'文件中。")