import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm


# --------------------------- 配置项 ---------------------------
voc_images_dir = '../datasets/VOCdevkit/VOC2012/JPEGImages'
voc_annotations_dir = '../datasets/VOCdevkit/VOC2012/Annotations'
classes_path = '../datasets/VOCdevkit/VOC2012/classes.txt'
output_dir = '../datasets/my_yolo_data'  # 最终输出的根目录
train_ratio = 0.8
# ------------------------------------------------------------

# 加载类别
with open(classes_path) as f:
    classes = [x.strip() for x in f.readlines()]
class_to_idx = {name: i for i, name in enumerate(classes)}

# 创建输出文件夹
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

# 获取所有 XML 文件
xml_files = [f for f in os.listdir(voc_annotations_dir) if f.endswith('.xml')]
random.shuffle(xml_files)

# 划分数据
split_index = int(len(xml_files) * train_ratio)
train_xmls = xml_files[:split_index]
val_xmls = xml_files[split_index:]

def convert_xml_to_yolo(xml_file, output_txt_file, image_w, image_h):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    with open(output_txt_file, 'w') as f:
        for obj in root.findall('object'):
            cls = obj.find('name').text
            if cls not in class_to_idx:
                continue
            cls_id = class_to_idx[cls]
            xml_box = obj.find('bndbox')
            xmin = float(xml_box.find('xmin').text)
            ymin = float(xml_box.find('ymin').text)
            xmax = float(xml_box.find('xmax').text)
            ymax = float(xml_box.find('ymax').text)
            # 计算yolo格式: x_center y_center width height (归一化)
            x_center = (xmin + xmax) / 2.0 / image_w
            y_center = (ymin + ymax) / 2.0 / image_h
            box_w = (xmax - xmin) / image_w
            box_h = (ymax - ymin) / image_h
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

def process_split(xml_list, split):
    print(f"开始处理 {split} 集，共 {len(xml_list)} 个样本...")
    for xml_file in tqdm(xml_list, desc=f"Processing {split}"):
        xml_path = os.path.join(voc_annotations_dir, xml_file)
        image_filename = xml_file.replace('.xml', '.jpg')
        image_path = os.path.join(voc_images_dir, image_filename)

        # 解析图像宽高
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        # 输出路径
        out_img_path = os.path.join(output_dir, 'images', split, image_filename)
        out_txt_path = os.path.join(output_dir, 'labels', split, image_filename.replace('.jpg', '.txt'))

        # 拷贝图像
        shutil.copy(image_path, out_img_path)

        # 转换标签
        convert_xml_to_yolo(xml_path, out_txt_path, w, h)

# 处理 train 和 val
process_split(train_xmls, 'train')
process_split(val_xmls, 'val')

print("------转换完成！YOLO 数据保存在:", output_dir)
