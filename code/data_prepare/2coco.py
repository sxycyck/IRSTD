# @skyclear
# -*-coding:utf-8 -*-
import json
import os
import shutil

import cv2

# info ，license，categories 结构初始化；
# 在train.json,val.json,test.json里面信息是一致的；

# info，license暂时用不到
info = {
    "year": 2022,
    "version": '1.0',
    "date_created": 2022 - 10 - 15
}

licenses = {
    "id": 1,
    "name": "null",
    "url": "null",
}

# 自己的标签类别，跟yolov5的要对应好；
categories = [
    {
        "id": 1,
        "name": 'ship',
        "supercategory": 'ship',
    }
]

# 初始化train,test数据字典
# info licenses categories 在train和test里面都是一致的；
train_data = {'info': info, 'licenses': licenses, 'categories': categories, 'images': [], 'annotations': []}
val_data = {'info': info, 'licenses': licenses, 'categories': categories, 'images': [], 'annotations': []}


# image_path 对应yolov5的图像路径，比如images/train；
# label_path 对应yolov5的label路径，比如labels/train 跟images要对应；
def v5_covert_coco_format(image_path, label_path,coco_format_path,key):
    images = []
    annotations = []
    dirs = os.listdir(image_path)
    if key == 'train':
        dirs = dirs
    else:
        dirs = dirs[-100:]
    
    for index, img_file in enumerate(dirs):
        if img_file.endswith('.jpg'):
            image_info = {}
            img = cv2.imread(os.path.join(image_path, img_file))
            shutil.copy(os.path.join(image_path,img_file),os.path.join(coco_format_path+f'/{key}2017',img_file))
            height, width, channel = img.shape
            image_info['id'] = index
            image_info['file_name'] = img_file
            image_info['width'], image_info['height'] = width, height
        else:
            continue
        if image_info != {}:
            images.append(image_info)
        # 处理label信息-------
        label_file = os.path.join(label_path, img_file.replace('.jpg', '.txt'))
        with open(label_file, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                info_annotation = {}

                class_id = 1
                xmin,ymin,xmax,ymax = line.strip().split(' ')[1:]
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                bbox_w = xmax - xmin
                bbox_h = ymax - ymin
                img_copy = img[int(ymin):int(ymax), int(xmin):int(xmax)].copy()

                info_annotation["category_id"] = class_id  # 类别的id
                info_annotation['bbox'] = [xmin, ymin, bbox_w, bbox_h]  ## bbox的坐标
                info_annotation['area'] = bbox_h * bbox_w  ###area
                info_annotation['image_id'] = index  # bbox的id
                info_annotation['id'] = index * 100 + idx  # bbox的id
                # cv2.imwrite(f"./temp/{info_annotation['id']}.jpg", img_copy)
                info_annotation['segmentation'] = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]  # 四个点的坐标
                info_annotation['iscrowd'] = 0  # 单例
                annotations.append(info_annotation)
    return images, annotations


# key == train，test，val
# 对应要生成的json文件，比如instances_train2017.json，instances_test2017.json，instances_val2017.json
# 只是为了不重复写代码。。。。。
def gen_json_file(yolov5_data_path, coco_format_path, key):
    # json path
    json_path = os.path.join(coco_format_path, f'annotations/instances_{key}2017.json')
    dst_path = os.path.join(coco_format_path, f'{key}2017')
    if not os.path.exists(os.path.dirname(json_path)):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    data_path = os.path.join(yolov5_data_path, f'images')
    label_path = os.path.join(yolov5_data_path, f'labels')
    images, anns = v5_covert_coco_format(data_path, label_path,coco_format_path,key)
    if key == 'train':
        train_data['images'] = images
        train_data['annotations'] = anns
        with open(json_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        # shutil.copy(data_path,'')
    elif key == 'val':
        val_data['images'] = images
        val_data['annotations'] = anns
        with open(json_path, 'w') as f:
            json.dump(val_data, f, indent=2)
    else:
        print(f'key is {key}')
    print(f'generate {key} json success!')
    return


if __name__ == '__main__':
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    yolov5_data_path = root+'/user_data/aug_dataset'
    coco_format_path = root+'/user_data/coco_dataset'
    gen_json_file(yolov5_data_path, coco_format_path, key='train')
    gen_json_file(yolov5_data_path, coco_format_path, key='val')