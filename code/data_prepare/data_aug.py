import glob
import shutil
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import os
import random
import numpy as np
import cv2
from utils import random_vertical_flip, random_horizontal_flip, random_rot90, show_anns, get_random_data
from tqdm import tqdm

if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if not os.path.exists(root+'/user_data/aug_dataset'):
        os.mkdir(root+'/user_data/aug_dataset')
    shutil.rmtree(root+'/user_data/aug_dataset/images')
    shutil.rmtree(root+'/user_data/aug_dataset/labels')
    shutil.copytree(root+'/xfdata/遥感图像舰船小目标检测挑战赛训练集-初赛/images', root+'/user_data/aug_dataset/images')
    shutil.copytree(root+'/xfdata/遥感图像舰船小目标检测挑战赛训练集-初赛/labels', root+'/user_data/aug_dataset/labels')
   
    image_dir = root+"/user_data/aug_dataset/images"
    label_dir = root+"/user_data/aug_dataset/labels"
    img_list = glob.glob(image_dir + "/*.jpg")
    ori_img_num = len(img_list)
    # flip and rotate
     #--------------------------------------------------翻转或旋转图像-----------------------------------------------------
    flip_rotate_aug = tqdm(img_list)
    for path in flip_rotate_aug:
        flip_rotate_aug.set_description('Flip and rotate data augmentation:')
        img_org = cv2.imread(path)
        img = img_org

        bboxes = []
        with open(label_dir + '/' + path.split("/")[-1][:-4] + '.txt', 'r') as file:
            label = file.read()
            label = label.split('\n')
            for i in range(len(label) - 1):
                bbox = []
                x1 = label[i].split(' ')[1]
                y1 = label[i].split(' ')[2]
                x2 = label[i].split(' ')[3]
                y2 = label[i].split(' ')[4]
                bbox.append(int(x1))
                bbox.append(int(y1))
                bbox.append(int(x2))
                bbox.append(int(y2))
                bbox.append(0)
                bboxes.append(bbox)

        aug = random.choice([1, 2, 3])
        if aug == 1:
            img, bboxes = random_horizontal_flip(img, np.array(bboxes), 1)
        elif aug == 2:
            img, bboxes = random_vertical_flip(img, np.array(bboxes), 1)
        else:
            img, bboxes = random_rot90(img, np.array(bboxes), 1)
        with open(label_dir + "/flip_" + path.split('/')[-1][:-4] + '.txt', 'w') as file:
            for box in bboxes:
                cls = 'ship'
                file.write(
                    f"{cls} {box[0]} {box[1]} {box[2]} {box[3]}\n")
        cv2.imwrite(image_dir + '/flip_' + path.split('/')[-1][:-4] + '.jpg', img)
    print("Flip or rotate finished,start increase noise.... ")
    # auto mask
    sam_checkpoint = root+"/user_data/model_data/sam_vit_l_0b3195.pth"
    model_type = "vit_l"
    device = "cuda"
#--------------------------------------------------使用语义分割增加噪声-----------------------------------------------------
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    mask_aug = tqdm(img_list)
    for path in mask_aug:
        mask_aug.set_description("Auto mask data augmentation:")
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')
        plt.savefig(image_dir + "/mask_" + path.split("/")[-1], bbox_inches='tight', pad_inches=0)
        plt.close()
        shutil.copy(os.path.join(label_dir, path.split("/")[-1].replace('jpg', 'txt')), os.path.join(label_dir, "mask_" + path.split("/")[-1].replace('jpg', 'txt')))
        image = Image.open(path)
        width, height = image.size
        image = Image.open(image_dir + "/mask_" + path.split("/")[-1])
        resized_image = image.resize((width, height))
        resized_image.save(image_dir + "/mask_" + path.split("/")[-1])
    print("Increase noise finished,start mosaic augment....")
    # mosaic
    #--------------------------------------------------mosaic数据增强-----------------------------------------------------
    n = 0
    # 读取4张图像及其检测框信息
    image_list = []
    image_names = os.listdir(image_dir)
    ori_img_names, flip_img_names, mask_img_names = [], [], []
    for i in range(len(image_names)):
        if 'flip' in image_names[i]:
            flip_img_names.append(image_names[i])
        elif 'mask' in image_names[i]:
            mask_img_names.append(image_names[i])
        else:
            ori_img_names.append(image_names[i])

    ori_img_mosaic = tqdm(ori_img_names)
    flip_img_mosaic = tqdm(flip_img_names)
    mask_img_mosaic = tqdm(mask_img_names)

    for i, image_name in enumerate(ori_img_mosaic):
        ori_img_mosaic.set_description("Original images mosaic data augmentation:")
        if i % 4 == 0 and i != 0:
            n += 1
            # 缩放、拼接图片
            get_random_data(image_list, input_shape=[640, 640], img_save_path=image_dir, label_save_path=label_dir, num=n)
            image_list = []  # 存放每张图像和该图像对应的检测框坐标信息

        image_box = []  # 存放每张图片的检测框信息

        # 某张图片位置及其标签文件位置
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name.replace('jpg', 'txt'))
        image = cv2.imread(image_path)  # 读取图像

        # 获取bbox信息
        with open(label_path, 'r') as file:
            label = file.read()
            label = label.split('\n')
            for j in range(len(label) - 1):
                bbox = []
                x1 = label[j].split(' ')[1]
                y1 = label[j].split(' ')[2]
                x2 = label[j].split(' ')[3]
                y2 = label[j].split(' ')[4]
                bbox.append(int(x1))
                bbox.append(int(y1))
                bbox.append(int(x2))
                bbox.append(int(y2))
                image_box.append(bbox)

        # 保存图像及其对应的检测框信息
        image_list.append([image, image_box])

    image_list = []
    for i, image_name in enumerate(flip_img_mosaic):
        flip_img_mosaic.set_description("Flip and rotate images mosaic data augmentation:")
        if i % 4 == 0 and i != 0:
            n += 1
            # 缩放、拼接图片
            get_random_data(image_list, input_shape=[640, 640], img_save_path=image_dir, label_save_path=label_dir, num=n)
            image_list = []  # 存放每张图像和该图像对应的检测框坐标信息

        image_box = []  # 存放每张图片的检测框信息

        # 某张图片位置及其标签文件位置
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name.replace('jpg', 'txt'))
        image = cv2.imread(image_path)  # 读取图像

        # 获取bbox信息
        with open(label_path, 'r') as file:
            label = file.read()
            label = label.split('\n')
            for j in range(len(label) - 1):
                bbox = []
                x1 = label[j].split(' ')[1]
                y1 = label[j].split(' ')[2]
                x2 = label[j].split(' ')[3]
                y2 = label[j].split(' ')[4]
                bbox.append(int(x1))
                bbox.append(int(y1))
                bbox.append(int(x2))
                bbox.append(int(y2))
                image_box.append(bbox)

        # 保存图像及其对应的检测框信息
        image_list.append([image, image_box])

    image_list = []
    for i, image_name in enumerate(mask_img_mosaic):
        mask_img_mosaic.set_description("Masked images mosaic data augmentation:")
        if i % 4 == 0 and i != 0:
            n += 1
            # 缩放、拼接图片
            get_random_data(image_list, input_shape=[640, 640], img_save_path=image_dir, label_save_path=label_dir, num=n)
            image_list = []  # 存放每张图像和该图像对应的检测框坐标信息

        image_box = []  # 存放每张图片的检测框信息

        # 某张图片位置及其标签文件位置
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name.replace('jpg', 'txt'))
        image = cv2.imread(image_path)  # 读取图像

        # 获取bbox信息
        with open(label_path, 'r') as file:
            label = file.read()
            label = label.split('\n')
            for j in range(len(label) - 1):
                bbox = []
                x1 = label[j].split(' ')[1]
                y1 = label[j].split(' ')[2]
                x2 = label[j].split(' ')[3]
                y2 = label[j].split(' ')[4]
                bbox.append(int(x1))
                bbox.append(int(y1))
                bbox.append(int(x2))
                bbox.append(int(y2))
                image_box.append(bbox)

        # 保存图像及其对应的检测框信息
        image_list.append([image, image_box])
    