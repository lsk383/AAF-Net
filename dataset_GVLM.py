# import cv2
# import numpy
# import torch.utils.data
# import os
#
#
# class Dataset(torch.utils.data.Dataset):
#     '''
#     Class to load the dataset
#     '''
#
#     def __init__(self, dataset, file_root='/home/LWGANet/GVLM-CD-Processed', transform=None, dataset_name='GVLM'):
#         """
#         dataset: 'train', 'val', or 'test'
#         file_root: root of the processed data directory
#         dataset_name: Name of the dataset, e.g., 'GVLM', 'WHUCD256'
#         """
#         self.transform = transform
#
#         # --- 新增: 初始化域标签映射字典 ---
#         # 确保无论何种数据集，该属性都存在
#         self.domain_map = {}
#
#         # 根据数据集名称确定文件路径结构
#         if dataset_name in ['WHUCD256', 'CDD']:
#             list_file_path = os.path.join(file_root, 'list', dataset + '.txt')
#             self.file_list = open(list_file_path).read().splitlines()
#             # 假设 WHU 和 CDD 的数据也按 train/val/test 文件夹组织
#             self.pre_images = [os.path.join(file_root, dataset, 'A', x) for x in self.file_list]
#             self.post_images = [os.path.join(file_root, dataset, 'B', x) for x in self.file_list]
#             self.gts = [os.path.join(file_root, dataset, 'label', x) for x in self.file_list]
#
#         # <<< 核心修改: 专门为GVLM数据集添加处理逻辑 >>>
#         elif dataset_name == 'GVLM':
#             # 1. 构建列表文件的路径 (e.g., /home/LWGANet/GVLM-CD-Processed/list/train.txt)
#             list_file_path = os.path.join(file_root, 'list', dataset + '.txt')
#             with open(list_file_path, 'r') as f:
#                 # 读取不带后缀的文件名
#                 self.file_list_no_ext = [line.strip() for line in f.readlines()]
#
#             # 2. 构建完整的图像和标签路径列表
#             self.pre_images = [os.path.join(file_root, dataset, 'A', x + '.png') for x in self.file_list_no_ext]
#             self.post_images = [os.path.join(file_root, dataset, 'B', x + '.png') for x in self.file_list_no_ext]
#             self.gts = [os.path.join(file_root, dataset, 'label', x + '.png') for x in self.file_list_no_ext]
#
#             # 3. 加载域标签映射文件
#             domain_file_path = os.path.join(file_root, 'list', 'domain_labels.txt')
#             if os.path.exists(domain_file_path):
#                 with open(domain_file_path, 'r') as f:
#                     for line in f:
#                         parts = line.strip().split()
#                         if len(parts) == 2:
#                             # key: 文件名 (e.g., '0.png'), value: 域标签 (int)
#                             self.domain_map[parts[0]] = int(parts[1])
#                 print(f"成功从 {domain_file_path} 加载域标签映射。")
#             else:
#                 print(f"警告: 未找到域标签文件 -> {domain_file_path}。所有域标签将默认为0。")
#
#         else:  # 处理 LEVIR, SYSU 等原有逻辑
#             self.file_list = open(os.path.join(file_root, dataset, 'list', dataset + '.txt')).read().splitlines()
#             self.pre_images = [os.path.join(file_root, dataset, 'A', x) for x in self.file_list]
#             self.post_images = [os.path.join(file_root, dataset, 'B', x) for x in self.file_list]
#             self.gts = [os.path.join(file_root, dataset, 'label', x) for x in self.file_list]
#
#     def __len__(self):
#         # 使用 pre_images 列表的长度，因为它对所有情况都通用
#         return len(self.pre_images)
#
#     def __getitem__(self, idx):
#         pre_image_name = self.pre_images[idx]
#         label_name = self.gts[idx]
#         post_image_name = self.post_images[idx]
#
#         pre_image = cv2.imread(pre_image_name)
#         label = cv2.imread(label_name, 0)
#         post_image = cv2.imread(post_image_name)
#
#         img = numpy.concatenate((pre_image, post_image), axis=2)
#
#         # --- 新增: 获取域标签 ---
#         # 从完整路径中提取基本文件名 (e.g., '0.png')
#         base_filename = os.path.basename(pre_image_name)
#         # 从字典中获取域标签，如果找不到则默认为0
#         domain_label = self.domain_map.get(base_filename, 0)
#
#         if self.transform:
#             # 假设您的 transform 函数只接受 img 和 label
#             # 如果它也需要修改，请告诉我
#             [img, label] = self.transform(img, label)
#
#         # <<< 核心修改: 返回值中增加 domain_label >>>
#         return img, label, domain_label
#
#     def get_img_info(self, idx):
#         img = cv2.imread(self.pre_images[idx])
#         return {"height": img.shape[0], "width": img.shape[1]}


# import cv2
# import numpy
# import torch.utils.data
# import os
#
#
# class Dataset(torch.utils.data.Dataset):
#     '''
#     Class to load the dataset with composite domain labels for t1 and t2.
#     '''
#
#     def __init__(self, dataset, file_root='/home/LWGANet/GVLM-CD-Processed', transform=None, dataset_name='GVLM'):
#         """
#         dataset: 'train', 'val', or 'test'
#         file_root: root of the processed data directory
#         dataset_name: Name of the dataset, e.g., 'GVLM'
#         """
#         self.transform = transform
#         self.domain_map = {}
#
#         # This implementation is now focused on the GVLM dataset structure
#         if dataset_name == 'GVLM':
#             # 1. Build path to the list file (e.g., .../list/train.txt)
#             list_file_path = os.path.join(file_root, 'list', dataset + '.txt')
#             with open(list_file_path, 'r') as f:
#                 self.file_list_no_ext = [line.strip() for line in f.readlines()]
#
#             # 2. Build full paths for images and labels
#             self.pre_images = [os.path.join(file_root, dataset, 'A', x + '.png') for x in self.file_list_no_ext]
#             self.post_images = [os.path.join(file_root, dataset, 'B', x + '.png') for x in self.file_list_no_ext]
#             self.gts = [os.path.join(file_root, dataset, 'label', x + '.png') for x in self.file_list_no_ext]
#
#             # 3. Load the city-level domain mapping
#             domain_file_path = os.path.join(file_root, 'list', 'domain_labels.txt')
#             if os.path.exists(domain_file_path):
#                 with open(domain_file_path, 'r') as f:
#                     for line in f:
#                         parts = line.strip().split()
#                         if len(parts) == 2:
#                             self.domain_map[parts[0]] = int(parts[1])
#                 print(f"成功从 {domain_file_path} 加载城市域标签映射。")
#             else:
#                 print(f"警告: 未找到域标签文件 -> {domain_file_path}。所有城市域标签将默认为0。")
#
#         else:
#             # Add logic for other datasets if needed, otherwise raise an error
#             raise NotImplementedError(f"Dataset loader for '{dataset_name}' is not implemented in this version.")
#
#     def __len__(self):
#         return len(self.pre_images)
#
#     def __getitem__(self, idx):
#         pre_image_name = self.pre_images[idx]
#         label_name = self.gts[idx]
#         post_image_name = self.post_images[idx]
#
#         pre_image = cv2.imread(pre_image_name)
#         label = cv2.imread(label_name, 0)
#         post_image = cv2.imread(post_image_name)
#
#         img = numpy.concatenate((pre_image, post_image), axis=2)
#
#         # --- 核心修改: 创建复合域标签 ---
#         base_filename = os.path.basename(pre_image_name)
#         # 1. 获取基础的城市域标签
#         city_domain_label = self.domain_map.get(base_filename, 0)
#
#         # 2. 创建 t1 和 t2 的独立域标签
#         domain_label_t1 = city_domain_label * 2
#         domain_label_t2 = city_domain_label * 2 + 1
#
#         if self.transform:
#             # Assuming your transform function takes img and label
#             [img, label] = self.transform(img, label)
#
#         # <<< 核心修改: 返回两个独立的域标签 >>>
#         return img, label, domain_label_t1, domain_label_t2
#
#     def get_img_info(self, idx):
#         img = cv2.imread(self.pre_images[idx])
#         return {"height": img.shape[0], "width": img.shape[1]}

import cv2
import numpy
import torch.utils.data
import os

class Dataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''

    def __init__(self, dataset, file_root='/home/A2Net/LEVIR-CD_256_patches', transform=None, dataset_name='WHU-CD'):
        """
        dataset: dataset name, e.g. train, val, or test
        file_root: root of data_path
        """
        self.split = dataset
        self.file_root = file_root
        self.transform = transform
        self.dataset_name = dataset_name

        # --- MODIFICATION 1: Standardize list path construction ---
        # Determine the path to the list file (e.g., train.txt)
        if dataset_name in ['WHUCD256', 'CDD']:
            list_path = os.path.join(file_root, 'list', dataset + '.txt')
        else: # Handles LEVIR, SYSU, and GVLM
            list_path = os.path.join(file_root, 'list', dataset + '.txt')
            # A small correction for LEVIR/SYSU structure if they are different
            if not os.path.exists(list_path):
                 list_path = os.path.join(file_root, dataset, 'list', dataset + '.txt')


        # Read the list of image names (without extension)
        with open(list_path, 'r') as f:
            self.file_list = f.read().splitlines()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_name = self.file_list[idx]

        # --- MODIFICATION 2: Correctly build the full image path with extension ---
        # Assumes the image format is .png. Change if it's .jpg, .tif, etc.
        image_extension = '.png'

        # Construct paths based on the dataset structure
        if self.dataset_name in ['WHUCD256', 'CDD', 'GVLM-CD-Processed']: # GVLM follows this structure
             pre_image_name = os.path.join(self.file_root, self.split, 'A', image_name + image_extension)
             post_image_name = os.path.join(self.file_root, self.split, 'B', image_name + image_extension)
             label_name = os.path.join(self.file_root, self.split, 'label', image_name + image_extension)
        else: # LEVIR, SYSU
             pre_image_name = os.path.join(self.file_root, self.split, 'A', image_name + image_extension)
             post_image_name = os.path.join(self.file_root, self.split, 'B', image_name + image_extension)
             label_name = os.path.join(self.file_root, self.split, 'label', image_name + image_extension)


        # Read images
        pre_image = cv2.imread(pre_image_name)
        post_image = cv2.imread(post_image_name)
        label = cv2.imread(label_name, 0) # Read label as grayscale

        # --- MODIFICATION 3: Add a robustness check ---
        if pre_image is None:
            raise FileNotFoundError(f"Could not read pre-change image: {pre_image_name}")
        if post_image is None:
            raise FileNotFoundError(f"Could not read post-change image: {post_image_name}")
        if label is None:
            raise FileNotFoundError(f"Could not read label image: {label_name}")

        # Concatenate images
        img = numpy.concatenate((pre_image, post_image), axis=2)

        # Apply transformations
        if self.transform:
            img, label = self.transform(img, label)

        return img, label

    def get_img_info(self, idx):
        # Also fix path construction here
        image_name = self.file_list[idx]
        image_extension = '.png'
        if self.dataset_name in ['WHUCD256', 'CDD', 'GVLM-CD-Processed']:
            pre_image_name = os.path.join(self.file_root, self.split, 'A', image_name + image_extension)
        else:
            pre_image_name = os.path.join(self.file_root, self.split, 'A', image_name + image_extension)

        img = cv2.imread(pre_image_name)
        if img is None:
            raise FileNotFoundError(f"Could not read image for info: {pre_image_name}")
        return {"height": img.shape[0], "width": img.shape[1]}
