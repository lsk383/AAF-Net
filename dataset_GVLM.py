
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
