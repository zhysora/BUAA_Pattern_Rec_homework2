from os import listdir
from os.path import join
from scipy.io import loadmat
from skimage.io import imread
from cv2 import resize

import numpy as np
import torch
import torch.utils.data as data


class PIEDataset(data.Dataset):
    def __init__(self, img_dirs=None, test=False):
        r""" PIE dataset, 一共68类, image shape: (1, 64, 64), label: 0-67

        Args:
            img_dirs (list[str] | None, optional): 文件路径列表，默认使用全部文件
            test (bool): 区分训练集还是测试集, Default: False
        """
        super().__init__()

        img_dirs = img_dirs or ['data/PIE dataset/Pose05_64x64.mat', 'data/PIE dataset/Pose07_64x64.mat',
                                'data/PIE dataset/Pose09_64x64.mat', 'data/PIE dataset/Pose27_64x64.mat',
                                'data/PIE dataset/Pose29_64x64.mat']
        self.test = test
        self.images = []
        self.labels = []

        for path in img_dirs:
            data = loadmat(path)
            isTest = np.array(data['isTest'], dtype=np.int).squeeze(-1)
            fea = np.array(data['fea'], dtype=np.float).reshape(-1, 1, 64, 64)
            gnd = np.array(data['gnd'], dtype=np.int).squeeze(-1)

            for i, label in enumerate(isTest):
                if label != self.test:
                    continue
                self.images.append(fea[i])
                self.labels.append(gnd[i] - 1)

        self.images = np.array(self.images, dtype=np.float)
        self.labels = np.array(self.labels, dtype=np.int)
        self.len = len(self.labels)

    def __getitem__(self, index):
        return torch.Tensor(self.images[index]).float(), self.labels[index]

    def __len__(self):
        return self.len


def load_FRDataset(img_dirs=None, ratio=.1):
    r"""
    Face Recognition Data, University of Essex, UK
    每人20张图像, image shape: (200, 180, 3) or (196, 196, 3) => (3, 200, 200)

    Args:
        img_dirs (list[str] | None, optional): 文件路径列表，默认使用全部文件
        ratio (float): 测试集划分比例, Default: 0.1
    Returns:
        np.ndarray: train_x, train_y, test_x, test_y
        int: cnt 类别数
    """
    img_dirs = img_dirs or ['data/Face Recognition Data/faces94/female', 'data/Face Recognition Data/faces94/male',
                            'data/Face Recognition Data/faces94/malestaff', 'data/Face Recognition Data/faces95',
                            'data/Face Recognition Data/faces96', 'data/Face Recognition Data/grimace']

    cnt = 0
    train_x, train_y, test_x, test_y = [], [], [], []
    for dir in img_dirs:
        for person_name in listdir(dir):
            for i, file_name in enumerate(listdir(join(dir, person_name))):
                if file_name.split('.')[-1] != 'jpg' or \
                        person_name not in file_name.split('.')[0]:
                    continue

                img_path = join(dir, person_name, file_name)
                img = np.array(imread(img_path), dtype=np.float)
                img = resize(img, (200, 200)).transpose(2, 0, 1)
                person_id = cnt
                if i < int(20 * ratio):
                    test_x.append(img)
                    test_y.append(person_id)
                else:
                    train_x.append(img)
                    train_y.append(person_id)

            cnt += 1

    train_x = np.array(train_x, dtype=np.float)
    train_y = np.array(train_y, dtype=np.int)
    test_x = np.array(test_x, dtype=np.float)
    test_y = np.array(test_y, dtype=np.int)
    return train_x, train_y, test_x, test_y, cnt


class FRDataset(data.Dataset):
    def __init__(self, images, labels):
        r""" Face Recognition Data, University of Essex, UK
            每人20张图像, image shape: (3, 200, 200)

        """
        super().__init__()
        self.images = images
        self.labels = labels
        self.len = len(labels)

    def __getitem__(self, index):
        return torch.Tensor(self.images[index]).float(), self.labels[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    train_set = PIEDataset()
    x, y = train_set.__getitem__(100)
    print(train_set.__len__())
    print(x.shape, y)

    test_set = PIEDataset(test=True)
    x, y = test_set.__getitem__(100)
    print(test_set.__len__())
    print(x.shape, y)

    train_x, train_y, test_x, test_y, cnt = load_FRDataset()
    print(f'class: {cnt}')
    train_set = FRDataset(images=train_x, labels=train_y)
    x, y = train_set.__getitem__(100)
    print(train_set.__len__())
    print(x.shape, y)

    test_set = FRDataset(images=test_x, labels=test_y)
    x, y = test_set.__getitem__(100)
    print(test_set.__len__())
    print(x.shape, y)
