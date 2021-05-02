from os import listdir
from os.path import join
from scipy.io import loadmat
from skimage.io import imread, imsave
from cv2 import resize, CascadeClassifier, cvtColor, COLOR_BGR2GRAY
from logging import getLogger
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
        self.tot_class = 0
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
        self.tot_class = np.max(self.labels) + 1

    def __getitem__(self, index):
        return torch.Tensor(self.images[index]).float(), self.labels[index]

    def __len__(self):
        return self.len


def load_FRDataset(img_dirs=None, ratio=.2):
    r"""
    Face Recognition Data, University of Essex, UK
    每人基本都是20张图像, image shape: (200, 180, 3) or (196, 196, 3)

    Args:
        img_dirs (list[str] | None, optional): 文件路径列表，默认使用全部文件
        ratio (float): 测试集划分比例, 按类别划分，类别不相交, Default: 0.2
    Returns:
        (list, list): train_samples, test_samples, 形状为(class, n, (C, H, W)), (H, W)不一定相同
    """
    img_dirs = img_dirs or ['data/Face Recognition Data/faces94/female', 'data/Face Recognition Data/faces94/male',
                            'data/Face Recognition Data/faces94/malestaff', 'data/Face Recognition Data/faces95',
                            'data/Face Recognition Data/faces96', 'data/Face Recognition Data/grimace']
    train_samples = []
    test_samples = []

    for dir in img_dirs:
        samples = []

        for person_name in listdir(dir):
            one_person = []
            for i, file_name in enumerate(listdir(join(dir, person_name))):
                if file_name.split('.')[-1] != 'jpg' or \
                        person_name not in file_name.split('.')[0]:
                    continue

                img_path = join(dir, person_name, file_name)
                img = np.array(imread(img_path), dtype=np.float).transpose(2, 0, 1)
                one_person.append(img)

            if len(one_person) > 0:
                samples.append(one_person)

        test_class = int(len(samples) * ratio)
        train_samples += samples[test_class:]
        test_samples += samples[:test_class]

    tot_class = len(train_samples) + len(test_samples)
    logger = getLogger()
    logger.info(f'loading class: {tot_class}, {len(test_samples)} for testing')

    return train_samples, test_samples


def face_align(img, size=64):
    r"""
    利用python的库，检测人脸裁剪下来，resize到固定大小
    主要是针对第2个数据集

    Params:
        img (np.ndarray): 形状是(C, H, W)
        size (int): 定义最后返回的图像size
    Returns：
        np.ndarray: 返回对齐后的图像，形状是(C, size, size)
    """
    img = img.transpose(1, 2, 0)
    face_cascade = CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cvtColor(img.astype(np.uint8), COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) != 1:
        logger = getLogger()
        logger.warning(f'detect {len(faces)} faces!!!')
    if len(faces) != 1:
        out = img
    else:
        x, y, w, h = faces[0]
        out = img[y:y + h, x:x + w, :]
    out = resize(out, (size, size))
    out = out.transpose(2, 0, 1)
    return out


def preprocess(samples, size=64, align=False):
    r""" 数据预处理，主要是扣除人脸，并全部resize为(64, 64)

    Params:
        samples (list): 维度分别为(类别，数量，图像)， 图像是(C, H, W)的
        size (int): 统一的图像大小, default: 64
        align (bool): 是否使用对齐, default: False
    Returns:
        list: 预处理后的samples， 维度分别为(类别，数量，图像)， 图像是(C, 64, 64)的
    """
    new_samples = samples.copy()
    for i in range(len(samples)):
        for j in range(len(samples[i])):
            if align:
                new_samples[i][j] = face_align(samples[i][j], size=size)
            else:
                img = samples[i][j].transpose(1, 2, 0)
                new_samples[i][j] = resize(img, (size, size)).transpose(2, 0, 1)

    return new_samples


class FRDataset(data.Dataset):
    def __init__(self, samples):
        r""" Face Recognition Data, University of Essex, UK
            每人基本20张图像, image shape: (3, 64, 64)
            常规的(feature, labels) 的数据集
        """
        super().__init__()
        self.images = []
        self.labels = []
        self.tot_class = len(samples)
        for i in range(len(samples)):
            for j in range(len(samples[i])):
                img = samples[i][j]
                self.images.append(img)
                self.labels.append(i)
        self.len = len(self.labels)

    def __getitem__(self, index):
        return torch.Tensor(self.images[index]).float(), self.labels[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    train_samples, test_samples = load_FRDataset()
    dataset = FRDataset(train_samples)
    img, label = dataset.__getitem__(100)
    print(label)
    print(img.shape)
    imsave('test1.jpg', img.numpy().transpose(2, 0, 1))
    img, label = dataset.__getitem__(100)
    print(label)
    imsave('test2.jpg', img.numpy().transpose(2, 0, 1))
