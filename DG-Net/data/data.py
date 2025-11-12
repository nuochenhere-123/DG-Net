from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from data.LOLdataset import *
from data.eval_sets import *

def transform1(size=256):
    return Compose([
        RandomCrop((size, size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])

def transform2():
    return Compose([ToTensor()])


def get_UIE_training_set(data_dir,size):
    return UIEDatasetFromFolder(data_dir, transform=transform1(size))


def get_UIE_training_set_with_NR(data_dir,size):
    return UIEDatasetFromFolder_with_NR(data_dir, transform=transform1(size))


def get_UIE_training_set_withNoise(data_dir,size):
    return UIEDatasetFromFolder_withNoise(data_dir, transform=transform1(size))

def get_UIE_training_set_withNoise_NR(data_dir,size):
    return UIEDatasetFromFolder_withNoise_NR(data_dir, transform=transform1(size))


def get_eval_set(data_dir):
    return DatasetFromFolderEval(data_dir, transform=transform2())
