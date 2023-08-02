import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from einops import rearrange
from PIL import Image
import torchvision.transforms as transforms

from dataset.sentinel_loader import Sentinel_Dataset


class Sentinel_Subset(Dataset):

    def __init__(self, X, Y, transform):
        self.b, self.t, self.g, _, _, _ = X.shape
        self.X = rearrange(X, 'b t g h w c -> (b t g) h w c')
        self.Y = rearrange(Y, 'b t g n d -> (b t g) n d')

        B = self.b * self.t * self.g
        self.indices = [i for i in range(B)]
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x = self.X[index, :, :, :]
        y = self.Y[index, :, :]

        img = Image.fromarray(np.uint8(x))
        xi, xj = self.transform(img)

        return xi, xj, y


class SimCLRDataTransform(object):
    def __init__(self, img_size=224, s=1, kernel_size=9):
        self.img_size = img_size
        self.s = s
        self.kernel_size = kernel_size
        self.transform = self.get_simclr_pipeline_transform()

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj

    def get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.img_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.GaussianBlur(kernel_size=self.kernel_size),
                                              transforms.ToTensor(),
                                              # transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192]),
                                              ])
        return data_transforms


def get_data_loader(X, Y, batch_size=128, num_workers=8):
    transform = SimCLRDataTransform()
    dataset = Sentinel_Subset(X, Y, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)
    # data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)

    return data_loader


if __name__ == '__main__':
    root_dir = "/mnt/data/Tiny CropNet"
    train = "./../data/soybean_train.json"
    dataset = Sentinel_Dataset(root_dir, train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    it = iter(train_loader)
    x, f, y = next(it)

    loader = get_data_loader(x)
    for xi, xj in loader:
        print(xi.shape)
        print(xj.shape)
