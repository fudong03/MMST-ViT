import torch
from sklearn import preprocessing
from torch.utils.data import Dataset
from einops import rearrange
import torchvision.transforms as transforms

from dataset.sentinel_loader import Sentinel_Dataset


class DataWrapper(object):
    def __init__(self, img_size=224, s=1, kernel_size=9, train=True):
        self.img_size = img_size
        self.s = s
        self.kernel_size = kernel_size

        if train:
            self.transform = self.get_simclr_pipeline_transform()
        else:
            self.transform = self.get_transform_val()

    def __call__(self, x):
        x = x.to(torch.float32)
        xi = self.transform(x)
        xj = self.transform(x)
        return xi, xj

    def get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.img_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.GaussianBlur(kernel_size=self.kernel_size),
                                              transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192]),
                                              ])
        return data_transforms

    def get_transform_val(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        data_transforms = transforms.Compose([transforms.CenterCrop(size=self.img_size),
                                              transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192]),
                                              ])
        return data_transforms


class ScalarNorm(object):
    def __init__(self):
        self.norm = preprocessing.StandardScaler()

    def __call__(self, x, reverse=False):
        if not reverse:
            x = self.norm.fit_transform(x)
        else:
            x = self.norm.inverse_transform(x)

        x = torch.from_numpy(x)
        x = x.to(torch.float32)
        return x


if __name__ == '__main__':
    root_dir = "/mnt/data/Crop"
    train = "./../data/soybean_train.json"
    dataset = Sentinel_Dataset(root_dir, train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    it = iter(train_loader)
    x, f, y = next(it)

    x = x[:, :, :22, :, :, :]

    x = rearrange(x, 'b t g h w c -> (b t g) c h w')

    wrapper = DataWrapper()

    xi, xj = wrapper(x)
    print(xi.shape)
    print(xj.shape)
