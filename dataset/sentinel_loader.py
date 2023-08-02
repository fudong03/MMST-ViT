import h5py
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import json

torch.manual_seed(0)
np.random.seed(0)


class Sentinel_Dataset(Dataset):

    def __init__(self, root_dir, json_file):
        self.fips_codes = []
        self.years = []
        self.file_paths = []

        data = json.load(open(json_file))
        for obj in data:
            self.fips_codes.append(obj["FIPS"])
            self.years.append(obj["year"])

            tmp_path = []
            relative_path_list = obj["data"]["sentinel"]
            for relative_path in relative_path_list:
                tmp_path.append(os.path.join(root_dir, relative_path))
            self.file_paths.append(tmp_path)

    def __len__(self):
        return len(self.fips_codes)

    def __getitem__(self, index):
        fips_code, year = self.fips_codes[index], self.years[index]
        file_paths = self.file_paths[index]

        temporal_list = []

        for file_path in file_paths:
            with h5py.File(file_path, 'r') as hf:
                groups = hf[fips_code]
                for i, d in enumerate(groups.keys()):
                    # only consider the 1st day of each month
                    # note that the h5 file contains the 1st and 15th of images for each month, e.g., "04-01" and "04-15"
                    if i % 2 == 0:
                        grids = groups[d]["data"]
                        grids = np.asarray(grids)
                        temporal_list.append(torch.from_numpy(grids))
                hf.close()

        x = torch.stack(temporal_list)

        return x, fips_code, year


if __name__ == '__main__':
    root_dir = "/mnt/data/Tiny CropNet"
    # train = "./../data/soybean_train.json"
    train = "./../data/soybean_val.json"
    dataset = Sentinel_Dataset(root_dir, train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    max_g = 0
    for x, f, y in train_loader:
        print("fips: {}, year: {}, shape: {}".format(f, y, x.shape))
        max_g = max(max_g, tuple(x.shape)[2])

    print(max_g)
