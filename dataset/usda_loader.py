import torch
from torch.utils.data import Dataset
import os
import numpy as np
import json
import pandas as pd

torch.manual_seed(0)
np.random.seed(0)


class USDA_Dataset(Dataset):

    def __init__(self, root_dir, json_file, crop_type="Soybeans"):

        self.crop_type = crop_type

        if crop_type == "Cotton":
            select_cols = ['PRODUCTION, MEASURED IN 480 LB BALES', 'YIELD, MEASURED IN BU / ACRE']
        else:
            select_cols = ['PRODUCTION, MEASURED IN BU', 'YIELD, MEASURED IN BU / ACRE']

        self.select_cols = select_cols

        self.fips_codes = []
        self.years = []
        self.state_ansi = []
        self.county_ansi = []
        self.file_paths = []

        data = json.load(open(json_file))
        for obj in data:
            self.fips_codes.append(obj["FIPS"])
            self.years.append(obj["year"])
            self.state_ansi.append(obj["state_ansi"])
            self.county_ansi.append(obj["county_ansi"])

            relative_path = obj["data"]["USDA"]
            self.file_paths.append(os.path.join(root_dir, relative_path))

    def __len__(self):
        return len(self.fips_codes)

    def __getitem__(self, index):
        fips_code, year = self.fips_codes[index], self.years[index]
        state_ansi, county_ansi = self.state_ansi[index], self.county_ansi[index]

        file_path = self.file_paths[index]
        df = pd.read_csv(file_path)

        # convert state_ansi and county_ansi to string with leading zeros
        df['state_ansi'] = df['state_ansi'].astype(str).str.zfill(2)
        df['county_ansi'] = df['county_ansi'].astype(str).str.zfill(3)

        df = df[(df["state_ansi"] == state_ansi) & (df["county_ansi"] == county_ansi)]

        df = df[self.select_cols]

        x = torch.from_numpy(df.values)
        x = x.to(torch.float32)
        x = torch.log(torch.flatten(x, start_dim=0))

        return x, fips_code, year


if __name__ == '__main__':
    root_dir = "/mnt/data/Tiny CropNet"
    # train = "./../data/soybean_train.json"
    train = "./../data/soybean_val.json"

    dataset = USDA_Dataset(root_dir, train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    for x, f, y in train_loader:
        print("fips: {}, year: {}, shape: {}".format(f, y, x.shape))
