import torch
from torch.utils.data import Dataset
import os
import numpy as np
import json
import pandas as pd
from sklearn import preprocessing

torch.manual_seed(0)
np.random.seed(0)


class HRRR_Dataset(Dataset):

    def __init__(self, root_dir, json_file):

        self.select_cols = ['Avg Temperature (K)',	'Max Temperature (K)',	'Min Temperature (K)',
                            'Precipitation (kg m**-2)', 'Relative Humidity (%)', 'Wind Gust (m s**-1)',	'Wind Speed (m s**-1)',
                            'Downward Shortwave Radiation Flux (W m**-2)',	'Vapor Pressure Deficit (kPa)']

        # 1st day range: from 1st to 14th
        # 2nd day range: from 15th to 28th
        self.day_range = [[i for i in range(1, 15)], [j for j in range(15, 29)]]

        data = json.load(open(json_file))
        self.fips_codes = []
        self.years = []
        self.short_term_file_path = []
        self.long_term_file_path = []
        self.scaler = preprocessing.StandardScaler()

        for obj in data:
            self.fips_codes.append(obj["FIPS"])
            self.years.append(obj["year"])

            short_term = []
            for file_path in obj["data"]["HRRR"]["short_term"]:
                short_term.append(os.path.join(root_dir, file_path))

            long_term = []
            for file_paths in obj["data"]["HRRR"]["long_term"]:
                tmp_long_term = []
                for file_path in file_paths:
                    tmp_long_term.append(os.path.join(root_dir, file_path))
                long_term.append(tmp_long_term)

            self.short_term_file_path.append(short_term)
            self.long_term_file_path.append(long_term)

    def __len__(self):
        return len(self.fips_codes)

    def __getitem__(self, index):
        fips_code, year, = self.fips_codes[index], self.years[index]

        short_term_file_paths = self.short_term_file_path[index]
        x_short = self.get_short_term_val(fips_code, short_term_file_paths)

        long_term_file_paths = self.long_term_file_path[index]
        x_long = self.get_long_term_val(fips_code, long_term_file_paths)

        # convert type
        x_short = x_short.to(torch.float32)
        x_long = x_long.to(torch.float32)

        return x_short, x_long, fips_code, year

    def get_short_term_val(self, fips_code, file_paths):
        df_list = []
        for file_path in file_paths:
            tmp_df = pd.read_csv(file_path)
            df_list.append(tmp_df)

        df = pd.concat(df_list, ignore_index=True)

        # read FIPS code as string
        df["FIPS Code"] = df["FIPS Code"].astype(str)

        # filter the county and daily variables
        df = df[(df["FIPS Code"] == fips_code) & (df["Daily/Monthly"] == "Daily")]
        df.columns = df.columns.str.strip()

        group_month = df.groupby(['Month'])

        temporal_list = []
        for month, df_month in group_month:
            group_grid = df_month.groupby(['Grid Index'])

            time_series1, time_series2 = [], []
            for grid, df_grid in group_grid:
                df_grid = df_grid.sort_values(by=['Day'], ascending=[True], na_position='first')

                # 1st day range: from 1st to 14th
                df_grid1 = df_grid[df_grid.Day.isin(self.day_range[0])]
                df_grid1 = df_grid1[self.select_cols]
                val1 = torch.from_numpy(df_grid1.values)
                time_series1.append(val1)

                # 2nd day range: from 15th to 28th
                df_grid2 = df_grid[df_grid.Day.isin(self.day_range[1])]
                df_grid2 = df_grid2[self.select_cols]
                val2 = torch.from_numpy(df_grid2.values)
                time_series2.append(val2)

            temporal_list.append(torch.stack(time_series1))
            temporal_list.append(torch.stack(time_series2))

        x_short = torch.stack(temporal_list)
        return x_short

    def get_long_term_val(self, fips_code, temporal_file_paths):
        temporal_list = []

        for file_paths in temporal_file_paths:
            df_list = []
            for file_path in file_paths:
                tmp_df = pd.read_csv(file_path)
                df_list.append(tmp_df)

            df = pd.concat(df_list, ignore_index=True)

            # read FIPS code as string
            df["FIPS Code"] = df["FIPS Code"].astype(str)

            # filter the county and daily variables
            df = df[(df["FIPS Code"] == fips_code) & (df["Daily/Monthly"] == "Monthly")]

            df.columns = df.columns.str.strip()
            group_month = df.groupby(['Month'])

            month_list = []
            for month, df_month in group_month:
                df_month = df_month[self.select_cols]
                val = torch.from_numpy(df_month.values)
                val = torch.flatten(val, start_dim=0)
                month_list.append(val)

            temporal_list.append(torch.stack(month_list))

        x_long = torch.stack(temporal_list)
        return x_long


if __name__ == '__main__':
    root_dir = "/mnt/data/Crop"
    train = "./../data/soybean_train.json"
    dataset = HRRR_Dataset(root_dir, train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for xs, xl, f, y in train_loader:
        print(xs.shape)
        print(xl.shape)

