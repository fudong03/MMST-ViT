import concurrent
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

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

    def __init__(self, root_dir, json_file, num_workers=4):

        self.select_cols = ['Avg Temperature (K)', 'Max Temperature (K)', 'Min Temperature (K)',
                            'Precipitation (kg m**-2)', 'Relative Humidity (%)', 'Wind Gust (m s**-1)',
                            'Wind Speed (m s**-1)',
                            'Downward Shortwave Radiation Flux (W m**-2)', 'Vapor Pressure Deficit (kPa)']

        # consider the first 28 days in each month
        self.day_range = [i for i in range(1, 29)]

        data = json.load(open(json_file))
        self.fips_codes = []
        self.years = []
        self.short_term_file_path = []
        self.long_term_file_path = []

        self.num_workers = num_workers

        self.executor = ThreadPoolExecutor(max_workers=num_workers)

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
            tmp_df = self.read_csv_file(file_path)
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

            time_series = []
            for grid, df_grid in group_grid:
                df_grid = df_grid.sort_values(by=['Day'], ascending=[True], na_position='first')

                df_grid = df_grid[df_grid.Day.isin(self.day_range)]
                df_grid = df_grid[self.select_cols]

                val = torch.from_numpy(df_grid.values)
                time_series.append(val)

            temporal_list.append(torch.stack(time_series))

        x_short = torch.stack(temporal_list)

        return x_short

    def get_long_term_val(self, fips_code, temporal_file_paths):
        temporal_list = []

        for file_paths in temporal_file_paths:

            # Submit read_csv_file function for each file path
            futures = [self.executor.submit(self.read_csv_file, file_path) for file_path in file_paths]

            # Wait for all tasks (reading files) to complete
            concurrent.futures.wait(futures)

            # Get the results (DataFrames) from the completed tasks
            dfs = [future.result() for future in futures]

            df = pd.concat(dfs, ignore_index=True)

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

    @lru_cache(maxsize=128)
    def read_csv_file(self, file_path):
        return pd.read_csv(file_path)


if __name__ == '__main__':
    root_dir = "/mnt/data/Tiny CropNet"
    train = "./../data/soybean_train.json"
    # train = "./../data/soybean_val.json"
    dataset = HRRR_Dataset(root_dir, train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    # Record start time
    start_time = time.time()
    for xs, xl, f, y in train_loader:
        print("fips: {}, year: {}, short shape: {}".format(f, y, xs.shape))
        print("fips: {}, year: {}, long shape: {}".format(f, y, xl.shape))

        # Record end time
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time = end_time - start_time
        print(f"Time Elapsed: {elapsed_time:.6f} seconds")

        start_time = time.time()
