#!/usr/bin/env python

import os
import argparse
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from modules.constants import T


def is_valid_segment(segment, threshold=T//2):
    if len(segment) < threshold:
        return False
    return True

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-l', '--location')
    arg_parser.add_argument('-i', '--in-dir')
    arg_parser.add_argument('-o', '--out-dir')
    arg_parser.add_argument('-d', '--divide', action='store_true')
    args = arg_parser.parse_args()

    in_loc = os.path.join(args.in_dir, args.location)
    out_loc = os.path.join(args.out_dir, args.location)

    os.makedirs(out_loc, exist_ok=True)

    metadata_df = pd.read_csv(os.path.join('metadata', f'{args.location}.csv'), dtype={'test_id': str}).set_index('test_id')

    print(f'Filtering {args.location} data')

    if args.location == 'helsinki':
        for test_id in tqdm(os.listdir(in_loc)):
            row = metadata_df.loc[test_id]
            for i, f in enumerate(sorted(glob(os.path.join(in_loc, test_id, '*.csv'))), start=1):
                
                interval = row[f'segment_{i}']

                if interval == 'bad':
                    continue
                elif interval == 'good':
                    df = pd.read_csv(f).fillna(method='ffill').dropna()
                else:
                    df = pd.read_csv(f, parse_dates=['timestamp'], index_col='timestamp').fillna(method='ffill').dropna()
                    start, end = interval.split('-')
                    df = df.between_time(start, end)

                if args.divide:
                    for j, chunk in enumerate(np.array_split(df, len(df)//T + 1)):
                        if is_valid_segment(chunk):
                            chunk.to_csv(os.path.join(out_loc, f'{test_id}_{i}_{j}.csv'), index=False)
                else:
                    if is_valid_segment(df):
                        df.to_csv(os.path.join(out_loc, f'{test_id}_{i}.csv'), index=False)

    if args.location == 'pisa':
        for idx in tqdm(range(len(metadata_df))):
            row = metadata_df.iloc[idx]
            df = pd.read_csv(os.path.join(in_loc, f'{row.name}.mp4.csv'), parse_dates=['timestamp'], index_col='timestamp').fillna(method='ffill').dropna()

            if row.to_be_cut_1 == 'ok':
                if args.divide:
                    for j, chunk in enumerate(np.array_split(df, len(df)//T + 1)):
                        if is_valid_segment(chunk):
                            chunk.to_csv(os.path.join(out_loc, f'{row.name}_1_{j}.csv'), index=False)
                else:
                    if is_valid_segment(df):
                        df.to_csv(os.path.join(out_loc, f'{row.name}_1.csv'), index=False)
            else:
                start_current_slice = '00:00:00'
                i = 1
                while True:
                    if f'to_be_cut_{i}' not in metadata_df.columns:
                        break

                    to_be_cut = row[f'to_be_cut_{i}']
                    if pd.isnull(to_be_cut):
                        break

                    end_current_slice, start_next_slice = to_be_cut.split('-')

                    if end_current_slice != start_current_slice:
                        df_slice = df.between_time(start_current_slice, end_current_slice)
                        if args.divide:
                            for j, chunk in enumerate(np.array_split(df_slice, len(df_slice)//T + 1)):
                                if is_valid_segment(chunk):
                                    chunk.to_csv(os.path.join(out_loc, f'{row.name}_{i}_{j}.csv'), index=False)
                        else:
                            if is_valid_segment(df_slice):
                                df_slice.to_csv(os.path.join(out_loc, f'{row.name}_{i}.csv'), index=False)

                    start_current_slice = start_next_slice
                    i = i + 1
                
                if start_current_slice != '01:00:00':
                    df_slice = df.between_time(start_current_slice, '01:00:00')
                    if len(df_slice) > 0:
                        if args.divide:
                            for j, chunk in enumerate(np.array_split(df_slice, len(df_slice)//T + 1)):
                                if is_valid_segment(chunk):
                                    chunk.to_csv(os.path.join(out_loc, f'{row.name}_{i}_{j}.csv'), index=False)
                        else:
                            if is_valid_segment(df_slice):
                                df_slice.to_csv(os.path.join(out_loc, f'{row.name}_{i}.csv'), index=False)

