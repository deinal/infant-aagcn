#!/usr/bin/env python

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import feather
from modules.constants import T, NODE_INDEX, PHYSICAL_EDGES


def create_b_stream(df, edges):
    b_stream_df = pd.DataFrame()

    # Iterate over edge labels to determine source and target joints
    for source_joint, target_joint in edges:
        for coord in ['x', 'y', 'z']:
            # Calculate the bone vector: target - source
            b_stream_df[f'{target_joint}_{coord}'] = df[f'{target_joint}_{coord}'] - df[f'{source_joint}_{coord}']

    return b_stream_df

def create_v_stream(df, joints):
    v_stream_df = pd.DataFrame()

    for joint in joints:
        for coord in ['x', 'y', 'z']:
            v_stream_df[f'{joint}_{coord}'] = df[f'{joint}_{coord}'].diff().fillna(0)

    return v_stream_df

def create_a_stream(df, joints):
    a_stream_df = pd.DataFrame()

    for joint in joints:
        for coord in ['x', 'y', 'z']:
            a_stream_df[f'{joint}_{coord}'] = df[f'{joint}_{coord}'].diff().fillna(0)

    return a_stream_df


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-l', '--location')
    arg_parser.add_argument('-i', '--in-dir')
    arg_parser.add_argument('-o', '--out-dir')
    args = arg_parser.parse_args()

    flist = glob(os.path.join(args.in_dir, args.location, '*.feather'))

    os.makedirs(os.path.join(args.out_dir, args.location), exist_ok=True)

    joints = list(NODE_INDEX.keys())
    joints_xyz = [joint + suffix for joint in joints for suffix in ['_x', '_y', '_z']]
    edges = [('neck', 'neck')] + PHYSICAL_EDGES

    for f in tqdm(flist, desc=f'Creating {args.location} streams'):
        df_multi = feather.read_dataframe(f)

        all_data = []

        for augment_name, df in df_multi.groupby(level=0):
            j_df = df.loc[:, joints_xyz]
            b_df = create_b_stream(j_df.copy(), edges)
            v_df = create_v_stream(j_df.copy(), joints)
            a_df = create_a_stream(v_df.copy(), joints)

            if j_df.isnull().values.any():
                print(f'Found NaN values in augmentation: {augment_name} of file: {f}. Skipping.')
                continue

            # Repeating data for each stream to ensure they have consistent shapes
            repeats = T // j_df.shape[0] + 1
            j_repeated = j_df.iloc[np.tile(range(j_df.shape[0]), repeats)][:T]
            b_repeated = b_df.iloc[np.tile(range(b_df.shape[0]), repeats)][:T]
            v_repeated = v_df.iloc[np.tile(range(v_df.shape[0]), repeats)][:T]
            a_repeated = a_df.iloc[np.tile(range(a_df.shape[0]), repeats)][:T]

            # Append data for all streams for the current augmentation
            all_data.append(pd.concat({
                (augment_name, 'j'): j_repeated,
                (augment_name, 'b'): b_repeated,
                (augment_name, 'v'): v_repeated,
                (augment_name, 'a'): a_repeated
            }))

        # Concatenate data for all augmentations and streams
        final_data = pd.concat(all_data)

        # Save as Feather file
        feather.write_dataframe(final_data, os.path.join(args.out_dir, args.location, os.path.basename(f)))