#!/usr/bin/env python

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
import feather


def read_csv(f):
    df = pd.read_csv(f, usecols=lambda x: x != 'timestamp')

    # Add neck joint between shoulders
    for coord in ['x', 'y', 'z']:
        df[f'neck_{coord}'] = (df[f'left_shoulder_{coord}'] + df[f'right_shoulder_{coord}']) / 2
    
    # Add hip joint between left and righ hip
    for coord in ['x', 'y', 'z']:
        df[f'hip_{coord}'] = (df[f'left_hip_{coord}'] + df[f'right_hip_{coord}']) / 2
    
    return df

def rotation_matrix_x(angle):
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])

def rotation_matrix_y(angle):
    return np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])

def rotation_matrix_z(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

def rotate_df(df, rot_matrix):
    n_col = len(df.columns)
    n_joints = int(n_col / 3)
    joint_coords = np.array(df.iloc[:, :]).T
    rot_matrices = np.kron(np.eye(n_joints), rot_matrix)
    joint_coords = np.dot(rot_matrices, joint_coords)
    df.iloc[:, :] = joint_coords.T
    return df

def generate_random_angle(min_angle, max_angle):
    """Generate a random angle between min_angle and max_angle."""
    min_radian = np.deg2rad(min_angle)
    max_radian = np.deg2rad(max_angle)
    return np.random.uniform(min_radian, max_radian)

def random_rotate(df, angle):
    """Apply random rotations to the dataframe."""
    angle_x = generate_random_angle(-angle, angle)
    angle_y = generate_random_angle(-angle, angle)
    angle_z = generate_random_angle(-angle, angle)
    
    rot_x = rotation_matrix_x(angle_x)
    df = rotate_df(df, rot_x)
    
    rot_y = rotation_matrix_y(angle_y)
    df = rotate_df(df, rot_y)
    
    rot_z = rotation_matrix_z(angle_z)
    df = rotate_df(df, rot_z)
    
    return df

def mirror(df):
    x_cols = [col for col in df.columns if col.endswith('_x')]
    df[x_cols] = df[x_cols] * -1

    return df

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-l', '--location', type=str)
    arg_parser.add_argument('-i', '--in-dir', type=str)
    arg_parser.add_argument('-o', '--out-dir', type=str)
    arg_parser.add_argument('-n', '--num-rotations', type=int)
    arg_parser.add_argument('-a', '--angle', default=5, type=int)
    args = arg_parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, args.location), exist_ok=True)

    flist = glob(os.path.join(args.in_dir, args.location, '*.csv'))

    for f in tqdm(flist, desc=f'Augmenting {args.location} data'):
        dfs_list = []
        
        # Save original dataframe
        df = read_csv(f)
        dfs_list.append(('original', df))
        
        # Random rotations
        for i in range(args.num_rotations):
            rotated_df = random_rotate(df, args.angle)
            dfs_list.append((f'rotation_{i}', rotated_df))
        
        # Convert list of dataframes to multi-index dataframe
        df_multi = pd.concat(dict(dfs_list), axis=0)
        
        # Save as Feather file
        filename = os.path.basename(f).replace('.csv', '.feather')
        feather.write_dataframe(df_multi, os.path.join(args.out_dir, args.location, filename))