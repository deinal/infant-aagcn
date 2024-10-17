#!/usr/bin/env python

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob


def read_csv(f, use_2d):
    df = pd.read_csv(f, usecols=lambda x: x != 'timestamp')

    # Add neck joint between shoulders
    coords = ['x', 'y'] if use_2d else ['x', 'y', 'z']
    for coord in coords:
        df[f'neck_{coord}'] = (df[f'left_shoulder_{coord}'] + df[f'right_shoulder_{coord}']) / 2

    # Add hip joint between left and right hip
    for coord in coords:
        df[f'hip_{coord}'] = (df[f'left_hip_{coord}'] + df[f'right_hip_{coord}']) / 2

    if use_2d:
        df = df[[col for col in df.columns if '_z' not in col]]

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

def rotation_matrix_z_2d(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

def rotate_df(df, rot_matrix):
    n_col = len(df.columns)
    n_joints = int(n_col / rot_matrix.shape[0])  # 2 for 2D or 3 for 3D data
    joint_coords = np.array(df.iloc[:, :]).T
    rot_matrices = np.kron(np.eye(n_joints), rot_matrix)
    joint_coords = np.dot(rot_matrices, joint_coords)
    df.iloc[:, :] = joint_coords.T
    return df

def rotate_video_2d(df):
    # 2D rotation (around z-axis)
    neck_joint = np.array(df[['neck_x', 'neck_y']])
    hip_joint = np.array(df[['hip_x', 'hip_y']])
    avg_spine = (hip_joint - neck_joint).mean(axis=0)

    cosine_angle_z = np.dot([0, 1], avg_spine) / (np.linalg.norm([0, 1]) * np.linalg.norm(avg_spine))
    angle_z = np.arccos(cosine_angle_z)
    if avg_spine[0] < 0:
        angle_z = -angle_z

    rot_matrix = rotation_matrix_z_2d(angle_z)

    df = rotate_df(df, rot_matrix)

    return df

def rotate_video_3d(df):
    # Rotation around z-axis
    neck_joint = np.array(df[['neck_x', 'neck_y', 'neck_z']])
    hip_joint = np.array(df[['hip_x', 'hip_y', 'hip_z']])
    avg_spine = (hip_joint - neck_joint).mean(axis=0)

    avg_spine_xy = np.array([avg_spine[0], avg_spine[1], 0])
    cosine_angle_z = np.dot([0, 1, 0], avg_spine) / (np.linalg.norm([0, 1, 0]) * np.linalg.norm(avg_spine_xy))
    angle_z = np.arccos(cosine_angle_z)
    if avg_spine[0] < 0:
        angle_z = -angle_z

    rot_matrix = rotation_matrix_z(angle_z)
    df = rotate_df(df, rot_matrix)

    # Rotation around x-axis
    neck_joint = np.array(df[['neck_x', 'neck_y', 'neck_z']])
    hip_joint = np.array(df[['hip_x', 'hip_y', 'hip_z']])
    avg_spine = (hip_joint - neck_joint).mean(axis=0)

    avg_spine_yz = np.array([0, avg_spine[1], avg_spine[2]])
    cosine_angle_x = np.dot([0, 1, 0], avg_spine_yz) / (np.linalg.norm([0, 1, 0]) * np.linalg.norm(avg_spine_yz))
    angle_x = np.arccos(cosine_angle_x)
    if avg_spine[2] > 0:
        angle_x = -angle_x

    rot_matrix = rotation_matrix_x(angle_x)
    df = rotate_df(df, rot_matrix)

    # Rotation around y-axis
    right_hip = np.array(df[['right_hip_x', 'right_hip_y', 'right_hip_z']])
    left_hip = np.array(df[['left_hip_x', 'left_hip_y', 'left_hip_z']])
    hipline = (left_hip - right_hip)

    right_shoulder = np.array(df[['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z']])
    left_shoulder = np.array(df[['left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z']])
    shoulderline = (left_shoulder - right_shoulder)

    avg_backline = ((hipline + shoulderline) / 2).mean(axis=0)

    avg_backline_xz = np.array([avg_backline[0], 0, avg_backline[2]])
    cosine_angle_y = np.dot([1, 0, 0], avg_backline_xz) / (np.linalg.norm([1, 0, 0]) * np.linalg.norm(avg_backline_xz))
    angle_y = np.arccos(cosine_angle_y)
    if avg_backline[2] < 0:
        angle_y = -angle_y

    rot_matrix = rotation_matrix_y(angle_y)
    df = rotate_df(df, rot_matrix)

    return df


def center_video(df, use_2d):
    n_col = len(df.columns)
    n_joints = int(n_col / (2 if use_2d else 3))  # 2 columns per joint for 2D, 3 for 3D

    # Translate so the neck joint is at the origin
    coords = ['x', 'y'] if use_2d else ['x', 'y', 'z']
    neck_joint = np.array(df[[f'neck_{coord}' for coord in coords]])
    for joint in range(n_joints):
        for coord_idx, coord in enumerate(coords):
            df.iloc[:, joint*len(coords) + coord_idx] -= neck_joint[:, coord_idx]

    return df


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-l', '--location')
    arg_parser.add_argument('-i', '--in-dir')
    arg_parser.add_argument('-o', '--out-dir')
    arg_parser.add_argument('--use-2d', action='store_true')
    args = arg_parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, args.location), exist_ok=True)

    flist = glob(os.path.join(args.in_dir, args.location, '*.csv'))

    for f in tqdm(flist, desc=f'Rotating {args.location} data'):
        df = read_csv(f, args.use_2d)
        df = center_video(df, args.use_2d)
        df = rotate_video_2d(df) if args.use_2d else rotate_video_3d(df)
        df.to_csv(os.path.join(args.out_dir, args.location, os.path.basename(f)), index=False)
