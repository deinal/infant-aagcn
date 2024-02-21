#!/usr/bin/env python

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


def get_calendar_age_and_id(segment, age_data):
    test_id = segment.split('_')[0]
    target_numpy = age_data.loc[test_id].corrected_age
    return float(target_numpy), test_id

def create_dataframe(y_pred_train, y_pred_val, train_segments, val_segments, age_file):
    age_data = pd.read_csv(age_file, dtype={'test_id': str}).set_index('test_id')

    y_true_train, ids_train = zip(*[get_calendar_age_and_id(segment, age_data) for segment in train_segments])
    y_true_val, ids_val = zip(*[get_calendar_age_and_id(segment, age_data) for segment in val_segments])

    df_train = pd.DataFrame({
        'y_true': np.array(y_true_train),
        'y_pred': np.array(y_pred_train),
        'test_id': ids_train
    })
    df_val = pd.DataFrame({
        'y_true': np.array(y_true_val),
        'y_pred': np.array(y_pred_val),
        'test_id': ids_val
    })

    return df_train, df_val

def plot_scatter_comparison(df_train, df_val, save_dir):
    plt.figure(figsize=(5, 5))
    plt.scatter(df_train.y_true, df_train.y_pred, s=7, 
        label=f'Train, R2 {r2_score(df_train.y_true, df_train.y_pred):.2f}, RMSE {mean_squared_error(df_train.y_true, df_train.y_pred, squared=False):.2f}')
    plt.scatter(df_val.y_true, df_val.y_pred, s=7, 
        label=f'Val, R2 {r2_score(df_val.y_true, df_val.y_pred):.2f}, RMSE {mean_squared_error(df_val.y_true, df_val.y_pred, squared=False):.2f}')

    p1 = max(df_train[['y_true', 'y_pred']].max().max(), df_val[['y_true', 'y_pred']].max().max())
    p2 = min(df_train[['y_true', 'y_pred']].min().min(), df_val[['y_true', 'y_pred']].min().min())
    plt.plot([p1, p2], [p1, p2], 'k-', linewidth=1)
    plt.xlabel('Observations')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'scatter_comparison.png'))
    plt.close()

def plot_scatter_average(df_train, df_val, save_dir):
    grouped_train = df_train.groupby('test_id').agg(['mean', 'std'])
    grouped_val = df_val.groupby('test_id').agg(['mean', 'std'])

    plt.figure(figsize=(5, 5))
    plt.errorbar(grouped_train['y_true']['mean'], grouped_train['y_pred']['mean'], 
        yerr=grouped_train['y_pred']['std'], fmt='o', capsize=0, markersize=4,
        label=f'Train, R2 {r2_score(grouped_train["y_true"]["mean"], grouped_train["y_pred"]["mean"]):.2f}, RMSE {mean_squared_error(grouped_train["y_true"]["mean"], grouped_train["y_pred"]["mean"], squared=False):.2f}')
    plt.errorbar(grouped_val['y_true']['mean'], grouped_val['y_pred']['mean'], 
        yerr=grouped_val['y_pred']['std'], fmt='o', capsize=0, markersize=4,
        label=f'Val, R2 {r2_score(grouped_val["y_true"]["mean"], grouped_val["y_pred"]["mean"]):.2f}, RMSE {mean_squared_error(grouped_val["y_true"]["mean"], grouped_val["y_pred"]["mean"], squared=False):.2f}')

    p1 = max(df_train[['y_true', 'y_pred']].max().max(), df_val[['y_true', 'y_pred']].max().max())
    p2 = min(df_train[['y_true', 'y_pred']].min().min(), df_val[['y_true', 'y_pred']].min().min())
    plt.plot([p1, p2], [p1, p2], 'k-', linewidth=1)
    plt.xlabel('Observations')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'scatter_average.png'))
    plt.close()

def compute_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    return Q3 - Q1

def plot_scatter_median(df_train, df_val, save_dir):
    grouped_train_median = df_train.groupby('test_id').median()
    grouped_val_median = df_val.groupby('test_id').median()
    
    train_iqr = df_train.groupby('test_id')['y_pred'].apply(compute_iqr)
    val_iqr = df_val.groupby('test_id')['y_pred'].apply(compute_iqr)

    plt.figure(figsize=(5, 5))
    plt.errorbar(grouped_train_median['y_true'], grouped_train_median['y_pred'], yerr=train_iqr, fmt='o', capsize=0, markersize=4,
        label=f'Train, R2 {r2_score(grouped_train_median.y_true, grouped_train_median.y_pred):.2f}, RMSE {mean_squared_error(grouped_train_median.y_true, grouped_train_median.y_pred, squared=False):.2f}')
    plt.errorbar(grouped_val_median['y_true'], grouped_val_median['y_pred'], yerr=val_iqr, fmt='o', capsize=0, markersize=4,
        label=f'Val, R2 {r2_score(grouped_val_median.y_true, grouped_val_median.y_pred):.2f}, RMSE {mean_squared_error(grouped_val_median.y_true, grouped_val_median.y_pred, squared=False):.2f}')

    p1 = max(df_train[['y_true', 'y_pred']].max().max(), df_val[['y_true', 'y_pred']].max().max())
    p2 = min(df_train[['y_true', 'y_pred']].min().min(), df_val[['y_true', 'y_pred']].min().min())
    plt.plot([p1, p2], [p1, p2], 'k-', linewidth=1)
    plt.xlabel('Observations')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'scatter_median.png'))
    plt.close()

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-o', '--output-dir', type=str)
    arg_parser.add_argument('-l', '--location', type=str, default='combined')
    args = arg_parser.parse_args()

    with open(os.path.join(args.output_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    train_predictions = np.load(os.path.join(args.output_dir, 'train_predictions.npy'))
    val_predictions = np.load(os.path.join(args.output_dir, 'val_predictions.npy'))

    df_train, df_val = create_dataframe(
        train_predictions, val_predictions,
        metadata['train_segments'],  metadata['val_segments'],
        os.path.join('metadata', f'{args.location}.csv')
    )

    plot_scatter_comparison(df_train, df_val, args.output_dir)
    plot_scatter_average(df_train, df_val, args.output_dir)
    plot_scatter_median(df_train, df_val, args.output_dir)