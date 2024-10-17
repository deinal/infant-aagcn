#!/usr/bin/env python

import os
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from modules.data import InfantMotionDataset
from modules.model import AdaptiveSTGCN
from modules.constants import N, PHYS_EDGE_INDEX, COORD_EDGE_INDEX, FC_EDGE_INDEX


if __name__ == '__main__':
    # Parse command line arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model-dir', required=True, type=str, help='Directory containing the trained models for each fold')
    arg_parser.add_argument('--output-dir', required=True, type=str, help='Directory to store the predictions')
    arg_parser.add_argument('--data-dir', default='data/streams/combined', type=str, help='Directory containing the data')
    arg_parser.add_argument('--age-file', default='metadata/combined.csv', type=str, help='Path to the age file')
    arg_parser.add_argument('--feature-file', default='data/features.csv', type=str, help='Path to the feature file')
    arg_parser.add_argument('--batch-size', default=1, type=int)
    arg_parser.add_argument('--num-workers', default=1, type=int)
    arg_parser.add_argument('--devices', default=1, type=int)
    args = arg_parser.parse_args()

    # Load age and feature data
    age_data = pd.read_csv(args.age_file, dtype={'test_id': str})
    fts_data = pd.read_csv(args.feature_file)

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # DataFrame to store combined predictions
    combined_predictions = pd.DataFrame(columns=['fold', 'test_id', 'y_true', 'y_pred'])

    # Read settings from the first fold's metadata as a reference
    metadata_path = Path(args.model_dir, 'fold_1', 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    k_folds = metadata['args']['k_folds']
    streams = metadata['args']['streams'].split(',')
    xy_data = metadata['args']['xy_data']
    if metadata['args']['edges'] == 'phys':
        edge_index = torch.tensor(PHYS_EDGE_INDEX)
    if metadata['args']['edges'] == 'coord':
        edge_index = torch.tensor(COORD_EDGE_INDEX)
    if metadata['args']['edges'] == 'fc':
        edge_index = torch.tensor(FC_EDGE_INDEX)

    # Iterate over each fold
    for fold_n in tqdm(range(1, k_folds + 1)):
        # Load metadata for the fold
        fold_metadata_path = Path(args.model_dir, f'fold_{fold_n}', 'metadata.json')
        with open(fold_metadata_path, 'r') as f:
            fold_metadata = json.load(f)

        # Load the best model for the fold
        model = AdaptiveSTGCN.load_from_checkpoint(
            checkpoint_path=fold_metadata['ckpt_path'],
            edge_index=edge_index,
            num_nodes=N,
            in_channels=2*len(streams) if xy_data else 3*len(streams),
            learning_rate=fold_metadata['args']['learning_rate'],
            adaptive=fold_metadata['args']['adaptive'],
            attention=fold_metadata['args']['attention'],
            masking=fold_metadata['args']['masking'],
            concat_features=fold_metadata['args']['concat_features'],
            hidden_dim=fold_metadata['args']['hidden_dim'],
            kt=fold_metadata['args']['kt']
        )

        # Predict for all test_ids with severe outcome
        prediction_ids = age_data[age_data.outcome == 2].test_id.values
        prediction_fts = fts_data[fts_data['test_id'].isin(prediction_ids)]

        # Set up DataLoader for prediction data
        prediction_dataset = InfantMotionDataset(fold_metadata['args']['data_dir'], prediction_fts, streams, xy_data=xy_data, predict=True)
        prediction_loader = DataLoader(prediction_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        # Predict
        trainer = pl.Trainer(accelerator='auto', devices=args.devices)
        predictions = trainer.predict(model, dataloaders=prediction_loader)
        predictions = torch.cat(predictions, dim=0).numpy()

        # Append predictions to the DataFrame
        fold_predictions = pd.DataFrame({
            'fold': fold_n,
            'segment': prediction_fts['segment'],
            'test_id': prediction_fts['test_id'],
            'outcome': prediction_fts['outcome'],
            'y_true': prediction_fts['corrected_age'],
            'y_pred': predictions.flatten()
        })
        combined_predictions = pd.concat([combined_predictions, fold_predictions], ignore_index=True)

    # Save the combined predictions to CSV
    model_name = Path(args.model_dir).name
    combined_predictions.to_csv(Path(args.output_dir, f'{model_name}_predictions.csv'), index=False)
