#!/usr/bin/env python

import os
import torch
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import pytorch_lightning as pl
import bin.plot as plot
from modules.utils import get_logger, LossCallback
from modules.data import InfantMotionDataset
from modules.model import AdaptiveSTGCN
from modules.constants import N, PHYS_EDGE_INDEX, COORD_EDGE_INDEX, FC_EDGE_INDEX


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data-dir', type=str)
    arg_parser.add_argument('--output-dir', type=str)
    arg_parser.add_argument('--age-file', default='metadata/combined.csv', type=str)
    arg_parser.add_argument('--feature-file', default='data/features.csv', type=str)
    arg_parser.add_argument('--streams', default='j,b,v,a', type=str)
    arg_parser.add_argument('--xy-data', action='store_true')
    arg_parser.add_argument('--edges', default='phys', choices=['phys', 'coord', 'fc'], type=str)
    arg_parser.add_argument('--concat-features', action='store_true')
    arg_parser.add_argument('--k-folds', default=10, type=int)
    arg_parser.add_argument('--log-file', type=str)
    arg_parser.add_argument('--epochs', default=20, type=int)
    arg_parser.add_argument('--batch-size', default=32, type=int)
    arg_parser.add_argument('--learning-rate', default=0.01, type=float)
    arg_parser.add_argument('--hidden-dim', default=256, type=int)
    arg_parser.add_argument('--kt', default=13, type=int)
    arg_parser.add_argument('--adaptive', action='store_true')
    arg_parser.add_argument('--attention', action='store_true')
    arg_parser.add_argument('--masking', action='store_true')
    arg_parser.add_argument('--num-workers', default=8, type=int)
    arg_parser.add_argument('--devices', default=1, type=int)
    args = arg_parser.parse_args()

    logger = get_logger('infant-aagcn', filename=args.log_file)

    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')

    age_data = pd.read_csv(args.age_file, dtype={'test_id': str})
    age_data = age_data[age_data.outcome != 2]
    fts_data = pd.read_csv(args.feature_file)
    
    streams = args.streams.split(',')

    if args.edges == 'phys':
        edge_index = torch.tensor(PHYS_EDGE_INDEX)
    elif args.edges == 'coord':
        edge_index = torch.tensor(COORD_EDGE_INDEX)
    elif args.edges == 'fc':
        edge_index = torch.tensor(FC_EDGE_INDEX)
    else:
        raise ValueError(f'Invalid edge type: {args.edges}')

    kfold = KFold(n_splits=args.k_folds, shuffle=False)

    for fold_n, (train_indices, val_indices) in enumerate(kfold.split(age_data), start=1):
        output_dir = os.path.join(args.output_dir, f'fold_{fold_n}')
        os.makedirs(output_dir, exist_ok=True)

        # Set up dataloaders
        train_metadata = age_data.iloc[train_indices]
        val_metadata = age_data.iloc[val_indices]

        train_ids = age_data.iloc[train_indices].test_id.values
        val_ids = age_data.iloc[val_indices].test_id.values

        train_fts = fts_data[fts_data['test_id'].isin(train_ids)]
        val_fts = fts_data[fts_data['test_id'].isin(val_ids)]

        train_segments = train_fts.segment.tolist()
        val_segments = val_fts.segment.tolist()

        train_dataset = InfantMotionDataset(args.data_dir, train_fts, streams, xy_data=args.xy_data, predict=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        val_dataset = InfantMotionDataset(args.data_dir, val_fts, streams, xy_data=args.xy_data, predict=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        # Initialize model and callbacks
        model = AdaptiveSTGCN(
            in_channels=2*len(streams) if args.xy_data else 3*len(streams),
            edge_index=edge_index,
            num_nodes=N,
            learning_rate=args.learning_rate,
            adaptive=args.adaptive,
            attention=args.attention,
            masking=args.masking,
            concat_features=args.concat_features,
            hidden_dim=args.hidden_dim,
            kt=args.kt
        )

        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        loss_callback = LossCallback()
        
        checkpointing = pl.callbacks.ModelCheckpoint(
            dirpath=output_dir,
            verbose=True,
            monitor='val_loss',
            mode='min',
            save_top_k=5
        )

        # Train the model
        trainer = pl.Trainer(
            accelerator='auto',
            devices=args.devices,
            max_epochs=args.epochs,
            log_every_n_steps=1,
            callbacks=[lr_monitor, checkpointing, loss_callback]
        )

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Load the best model for testing
        model = AdaptiveSTGCN.load_from_checkpoint(
            checkpoint_path=checkpointing.best_model_path,
            in_channels=2*len(streams) if args.xy_data else 3*len(streams),
            edge_index=edge_index,
            num_nodes=N,
            learning_rate=args.learning_rate,
            adaptive=args.adaptive,
            attention=args.attention,
            masking=args.masking,
            concat_features=args.concat_features,
            hidden_dim=args.hidden_dim,
            kt=args.kt
        )
        [results] = trainer.test(model, dataloaders=val_loader)
        
        # Save predictions
        train_dataset = InfantMotionDataset(args.data_dir, train_fts, streams, xy_data=args.xy_data, predict=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        val_dataset = InfantMotionDataset(args.data_dir, val_fts, streams, xy_data=args.xy_data, predict=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        train_predictions = trainer.predict(model, dataloaders=train_loader)
        train_predictions = torch.cat(train_predictions, dim=0)
        train_predictions = [float(pred) for pred in train_predictions]

        val_predictions = trainer.predict(model, dataloaders=val_loader)
        val_predictions = torch.cat(val_predictions, dim=0)
        val_predictions = [float(pred) for pred in val_predictions]
        
        np.save(Path(output_dir, 'train_predictions.npy'), train_predictions)
        np.save(Path(output_dir, 'val_predictions.npy'), val_predictions)

        df_train, df_val = plot.create_dataframe(
            train_predictions, val_predictions,
            train_segments, val_segments,
            args.age_file
        )

        plot.plot_scatter_comparison(df_train, df_val, output_dir)
        plot.plot_scatter_average(df_train, df_val, output_dir)
        plot.plot_scatter_median(df_train, df_val, output_dir)

        # Rename best model and remove excess models
        best_model_path = os.path.join(output_dir, 'best_model.ckpt')
        os.rename(checkpointing.best_model_path, best_model_path)
        for filename in os.listdir(output_dir):
            if filename.endswith('.ckpt') and filename != 'best_model.ckpt':
                file_path = os.path.join(output_dir, filename)
                os.remove(file_path)

        # Save metadata as json
        metadata = {
            'ckpt_path': best_model_path,
            'results': results,
            'args': vars(args),
            'params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'epoch_losses': loss_callback.losses,
            'train_segments': train_segments,
            'val_segments': val_segments,
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)