import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from modules.constants import FEATURE_LIST
from modules.aagcn import AAGCN
from momo import Momo


class AdaptiveSTGCN(pl.LightningModule):
    def __init__(self, in_channels, edge_index, num_nodes, learning_rate, 
                 adaptive, attention, masking, concat_features,
                 store_graphs=False, store_attention=False):
        super(AdaptiveSTGCN, self).__init__()
        self.data_bn = torch.nn.BatchNorm1d(in_channels * num_nodes)
        self.l1 = AAGCN(
            in_channels=in_channels,
            out_channels=32,
            edge_index=edge_index,
            num_nodes=num_nodes,
            adaptive=adaptive,
            attention=attention,
            masking=masking,
            store_graphs=store_graphs,
            store_attention=store_attention,
            stride=1,
            residual=False,
        )
        self.l2 = AAGCN(
            in_channels=32,
            out_channels=32,
            edge_index=edge_index,
            num_nodes=num_nodes,
            adaptive=adaptive,
            attention=attention,
            masking=masking,
            store_graphs=store_graphs,
            store_attention=store_attention,
            stride=1
        )
        self.l3 = AAGCN(
            in_channels=32,
            out_channels=64,
            edge_index=edge_index,
            num_nodes=num_nodes,
            adaptive=adaptive,
            attention=attention,
            masking=masking,
            store_graphs=store_graphs,
            store_attention=store_attention,
            stride=1
        )
        self.l4 = AAGCN(
            in_channels=64,
            out_channels=64,
            edge_index=edge_index,
            num_nodes=num_nodes,
            adaptive=adaptive,
            attention=attention,
            masking=masking,
            store_graphs=store_graphs,
            store_attention=store_attention,
            stride=1
        )

        self.learning_rate = learning_rate
        self.concat_features = concat_features

        fc_input_size = 64 + len(FEATURE_LIST) if concat_features else 64

        self.fc1 = torch.nn.Linear(fc_input_size, 32)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, x, fts):
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 1, 3, 2).contiguous().view(N, C, T, V)

        x = self.l1(x) # (N, C_in, T, V) -> (N, C_out, T//stride, V)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)

        c_new = x.size(1)
        x = x.view(N, c_new, -1) # N, C, T*V
        x = x.mean(2) # N, C

        if self.concat_features:
            x = torch.cat((x, fts), dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x) # N, 1
        return x

    def configure_optimizers(self):
        optimizer = Momo(self.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     'monitor': 'val_loss'
            # }
        }

    def training_step(self, train_batch, batch_idx):
        X, y, fts = train_batch
        y_pred = self.forward(X, fts)
        loss = F.mse_loss(y_pred.squeeze(), y.squeeze())
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y, fts = val_batch
        y_pred = self.forward(X, fts)
        loss = F.mse_loss(y_pred.squeeze(), y.squeeze())
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        X, y, fts = test_batch
        y_pred = self.forward(X, fts)
        y, y_pred = y.squeeze(), y_pred.squeeze()

        loss = F.mse_loss(y_pred, y)
        self.log('val_loss', loss)

        rmse = torch.sqrt(loss)
        self.log('val_rmse', rmse)
        
        mae = F.l1_loss(y_pred, y)
        self.log('val_mae', mae)

        mape = 100 * torch.mean(torch.abs((y - y_pred) / y))
        self.log('val_mape', mape)

    def predict_step(self, batch, batch_idx):
        X, y, fts = batch
        y_pred = self.forward(X, fts)
        return y_pred