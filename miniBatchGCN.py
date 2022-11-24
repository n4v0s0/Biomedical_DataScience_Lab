"""
Based on the GCN implementation from Matthias Fey:
https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/proteins/gnn.py

Node predictian on OGB Proteins, using Mini-Batch Training and Optuna Hyperparameter Optimization
"""

#%%
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from tqdm import tqdm

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

def train(model, data, train_idx, optimizer, trainLoader):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    totalLoss = 0
    # iterate over data as batches, add up loss and return averaged total loss
    for batch in trainLoader:
        optimizer.zero_grad()
        # redefine train_idx to circumvent index out range error for batch
        train_idx = train_idx[:batch.num_nodes]
        out = model(batch.x, batch.adj_t)[train_idx]
        loss = criterion(out, batch.y[train_idx].to(torch.float))
        loss.backward()
        totalLoss += loss.item()
        optimizer.step()

    return totalLoss/len(trainLoader)


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    y_pred = model(data.x, data.adj_t)

    train_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc
# %%
# cant use cpu as device due to memory limit
# still fighting with cuda aswell :D
#device = 'cpu'
#device = torch.device(device)

dataset = PygNodePropPredDataset(
    name='ogbn-proteins', transform=T.ToSparseTensor(attr='edge_attr'))
data = dataset[0]

# Move edge features to node features.
data.x = data.adj_t.mean(dim=1)
data.adj_t.set_value_(None)

split_idx = dataset.get_idx_split()
train_idx = split_idx['train']#.to(device)

# %%
import optuna
from torch_geometric.loader import NeighborLoader
# this is the routine called by the different optuna trials
def routine(data,params,trial):
  # init hyperparameters for the trial
  hidden_channels = params["hidden_channels"]
  num_layers = params["layers"]
  dropout = params["dropout"]
  learning_rate = params["learning_rate"]
  epochs = params["epochs"]
  batch_size = params["batch_size"]
  output_layers = 112
  model = GCN(data.num_features, hidden_channels, output_layers, num_layers, dropout)

  # Pre-compute GCN normalization.
  adj_t = data.adj_t.set_diag()
  deg = adj_t.sum(dim=1).to(torch.float)
  deg_inv_sqrt = deg.pow(-0.5)
  deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
  adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
  data.adj_t = adj_t
  #data = data.to(device)
  evaluator = Evaluator(name='ogbn-proteins')
  model.reset_parameters()
  optimizer = torch.optim.Adam(model.parameters(), learning_rate)

  # datalaoder that takes up to 30 neighbours of each node into account
  trainLoader = NeighborLoader(data,num_neighbors=[30],batch_size=batch_size,input_nodes=split_idx['train'])
  
  for epoch in tqdm(range(1, epochs+1)):
    loss = train(model, data, train_idx, optimizer, trainLoader)
    result = test(model, data, split_idx, evaluator)
    train_rocauc, valid_rocauc, test_rocauc = result
    print(
            f'Epoch: {epoch:02d}, '
            f'Loss: {loss:.4f}, '
            f'Train: {100 * train_rocauc:.2f}%, '
            f'Valid: {100 * valid_rocauc:.2f}% '
        )
    # use valid score to decide to prune current trial
    trial.report(valid_rocauc, epoch)
    if trial.should_prune():
      raise optuna.exceptions.TrialPruned()
  result = test(model, data, split_idx, evaluator)
  train_rocauc, valid_rocauc, test_rocauc = result
  return test_rocauc
def objective(trial,data):
    params = {
        "learning_rate" : trial.suggest_loguniform("learning_rate", low=10e-5, high=10e-1),
        "layers" : trial.suggest_int("layers", low=3, high=10),
        # had to choose hidden channels rather low due to my pc crashing on high values :D
        # should actually be between outputsize and featuresize/10
        "hidden_channels" : trial.suggest_int("hidden_channels", low=16, high=128),
        "optimizer" : trial.suggest_categorical("optimizier",["Adam","RMSprop","SGD"]),
        "dropout" : trial.suggest_float("dropout", low=0.1, high=0.5, step=0.1),
        "epochs" : trial.suggest_int("epochs", low=20, high=500),
        # choose batch size between average degree of nodes and number of nodes
        # in my case smaller values to prevent pc crashes 
        "batch_size": trial.suggest_int("batch_size", low=500, high=data.num_nodes/10)
    }
    print("\nThis trials parameters:\n",params)
    result = routine(data,params,trial)
    return result
# %%
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(lambda trial: objective(trial,data),n_trials=5)
# %%
