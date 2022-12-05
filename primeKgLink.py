#%%
from tdc.resource import PrimeKG
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import optuna
data = PrimeKG(path = './data')
# %%
from primeKGmethods import createGraph, encoder, LinkPredictor, train, test
graph = createGraph(data)
# computes and saves embedding with node2vec
encoder(graph)
#%%
graph.x = torch.load("embedding.pt", map_location="cpu")
from torch_geometric.transforms import RandomLinkSplit
# TODO: richtige neg edges erstellen, jetz werden tatsächliche genommen
transform = RandomLinkSplit(is_undirected=True,split_labels=True)
split_edge = transform(graph)
#%%
from torch.utils.data import DataLoader

def train(predictor, x, edges, optimizer, batch_size):
    predictor.train()

    edges = edges.to(device)#TODO

    total_loss = total_examples = 0
    for perm in DataLoader(range(edges.edge_index.shape[1]), batch_size, shuffle=True):
        optimizer.zero_grad()

        edge = edges.edge_index.t()[perm]
        edge = edge.t().to(device)

        pos_out = predictor(x[edge[0]], x[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, x.size(0), edge.size(), dtype=torch.long,
                             device=x.device)
        neg_out = predictor(x[edge[0]], x[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples
# %%
from tqdm import tqdm
num_layers = 3
hidden_channels = 256
dropout = 0.3
batch_size = 64 * 1024
lr = 0.01
epochs = 20


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

predictor = LinkPredictor(graph.x.size(-1), hidden_channels, 1,
                          num_layers, dropout).to(device)

optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)

train_edges,valid_edges,test_edges = split_edge
train_edges = train_edges.to(device)

predictor.reset_parameters()
graph.x = graph.x.to(device)
for epoch in tqdm(range(1, 1 + epochs)):
    loss = train(predictor, graph.x, train_edges, optimizer, batch_size)
    print("Epoch:",epoch,"\tLoss:",loss)
#%%

#def test(predictor, x, split_edge, batch_size):
test_edges = test_edges.to(device)
e = test_edges.pos_edge_label_index[0][:100]
e2 = test_edges.pos_edge_label_index[1][:100]
p = predictor(graph.x[e],graph.x[e2])
# %%
import numpy as np
acc = 0
predCount = 0
for perm in tqdm(DataLoader(range(test_edges.edge_index.shape[1]), batch_size, shuffle=True)):
    edge = test_edges.pos_edge_index_label.t()[perm]
    edge = edge.t().to(device)
    pos_out = predictor(graph.x[edge[0]], graph.x[edge[1]])
    acc += sum(pos_out.numpy().round(decimals=3) == torch.ones(pos_out.shape[0],1).numpy())[0]
    predCount += pos_out.shape[0]
    # Just do some trivial random sampling.
    neg_edge_label_index
    neg_out = predictor(x[edge[0]], x[edge[1]])
# %%
