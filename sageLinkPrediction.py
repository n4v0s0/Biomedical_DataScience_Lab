from tdc.resource import PrimeKG
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import optuna
from torch_geometric.transforms import RandomLinkSplit
import mlflow
from mlflow import pytorch
from tqdm import tqdm
from torch_geometric.nn import Node2Vec
from torch.utils.data import DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import negative_sampling
from torchmetrics.classification import BinaryAUROC

def createGraph(data):
    print("Building graph...")
    df = data.get_data()
    id_dict = {}
    i = 0
    for row in tqdm(df.values):
        if row[2] not in id_dict:
            id_dict[row[2]] = {"ID": i, "type": row[3]}
            i += 1
        if row[6] in id_dict:
            continue
        else:
            id_dict[row[6]] = {"ID": i, "type": row[7]}
            i += 1
    x_list = []
    y_list = []
    for id in df.x_id:
        x_list.append(id_dict[id]["ID"])
    for id in df.y_id:
        y_list.append(id_dict[id]["ID"])
    edge_index = torch.tensor([x_list,y_list])
    relationEncode = {}
    i = 0
    relations = len(df.relation.unique())
    nodeTypes = len(df.x_type.unique())
    for relation in df.relation.unique():
        relationEncode[relation] = torch.zeros(relations)
        relationEncode[relation][i] = 1.0
        i += 1
    i = 0
    nodeTypeEncode = {}
    for nodeType in df.x_type.unique():
        nodeTypeEncode[nodeType] = torch.zeros(nodeTypes)
        nodeTypeEncode[nodeType][i] = 1.0

    for id in id_dict:
        id_dict[id]["encodedType"] = nodeTypeEncode[id_dict[id]["type"]]
    x = torch.rand(len(id_dict),256)
    y = [x["encodedType"] for x in id_dict.values()]
    y = torch.stack((y))
    graph = Data(edge_index=edge_index, x=x, y=y)
    print("Graph done!")
    return graph
class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

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

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(predictor, x, train_edges, optimizer, batch_size,device):
    predictor.train()

    train_edges, neg_train_edges = train_edges
    train_edges = train_edges.to(device)
    neg_train_edges = neg_train_edges.to(device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(train_edges.edge_index.shape[1]), batch_size, shuffle=True):
        optimizer.zero_grad()
        
        
        edge = train_edges.edge_index.t()[perm]
        edge = edge.t().to(device)

        pos_out = predictor(x[edge[0]], x[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        #random sampling for negative edges
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

def test(model, x, batch_size,train_edges, valid_edges, test_edges):
  model.eval()
  metric = BinaryAUROC(thresholds=None)
  train_edges, neg_train_edges = train_edges
  train_edges = train_edges.to(device)
  neg_train_edges = neg_train_edges.to(device)
  valid_edges, neg_valid_edges = valid_edges
  valid_edges = valid_edges.to(device)
  neg_valid_edges = neg_valid_edges.to(device)
  test_edges, neg_test_edges = test_edges
  test_edges = test_edges.to(device)
  neg_test_edges = neg_test_edges.to(device)
  #train
  for perm in DataLoader(range(train_edges.edge_index.shape[1]), batch_size, shuffle=True):
    edge = train_edges.edge_index.t()[perm]
    edge = edge.t().to(device)
    yPred = model(x[edge[0]],x[edge[1]])
    yPred = yPred.detach()
    y = torch.ones(len(yPred),1).to(device)
    yPred = torch.where(yPred>=0.9, y ,yPred)
    auroc = metric(yPred, y)
  for perm in DataLoader(range(neg_train_edges.shape[1]), batch_size, shuffle=True):
    edge = neg_train_edges.t()[perm]
    edge = edge.t().to(device)
    yPred = model(x[edge[0]],x[edge[1]])
    yPred = yPred.detach()
    y = torch.zeros(len(yPred),1).to(device)
    yPred = torch.where(yPred<=0.5, y ,yPred)
    auroc = metric(yPred, y)
  trainRoc = metric.compute()
  metric.reset()
  #valid
  total_pos_loss_valid = total_neg_loss_valid = total_examples = 0
  for perm in DataLoader(range(valid_edges.edge_index.shape[1]), batch_size, shuffle=True):
    edge = valid_edges.edge_index.t()[perm]
    edge = edge.t().to(device)
    yPred = model(x[edge[0]],x[edge[1]])
    yPred = yPred.detach()
    y = torch.ones(len(yPred),1).to(device)
    yPred = torch.where(yPred>=0.9, y ,yPred)
    auroc = metric(yPred, y)
    #valid loss pos
    pos_loss = -torch.log(yPred + 1e-15).mean()
    num_examples = yPred.size(0)
    total_examples += num_examples
    total_pos_loss_valid += pos_loss.item() * num_examples
  for perm in DataLoader(range(neg_valid_edges.shape[1]), batch_size, shuffle=True):
    edge = neg_valid_edges.t()[perm]
    edge = edge.t().to(device)
    yPred = model(x[edge[0]],x[edge[1]])
    yPred = yPred.detach()
    y = torch.zeros(len(yPred),1).to(device)
    yPred = torch.where(yPred<=0.5, y ,yPred)
    auroc = metric(yPred, y)
    #valid loss neg
    neg_loss = -torch.log(1 - yPred + 1e-15).mean()
    num_examples = yPred.size(0)
    total_neg_loss_valid += neg_loss.item() * num_examples
  validLoss = (total_pos_loss_valid + total_neg_loss_valid) / total_examples
  validRoc = metric.compute()
  metric.reset()
  #test
  for perm in DataLoader(range(test_edges.edge_index.shape[1]), batch_size, shuffle=True):
    edge = test_edges.edge_index.t()[perm]
    edge = edge.t().to(device)
    yPred = model(x[edge[0]],x[edge[1]])
    yPred = yPred.detach()
    y = torch.ones(len(yPred),1).to(device)
    yPred = torch.where(yPred>=0.9, y ,yPred)
    auroc = metric(yPred, y)
  for perm in DataLoader(range(neg_test_edges.shape[1]), batch_size, shuffle=True):
    edge = neg_test_edges.t()[perm]
    edge = edge.t().to(device)
    yPred = model(x[edge[0]],x[edge[1]])
    yPred = yPred.detach()
    y = torch.zeros(len(yPred),1).to(device)
    yPred = torch.where(yPred<=0.5, y ,yPred)
    auroc = metric(yPred, y)
  testRoc = metric.compute()
  metric.reset()
  return trainRoc, validRoc, testRoc, validLoss

def routine(x,train_edges,valid_edges,test_edges,params,trial,device):
  # init hyperparameters for the trial
  hidden_channels = params["hidden_channels"]
  num_layers = params["layers"]
  dropout = params["dropout"]
  learning_rate = params["learning_rate"]
  epochs = 100
  batch_size = params["batch_size"]
  output_layers = 1
  model = LinkPredictor(x.size(-1), hidden_channels, output_layers, num_layers, dropout).to(device)
  model.reset_parameters()
  optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), lr= learning_rate)
  best_loss = 1000
  x = x.to(device)
  for epoch in tqdm(range(1, 1 + epochs),position=0,leave=True):
    loss = train(model, x, train_edges, optimizer, batch_size,device)
    if loss < best_loss:
      best_loss = loss
    mlflow.log_metric("Loss",loss,step=epoch)
    trainRoc, validRoc, testRoc, validLoss = test(model, x, batch_size, train_edges, valid_edges,test_edges)
    mlflow.log_metric("Valid-Loss",validLoss,step=epoch)
    mlflow.log_metric("auroc-Train",trainRoc,step=epoch)
    mlflow.log_metric("auroc-Valid",validRoc,step=epoch)
    mlflow.log_metric("auroc-Test",testRoc,step=epoch)
    print("Epoch: ", epoch)
    print("Loss: ", loss)
    print("Valid-Loss: ", validLoss)
    print("auroc-Train: ", trainRoc)
    print("auroc-Valid: ", validRoc)
    print("auroc-Test: ", testRoc)
    # use valid score to decide to prune current trial
    trial.report(loss, epoch)
    if trial.should_prune():
      raise optuna.exceptions.TrialPruned()
  return best_loss

def objective(trial,x,full_edges,train_edges,valid_edges,test_edges,device):
  with mlflow.start_run():
      params = {
          "learning_rate" : trial.suggest_loguniform("learning_rate", low=10e-5, high=10e-1),
          "layers" : trial.suggest_int("layers", low=3, high=10),
          "hidden_channels" : trial.suggest_int("hidden_channels", low=122, high=512),
          "optimizer" : trial.suggest_categorical("optimizier",["Adam","RMSprop","SGD"]),
          "dropout" : trial.suggest_float("dropout", low=0.1, high=0.5, step=0.1),
          "batch_size" : trial.suggest_int("batch_size", low=512, high=2048),
          "embedding_dim" : trial.suggest_int("embedding_dim", low=64, high=256)
      }
      print("\nThis trials parameters:\n",params)
      mlflow.log_params(params)
      x = x.cpu()
      full_edges = full_edges.cpu()
      model = SAGE(x.size(-1), params["hidden_channels"], params["embedding_dim"], params["layers"], params["dropout"]).cpu()
      embeddings = model(x, full_edges)
      embeddings = embeddings.detach()
      result = routine(embeddings, train_edges, valid_edges, test_edges, params, trial, device)
      return result

data = PrimeKG(path = './data')

graph = createGraph(data)



transform = RandomLinkSplit(is_undirected=True)
split_edge = transform(graph)


# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
train_edges,valid_edges,test_edges = split_edge
neg_train_edges = negative_sampling(train_edges.edge_index)
neg_valid_edges = negative_sampling(valid_edges.edge_index)
neg_test_edges = negative_sampling(test_edges.edge_index)
train_edges = train_edges, neg_train_edges
valid_edges = valid_edges, neg_valid_edges
test_edges = test_edges, neg_test_edges
print(graph)
print(train_edges)
print(valid_edges)
print(test_edges)
# study = optuna.create_study(
#     study_name="PrimeKG LinkPrediction with SAGE encoder",
#     direction="minimize",
#     sampler=optuna.samplers.TPESampler(),
#     pruner = optuna.pruners.MedianPruner(n_startup_trials=0, n_warmup_steps=5)
#     )
# study.optimize(lambda trial: objective(trial,graph.x, graph.edge_index, train_edges,valid_edges,test_edges,device),n_trials=30)