from tdc.resource import PrimeKG
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import BCELoss
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
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryF1Score, BinaryAccuracy, BinaryPrecision, BinaryRecall
from sklearn.model_selection import train_test_split
from torch_geometric.loader import NeighborLoader
from torch.nn import BCEWithLogitsLoss
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
            x = F.tanh(x)
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
            x = F.tanh(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, optimizer, loader, device):
    predictor.train()
    model.train()
    crit = BCEWithLogitsLoss()
    total_loss = total_pos_loss = total_neg_loss = total_examples = i = 0
    for subgraph in loader:
        optimizer.zero_grad()
        
        emb = model(subgraph.x.to(device), subgraph.edge_index.to(device))
        pos_out = predictor(emb[subgraph.edge_index[0]],emb[subgraph.edge_index[1]])
        edge = negative_sampling(subgraph.edge_index)
        neg_out = predictor(emb[edge[0]], emb[edge[1]])
        logits = torch.cat([pos_out,neg_out])
        labels = torch.cat([torch.ones(len(pos_out),1),torch.zeros(len(neg_out),1)])
        pos_loss = crit(pos_out, torch.ones(len(pos_out),1).to(device))
        neg_loss = crit(neg_out, torch.zeros(len(neg_out),1).to(device))
        loss = crit(logits,labels.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        total_pos_loss += pos_loss.item() * pos_out.size(0)
        total_neg_loss += neg_loss.item() * neg_out.size(0)
        total_loss += loss.item() * logits.size(0)
        total_examples += logits.size(0)

    loss = total_loss / total_examples
    pos_loss = total_pos_loss / total_examples
    neg_loss = total_neg_loss / total_examples
    return loss, pos_loss, neg_loss

def test(model, predictor, trainLoader, validLoader):
  with torch.no_grad():
    model.eval()
    predictor.eval()

    metrics = MetricCollection([
    BinaryAUROC(thresholds=None),
    BinaryAccuracy(thresholds=None),
    BinaryPrecision(thresholds=None),
    BinaryRecall(thresholds=None),
    BinaryF1Score(thresholds=None)
    ])
    validMetric = metrics.clone(prefix="valid_").to(device)
    trainMetric = metrics.clone(prefix="train_").to(device)

    #train set

    for tmp in trainLoader:

        emb = model(tmp.x.to(device), tmp.edge_index.to(device))
        pos_out = predictor(emb[tmp.edge_index[0]],emb[tmp.edge_index[1]])
        edge = negative_sampling(tmp.edge_index)
        neg_out = predictor(emb[edge[0]], emb[edge[1]])
        logits = torch.cat([pos_out,neg_out])
        labels = torch.cat([torch.ones(len(pos_out),1),torch.zeros(len(neg_out),1)])
        trainMetric.update(logits, labels.to(device))


    #valid set
    total_loss = total_pos_loss = total_neg_loss = total_examples = 0
    crit = BCEWithLogitsLoss()
    for tmp in validLoader:

        emb = model(tmp.x.to(device), tmp.edge_index.to(device))
        pos_out = predictor(emb[tmp.edge_index[0]],emb[tmp.edge_index[1]])
        edge = negative_sampling(tmp.edge_index)
        neg_out = predictor(emb[edge[0]], emb[edge[1]])
        logits = torch.cat([pos_out,neg_out])
        labels = torch.cat([torch.ones(len(pos_out),1),torch.zeros(len(neg_out),1)])
        pos_loss = crit(pos_out, torch.ones(len(pos_out),1).to(device))
        neg_loss = crit(neg_out, torch.zeros(len(neg_out),1).to(device))
        loss = crit(logits,labels.to(device))

        total_pos_loss += pos_loss.item() * pos_out.size(0)
        total_neg_loss += neg_loss.item() * neg_out.size(0)
        total_loss += loss.item() * logits.size(0)
        total_examples += logits.size(0)
        validMetric.update(logits, labels.to(device))


    loss = total_loss / total_examples
    pos_loss = total_pos_loss / total_examples
    neg_loss = total_neg_loss / total_examples

    trainOutput = trainMetric.compute()
    trainMetric.reset()  
    validOutput = validMetric.compute()
    validMetric.reset()
    return trainOutput, validOutput, loss, pos_loss, neg_loss


def routine(train_edges, valid_edges, params, trial, device):
  # init hyperparameters for the trial
  hidden_channels = params["hidden_channels"]
  num_layers = params["layers"]
  dropout = params["dropout"]
  learning_rate = params["learning_rate"]
  epochs = 100
  batch_size = params["batch_size"]
  output_layers = 1
  predictor = LinkPredictor(params["embedding_dim"], hidden_channels, output_layers, num_layers, dropout).to(device)
  predictor.reset_parameters()
  model = SAGE(256, params["hidden_channels"], params["embedding_dim"], params["layers"], params["dropout"]).to(device)
  model.reset_parameters()
  optimizer = getattr(torch.optim, params["optimizer"])(list(model.parameters()) + list(predictor.parameters()), lr = learning_rate)
  
  trainLoader = NeighborLoader(train_edges[0],num_neighbors=[30] * 2,batch_size=batch_size)
  validLoader = NeighborLoader(valid_edges[0],num_neighbors=[30] * 2,batch_size=batch_size)
  for epoch in tqdm(range(1, 1 + epochs),position=0,leave=True):
    loss, pos_loss, neg_loss = train(model, predictor, optimizer, trainLoader, device)
    mlflow.log_metric("Loss",loss,step=epoch)
    mlflow.log_metric("pos-Loss",pos_loss,step=epoch)
    mlflow.log_metric("neg-Loss",neg_loss,step=epoch)
    trainOutput, validOutput, validLoss, posValidLoss, negValidLoss = test(model, predictor,  trainLoader, validLoader)
    mlflow.log_metric("Valid-Loss",validLoss,step=epoch)
    mlflow.log_metric("pos-Valid-Loss",posValidLoss,step=epoch)
    mlflow.log_metric("neg-Valid-Loss",negValidLoss,step=epoch)
    mlflow.log_metric("auroc-Valid",validOutput["valid_BinaryAUROC"],step=epoch)
    mlflow.log_metric("auroc-Train",trainOutput["train_BinaryAUROC"],step=epoch)
    mlflow.log_metric("precision-Valid",validOutput["valid_BinaryPrecision"],step=epoch)
    mlflow.log_metric("precision-Train",trainOutput["train_BinaryPrecision"],step=epoch)
    mlflow.log_metric("recall-Valid",validOutput["valid_BinaryRecall"],step=epoch)
    mlflow.log_metric("recall-Train",trainOutput["train_BinaryRecall"],step=epoch)
    mlflow.log_metric("f1-Valid",validOutput["valid_BinaryF1Score"],step=epoch)
    mlflow.log_metric("f1-Train",trainOutput["train_BinaryF1Score"],step=epoch)
    mlflow.log_metric("accuracy-Valid",validOutput["valid_BinaryAccuracy"],step=epoch)
    mlflow.log_metric("accuracy-Train",trainOutput["train_BinaryAccuracy"],step=epoch)
    print("Epoch: ", epoch)
    print("Loss: ", loss)
    print("pos-Loss: ", pos_loss)
    print("neg-Loss: ", neg_loss)
    print("Valid-Loss: ", validLoss)
    print("pos-Valid-Loss: ", posValidLoss)
    print("neg-Valid-Loss: ", negValidLoss)
    print("auroc-Valid: ", validOutput["valid_BinaryAUROC"])
    print("auroc-Train: ", trainOutput["train_BinaryAUROC"])
    print("accuracy-Valid: ", validOutput["valid_BinaryAccuracy"])
    print("accuracy-Train: ", trainOutput["train_BinaryAccuracy"])
    print("precision-Valid: ", validOutput["valid_BinaryPrecision"])
    print("precision-Train: ", trainOutput["train_BinaryPrecision"])
    print("recall-Valid: ", validOutput["valid_BinaryRecall"])
    print("recall-Train: ", trainOutput["train_BinaryRecall"])
    print("f1-Valid: ", validOutput["valid_BinaryF1Score"])
    print("f1-Train: ", trainOutput["train_BinaryF1Score"])
    # use valid score to decide to prune current trial
    trial.report(validLoss, epoch)
    if trial.should_prune():
      raise optuna.exceptions.TrialPruned()
  return validLoss

def objective(trial, train_edges, valid_edges, device):
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

      result = routine(train_edges, valid_edges, params, trial, device)
      return result

data = PrimeKG(path = './data')

graph = createGraph(data)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
transform = RandomLinkSplit(is_undirected=True)
split_edge = transform(graph)
train_edges,valid_edges,_ = split_edge
neg_train_edges = negative_sampling(train_edges.edge_label_index)
neg_valid_edges = negative_sampling(valid_edges.edge_label_index)
train_edges = train_edges, neg_train_edges
valid_edges = valid_edges, neg_valid_edges
study = optuna.create_study(
    study_name="PrimeKG LinkPrediction with SAGE encoder training, random valid edges and log loss ",
    direction="minimize",
    sampler=optuna.samplers.TPESampler(),
    pruner = optuna.pruners.MedianPruner(n_startup_trials=0, n_warmup_steps=5)
    )
study.optimize(lambda trial: objective(trial, train_edges, valid_edges, device), n_trials=30)