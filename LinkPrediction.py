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
from torch.nn import BCEWithLogitsLoss, BCELoss
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
            x = F.tanh(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, trainGraph, optimizer, batch_size, device):
    predictor.train()
    model.train()
    
    total_loss = total_pos_loss = total_neg_loss = total_examples = 0
    for perm in DataLoader(range(len(trainGraph.edge_index[0])), batch_size, shuffle=True):
        optimizer.zero_grad()
        emb = model(trainGraph.x.to(device), trainGraph.edge_index.to(device))
        edge = trainGraph.edge_index.t()[perm].t().to(device)
        pos_out = predictor(emb[edge[0]],emb[edge[1]])
        edge = negative_sampling(edge).to(device)
        neg_out = predictor(emb[edge[0]], emb[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()


        total_pos_loss += pos_loss.item() * pos_out.size(0)
        total_neg_loss += neg_loss.item() * neg_out.size(0)
        total_loss += loss.item() * pos_out.size(0)
        total_examples += pos_out.size(0)

    loss = total_loss / total_examples
    pos_loss = total_pos_loss / total_examples
    neg_loss = total_neg_loss / total_examples
    return loss, pos_loss, neg_loss

def test(model, predictor, trainGraph, validGraph):
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
    emb = model(trainGraph.x.to(device), trainGraph.edge_index.to(device))
    pos_out = predictor(emb[trainGraph.edge_index[0]],emb[trainGraph.edge_index[1]])
    neg_out = predictor(emb[trainGraph.neg_edge_index[0]], emb[trainGraph.neg_edge_index[1]])
    logits = torch.cat([pos_out,neg_out])
    labels = torch.cat([torch.ones(len(pos_out),1),torch.zeros(len(neg_out),1)])
    trainMetric.update(logits,labels.to(device))

    #valid set
    emb = model(validGraph.x.to(device), validGraph.edge_index.to(device))
    pos_out = predictor(emb[validGraph.edge_index[0]],emb[validGraph.edge_index[1]])
    neg_out = predictor(emb[validGraph.neg_edge_index[0]], emb[validGraph.neg_edge_index[1]])
    logits = torch.cat([pos_out,neg_out])
    labels = torch.cat([torch.ones(len(pos_out),1),torch.zeros(len(neg_out),1)])
    validMetric.update(logits,labels.to(device))
    pos_loss = -torch.log(pos_out + 1e-15).mean()
    neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
    loss = pos_loss + neg_loss

    trainOutput = trainMetric.compute()
    trainMetric.reset()  
    validOutput = validMetric.compute()
    validMetric.reset()
    return trainOutput, validOutput, loss, pos_loss, neg_loss


def routine(trainGraph, validGraph, params, trial, device):
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
  model = SAGE(params["embedding_dim"], params["hidden_channels"], params["embedding_dim"], params["layers"], params["dropout"]).to(device)
  model.reset_parameters()
  optimizer = getattr(torch.optim, params["optimizer"])(list(model.parameters()) + list(predictor.parameters()), lr = learning_rate)
  
  for epoch in tqdm(range(1, 1 + epochs),position=0,leave=True):
    loss, pos_loss, neg_loss = train(model, predictor, trainGraph, optimizer, batch_size, device)
    mlflow.log_metric("Loss",loss,step=epoch)
    mlflow.log_metric("pos-Loss",pos_loss,step=epoch)
    mlflow.log_metric("neg-Loss",neg_loss,step=epoch)
    trainOutput, validOutput, validLoss, posValidLoss, negValidLoss = test(model, predictor, trainGraph, validGraph)
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
    # use valid Auroc score to decide to prune current trial
    trial.report(validOutput["valid_BinaryAUROC"], epoch)
    if trial.should_prune():
      raise optuna.exceptions.TrialPruned()
  return validOutput["valid_BinaryAUROC"]

def objective(trial, trainGraph, validGraph, device):
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
      
      
      model = SAGE(trainGraph.x.size(-1), params["hidden_channels"], params["embedding_dim"], params["layers"], params["dropout"]).to(device)
      x = model(trainGraph.x.to(device), trainGraph.edge_index.to(device))
      trainGraph.x = x.detach()

      x = model(validGraph.x.to(device), validGraph.edge_index.to(device))
      validGraph.x = x.detach()
     
      result = routine(trainGraph, validGraph, params, trial, device)
      return result
      

data = PrimeKG(path = './data')

graph = createGraph(data)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
print("Using device: ", device)
train_size = int(0.2 * len(torch.unique(graph.edge_index[0])))
valid_size = int(0.05 * len(torch.unique(graph.edge_index[0])))
test_size = len(torch.unique(graph.edge_index[0])) - train_size - valid_size
train_dataset, valid_dataset, _ = torch.utils.data.random_split(torch.unique(graph.edge_index[0]), [train_size, valid_size, test_size])
trainGraph = graph.subgraph(torch.tensor(train_dataset.indices))
validGraph = graph.subgraph(torch.tensor(valid_dataset.indices))
trainGraph.neg_edge_index = negative_sampling(trainGraph.edge_index)
validGraph.neg_edge_index = negative_sampling(validGraph.edge_index)
study = optuna.create_study(
    study_name="PrimeKG LinkPrediction with subset of PrimeKG ",
    direction="maximize",
    sampler=optuna.samplers.TPESampler(),
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
    )
study.optimize(lambda trial: objective(trial, trainGraph, validGraph, device), n_trials=30)