from tqdm import tqdm
import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data
import torch.nn.functional as F
from torch.utils.data import DataLoader
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

def encoder(graph):
    print("Computing embeddings...")
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    def save_embedding(model):
        torch.save(model.embedding.weight.data.cpu(), 'embedding.pt')

    embedding_dim = 128
    walk_length = 40
    context_size = 20
    walks_per_node = 10
    batch_size = 256
    epochs = 2
    lr = 0.1
    log_steps = 10
    model = Node2Vec(graph.edge_index, embedding_dim, walk_length,
                    context_size, walks_per_node,
                    sparse=True).to(device)

    loader = model.loader(batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=lr)

    model.train()
    for epoch in tqdm(range(1, epochs + 1)):
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

            if (i + 1) % log_steps == 0:
                print(f'Epoch: {epoch:02d}, Step: {i+1:03d}/{len(loader)}, '
                    f'Loss: {loss:.4f}')

            if (i + 1) % 100 == 0:  # Save model every 100 steps.
                save_embedding(model)
    save_embedding(model)
    print("Embeddings done!")
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


@torch.no_grad()
def test(predictor, x, split_edge, evaluator, batch_size):
    predictor.eval()
    # split edge contains all the edge splits as a triple
    train_edge,valid_edge,test_edge = split_edge
    pos_train_edge = train_edge.pos_edge_label_index.t().to(x.device)
    neg_train_edge = train_edge.neg_edge_label_index.t().to(x.device)
    pos_valid_edge = valid_edge.pos_edge_label_index.t().to(x.device)
    neg_valid_edge = valid_edge.neg_edge_label_index.t().to(x.device)
    pos_test_edge = test_edge.pos_edge_label_index.t().to(x.device)
    neg_test_edge = test_edge.pos_edge_label_index.t().to(x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)