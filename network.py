import torch
import argparse
import torch.nn.functional as F
parser = argparse.ArgumentParser()
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GraphConv,GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import degree

parser.add_argument('--seed', type=int, default=123,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=64,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.9,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.8,
                    help='dropout ratio')

args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:5'
# Define the neural network architecture
class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.convf = GraphConv(self.num_features, self.nhid)#num_layers=3
        self.conv1 = GraphConv(self.num_features, self.nhid)
        self.conv2 = GraphConv(self.nhid, self.nhid)
        self.conv3 = GraphConv(self.nhid, self.nhid)
        self.lin1 = torch.nn.Linear(self.nhid*2*3, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if x is None:
            x = torch.ones(data.num_nodes, 1, device=next(self.convf.parameters()).device)
            x = x.to(args.device)
        x = F.relu(self.conv1(x, edge_index))
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.conv2(x, edge_index))
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.conv3(x, edge_index))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)
        return x