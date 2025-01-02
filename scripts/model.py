import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=2):
        super(GATLayer, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=heads, concat=True)
        self.conv2 = GATConv(out_channels * heads, out_channels, heads=heads, concat=True)

    def forward(self, x, edge_index):
        x, attn_weights1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = x.relu()
        x, attn_weights2 = self.conv2(x, edge_index, return_attention_weights=True)
        return x, attn_weights1, attn_weights2

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        attn_weights = torch.tanh(self.attention(lstm_out))
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context, attn_weights

class GAT_LSTM_Attn_Model(nn.Module):
    def __init__(self, sequence_length, in_channels, hidden_channels, out_channels, num_classes, num_gat_layers=2, num_lstm_layers=2):
        super(GAT_LSTM_Attn_Model, self).__init__()
        self.sequence_length = sequence_length
        self.gat_layers = nn.ModuleList([GATLayer(in_channels if i == 0 else hidden_channels * 2, hidden_channels) for i in range(num_gat_layers)])
        self.lstm = nn.LSTM(hidden_channels * 17 * 2, out_channels, num_layers=num_lstm_layers, batch_first=True)
        self.attention = Attention(out_channels)
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x, edge_index):
        batch_size, seq_len, num_nodes, in_channels = x.size()
        edge_index = edge_index.to(x.device)
        
        gat_out_list = []
        for i in range(batch_size):
            for j in range(seq_len):
                x_i_j = x[i, j, :, :]
                for gat_layer in self.gat_layers:
                    x_i_j, _, _ = gat_layer(x_i_j, edge_index)
                gat_out_list.append(x_i_j)

        gat_out = torch.stack(gat_out_list).view(batch_size, seq_len, num_nodes, -1)
        lstm_in = gat_out.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(lstm_in)
        context, attn_weights = self.attention(lstm_out)
        out = self.fc(context)
        return out
