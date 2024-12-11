import torch
from torch import nn
import torch.nn.functional as F
from core_ave import COREave

class RNNNet(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.hidden_size = config['embedding_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        
        if 'rnn_num_layers' in config:
            self.rnn_num_layers = config['rnn_num_layers']
        else:
            self.rnn_num_layers = 1
        
        if 'rnn_type' in config:
            self.rnn_type = config['rnn_type']
        else:
            self.rnn_type = 'GRU'

        if self.rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.rnn_num_layers,
                batch_first=True
            )
        else:
            raise NotImplementedError("Only GRU is supported currently.")

        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.fn = nn.Linear(self.hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fn.weight)
        if self.fn.bias is not None:
            nn.init.zeros_(self.fn.bias)

    def forward(self, item_seq, x):
        mask = item_seq.gt(0)  

        output, _ = self.rnn(x)  
        output = self.dropout(output)

        alpha = self.fn(output).double()  
        alpha = torch.where(mask.unsqueeze(-1), alpha, torch.tensor(-9e15, dtype=torch.double, device=alpha.device))
        alpha = torch.softmax(alpha, dim=1).float()
        return alpha


class CORErnn(COREave):
    def __init__(self, config, dataset):
        super(CORErnn, self).__init__(config, dataset)
        self.net = RNNNet(config, dataset)

    def forward(self, item_seq):
        x = self.item_embedding(item_seq)
        x = self.sess_dropout(x)
        # RCE + RNN
        alpha = self.net(item_seq, x)
        seq_output = torch.sum(alpha * x, dim=1)
        seq_output = F.normalize(seq_output, dim=-1)
        return seq_output
