import torch
from torch import nn
import torch.nn.functional as F
from core_ave import COREave


class HierarchicalNet(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.hidden_size = config['embedding_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']

        self.segment_length = config['segment_length']

        self.lower_rnn_type = config['lower_rnn_type'] if 'lower_rnn_type' in config else 'GRU'
        self.lower_rnn_num_layers = config['lower_rnn_num_layers'] if 'lower_rnn_num_layers' in config else 1

        self.upper_rnn_type = config['upper_rnn_type'] if 'upper_rnn_type' in config else 'GRU'
        self.upper_rnn_num_layers = config['upper_rnn_num_layers'] if 'upper_rnn_num_layers' in config else 1

        if self.lower_rnn_type.upper() == 'GRU':
            self.lower_rnn = nn.GRU(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.lower_rnn_num_layers,
                batch_first=True
            )
        else:
            raise NotImplementedError("Only GRU is supported for lower RNN.")

        if self.upper_rnn_type.upper() == 'GRU':
            self.upper_rnn = nn.GRU(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.upper_rnn_num_layers,
                batch_first=True
            )
        else:
            raise NotImplementedError("Only GRU is supported for upper RNN.")

        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        
        self.lower_fn = nn.Linear(self.hidden_size, 1)
        self.upper_fn = nn.Linear(self.hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.lower_fn.weight)
        if self.lower_fn.bias is not None:
            nn.init.zeros_(self.lower_fn.bias)
        nn.init.xavier_uniform_(self.upper_fn.weight)
        if self.upper_fn.bias is not None:
            nn.init.zeros_(self.upper_fn.bias)

    def forward(self, item_seq, x):

        B, L, H = x.size()

        segment_len = self.segment_length
        num_segments = (L + segment_len - 1) // segment_len  

        if L % segment_len != 0:
            pad_len = segment_len - (L % segment_len)
            pad_item_seq = F.pad(item_seq, (0, pad_len), value=0)
            pad_x = F.pad(x, (0,0,0,pad_len,0,0))
        else:
            pad_item_seq = item_seq
            pad_x = x

        pad_x = pad_x.view(B, num_segments, segment_len, H)
        pad_item_seq = pad_item_seq.view(B, num_segments, segment_len)

        segment_embs = []
        for seg_idx in range(num_segments):
            seg_item_seq = pad_item_seq[:, seg_idx, :]  
            seg_x = pad_x[:, seg_idx, :, :]             

            mask = seg_item_seq.gt(0)
            seg_output, _ = self.lower_rnn(seg_x)  
            seg_output = self.dropout(seg_output)

            alpha = self.lower_fn(seg_output).double() 
            alpha = torch.where(mask.unsqueeze(-1), alpha, torch.tensor(-9e15, dtype=torch.double, device=alpha.device))
            alpha = torch.softmax(alpha, dim=1).float() 
            
            seg_vec = torch.sum(alpha * seg_x, dim=1) 
            segment_embs.append(seg_vec)

        segment_embs = torch.stack(segment_embs, dim=1)  
        upper_output, _ = self.upper_rnn(segment_embs)  
        upper_output = self.dropout(upper_output)

        segment_mask = (pad_item_seq.sum(dim=-1) > 0)  
        alpha_upper = self.upper_fn(upper_output).double() 
        alpha_upper = torch.where(segment_mask.unsqueeze(-1), alpha_upper, torch.tensor(-9e15, dtype=torch.double, device=alpha_upper.device))
        alpha_upper = torch.softmax(alpha_upper, dim=1).float() 

        session_vec = torch.sum(alpha_upper * segment_embs, dim=1) 
        return session_vec


class COREhierarchical(COREave):
    def __init__(self, config, dataset):
        super(COREhierarchical, self).__init__(config, dataset)
        self.net = HierarchicalNet(config, dataset)

    def forward(self, item_seq):
        x = self.item_embedding(item_seq)
        x = self.sess_dropout(x)
        seq_output = self.net(item_seq, x)
        seq_output = F.normalize(seq_output, dim=-1)
        return seq_output
