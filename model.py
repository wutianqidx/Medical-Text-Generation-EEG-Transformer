import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


def loss_func(output, target, length_t):
    output = pack_padded_sequence(output, length_t, enforce_sorted=False).data
    target = pack_padded_sequence(target, length_t, enforce_sorted=False).data.view(-1)
    loss = F.cross_entropy(output, target)
    return loss


class EEGtoReport(nn.Module):
    def __init__(self, emb_dim=512, emb_dim_t=512, eeg_epoch_max=33, report_epoch_max=169, vocab_size=76):
        super().__init__()
        self.eeg_epoch_max = eeg_epoch_max
        self.report_epoch_max = report_epoch_max
        # input eeg embedding
        self.eeg_encoder = EEGEncoder(emb_dim=emb_dim)
        # target report embedding
        self.embedding = nn.Embedding(vocab_size, emb_dim_t, padding_idx=0)
        nn.init.xavier_uniform_(self.embedding.weight)
        # position encoder for src & target
        self.eeg_pos_encoder = PositionalEncoding(emb_dim=emb_dim, input_type='eeg', eeg_max=eeg_epoch_max)
        self.report_pos_encoder = PositionalEncoding(emb_dim=emb_dim_t, input_type='report')
        # transformer
        self.eeg_transformer = nn.Transformer(d_model=emb_dim, nhead=1, num_encoder_layers=1,
                                              num_decoder_layers=1, dim_feedforward=128, dropout=0.1)
        self.word_net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, vocab_size)
        )

    def forward(self, input, target, length, length_t, train=True):
        eeg_embedding = self.eeg_encoder(input)
        eeg_embedding, src_padding_mask = self.eeg_pos_encoder(eeg_embedding, length)

        if train:
            target, tgt_padding_mask = pad_target(target, self.report_epoch_max)
            report_embedding = self.embedding(target)
            report_embedding = self.report_pos_encoder(report_embedding, length_t)

            word_embedding = self.eeg_transformer(eeg_embedding, report_embedding,
                                                  src_key_padding_mask=src_padding_mask,
                                                  tgt_key_padding_mask=tgt_padding_mask,
                                                  memory_key_padding_mask=src_padding_mask)
            word_logits = self.word_net(word_embedding)
        else:
            """TO DO: During evaluation, output of t is input of t+1
            start-token is 1
            input at t+1 is argmax(word_logit @ t)
            stop when output @ t is 2:'<end>'
            Return:
            word_logits_argmax: words idx with size(T_real) just like target before padding
            """
            target, tgt_padding_mask = pad_target(target, self.report_epoch_max)
            report_embedding = self.embedding(target)
            report_embedding = self.report_pos_encoder(report_embedding, length_t)

            word_embedding = self.eeg_transformer(eeg_embedding, report_embedding,
                                                  src_key_padding_mask=src_padding_mask,
                                                  tgt_key_padding_mask=tgt_padding_mask,
                                                  memory_key_padding_mask=src_padding_mask)
            word_logits = self.word_net(word_embedding)

        return word_logits


def pad_target(report, max_len):
    n = len(report)
    tgt, tgt_mask = torch.zeros((n, max_len), dtype = int), torch.zeros((n, max_len), dtype = bool)
    for i, x in enumerate(report):
        tgt[i, :len(x)] = x
        tgt_mask[i, len(x):] = True
    return tgt, tgt_mask

class EEGEncoder(nn.Module):
    """BY TIANQI: encode eeg recording to embedding
    Check video at 3:17 for 1D covolutions https://www.youtube.com/watch?v=wNBaNhvL4pg
    1:15:00 https://www.youtube.com/watch?v=FrKWiRv254g
    1D convolutions
    Input:
    batch of eeg recording torch.tensor(*, 18, frequency*60)
    where '*' is sum of sample length in the batch
    E.g. torch.tensor([[[7, 0, 7, 7],
                        [0, 7, 0, 0]],
                       [[7, 0, 7, 0],
                        [0, 7, 0, 7]]]),
                        [[1, 0, 1, 1],
                        [0, 1, 0, 0]],
                       [[1, 0, 1, 0],
                        [0, 1, 0, 1]],
                        [[1, 0, 1, 0],
                         [0, 1, 0, 1]]])
                    )
            '*' = 2+3, 2 samples in this batch with length of 2 & 3 respectively
    Return:
    torch.tensor(*, 512)
    epoch of eeg recording (18, frequency*60) -> eeg embedding (512)
    """

    def __init__(self, emb_dim = 512):
        super().__init__()
        self.conv1 = nn.Conv1d(18, 32, 5)
        self.pool1 = nn.MaxPool1d(23)
        self.batch1 = nn.BatchNorm1d(32)
        #self.dropout1 = nn.Dropout(0.15)
        self.conv2 = nn.Conv1d(32, 64, 5)
        self.pool2 = nn.MaxPool1d(9)
        self.batch2 = nn.BatchNorm1d(64)
        #self.dropout2 = nn.Dropout(0.15)
        self.conv3 = nn.Conv1d(64, 256, 5)
        self.pool3 = nn.MaxPool1d(4)
        self.batch3 = nn.BatchNorm1d(256)
        #self.dropout3 = nn.Dropout(0.15)
        self.conv4 = nn.Conv1d(256, emb_dim, 5)
        self.pool4 = nn.MaxPool1d(13)
        self.batch4 = nn.BatchNorm1d(emb_dim)

    def forward(self, input):
        input = input.float()
        x = self.conv1(input)
        x = self.pool1(x)
        x = F.relu(self.batch1(x))
        #x = self.dropout1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(self.batch2(x))
        #x = self.dropout2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = F.relu(self.batch3(x))
        #x = self.dropout3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = F.relu(self.batch4(x))
        return x.squeeze()


class PositionalEncoding(nn.Module):
    "BY TIANQI"
    def __init__(self, emb_dim=512, dropout=0.1, pos_max_len=5000, input_type='eeg', eeg_max=20):   #input_type=['eeg','report']
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.emb_dim = emb_dim
        self.input_type = input_type
        self.eeg_max = eeg_max

        pe = torch.zeros(pos_max_len, emb_dim)
        position = torch.arange(0, pos_max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, length):
        """Return: torch.tensor((N,S,E)), where S is the sequence length, N is the batch size, E is the feature number
               E.g. eeg_embedding, len = torch.tensor(5, 512), (1,4) -> torch.tensor(4, 2, 512) padding with 0 and then do positional encoding
               torch.split may be helpful
        """
        if self.input_type == 'report':
            final_embedding = x + self.pe[:x.size(0), :]    # N,S,E
            return self.dropout(final_embedding.permute(1,0,2))
        elif self.input_type == 'eeg':
            position_indicator = sum([list(range(x)) for x in length], [])
            #print(x.size(), self.pe[position_indicator, :].squeeze().size())
            x = x + self.pe[position_indicator, :].squeeze()
            x = torch.split(x, length)
            batch_size = len(length)
            final_embedding = torch.zeros(batch_size, self.eeg_max, self.emb_dim)
            eeg_mask = torch.zeros((batch_size, self.eeg_max), dtype=bool)
            for i, k in enumerate(length):
                final_embedding[i, :k, :] = x[i]   # N,S,E
                eeg_mask[i, k:] = 1
            final_embedding = final_embedding.permute(1,0,2)        # N,S,E ->  S,N,E
            return self.dropout(final_embedding), eeg_mask
