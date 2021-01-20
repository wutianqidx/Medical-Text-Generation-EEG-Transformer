import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pandas as pd

class EEGDataset(Dataset):
    def __init__(self, pkl_file):
        """BY PEIYAO: Load and prepare data for NN
        Args:
        pkl_file: [(eeg, report), ...], word_bag, frequency = pd.read_pickle(pkl_file)
            eeg: np.array(18, SampleLength)
            report: ['IMPRESSION', 'DESCRIPTION OF THE RECORD']
            word_bag: {word: frequency}
            frequency: n Hz which means n*60 samples/min
        """
        self.THRESHOLD = 2
        self.data, self.word_bag, self.freq = pd.read_pickle(pkl_file)
        self.eeg_epoch_len = self.freq * 60
        self.textual_ids, self.max_len_t, self.ixtoword, self.wordtoix, self.max_len = self.build_dict()
        self.vocab_size = len(self.ixtoword) + 2 # 1 for start-token, 0 for padding

    def build_dict(self):
        ixtoword = {2:'<end>', 3:'<sep>', 4:'<punc>', 5:'<unk>'} # 1 for start-token, 0 for padding
        wordtoix = {'<end>':2, '<sep>':3, '<punc>':4, '<unk>':5}
        textual_ids = []
        max_len_t = 0
        max_len = 0

        idx = max(ixtoword.keys()) + 1
        for word, freq in self.word_bag.items():
            if word not in wordtoix:
                if freq >= self.THRESHOLD:
                    ixtoword[idx] = word
                    wordtoix[word] = idx
                    idx += 1

        for eeg, report in self.data:
            length_t = len(report)
            if length_t > max_len_t:
                max_len_t = length_t
            length = eeg.shape[1]//self.eeg_epoch_len
            if length > max_len:
                max_len = length
            temp = []
            for word in report:
                if word in wordtoix:
                    temp.append(wordtoix[word])
                else:
                    temp.append(wordtoix['<unk>'])
            textual_ids.append(temp)
        return textual_ids, max_len_t, ixtoword, wordtoix, max_len

    def get_text(self, idx):
        text_i = torch.tensor([1]+self.textual_ids[idx][:-1])
        text = F.pad(torch.tensor(self.textual_ids[idx]), (0, self.max_len_t-len(text_i))).view(-1, 1)
        return text_i, text

    def get_eeg(self, idx):
        eeg, report = self.data[idx]
        shape1 = eeg.shape[0]
        shape2 = self.freq * 60
        shape0 = int(eeg.shape[1]/self.freq/60)
        cutoff = shape2 * shape0
        new_eeg = eeg[:, :cutoff]
        reshape_eeg = new_eeg.reshape(shape0, shape1, shape2)
        return torch.tensor(reshape_eeg)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """BY PEIYAO: a) represent eeg from np.array(18, SampleLength) to torch.tensor(*, 18, frequency*60)
        SampleLength is varible but frequency*60 is fixed
        np.array([[1,0,1,0],[0,1,0,1]]) ->
        torch.tensor([ [[1,0],
                        [0,1]],
                        [[1,0],
                        [0,1]] ])
        b) represent report as one hot vector according to word bag
            ['2 0 2', '0 2 0 2 0'] -> torch.tensor([[0, 1, 0, ...], ...]) seperated by
            special token [SEP] between 'IMPRESSION' and 'DESCRIPTION OF THE RECORD'
        Args:
        idx: idx'th (eeg, report) from self.data
        Return:
        torch.tensor(*, 18, frequency*60), torch.tensor([[0, 1, 0, ...], ...]), len(torch.tensor(*, 18, frequency*60))
        """
        eeg = self.get_eeg(idx)
        report_i, report = self.get_text(idx)
        return eeg, report_i, report, len(eeg), len(report_i)

def collate_wrapper(batch):
    input, target_i, target, length, length_t = list(zip(*batch))
    input = torch.cat(input, 0)
    target = torch.stack(target, 1)
    return input, target_i, target, length, length_t

class CollateWrapper:
    def __init__(self, max_len=None, max_len_t=None):
        self.max_len = max_len
        self.max_len_t = max_len_t

    def __call__(self, batch):
        input, target_i, target, length, length_t = batch[0]
        len_x = min(length, self.max_len)
        len_x_t = min(length_t, self.max_len_t)

        input = input[:len_x]
        target_i = (target_i[:len_x_t],)
        target = target[:len_x_t].unsqueeze(1)
        length = (len_x,)
        length_t = (len_x_t,)
        return input, target_i, target, length, length_t
