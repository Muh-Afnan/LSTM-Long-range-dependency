import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class CharTokenizer():
    def __init__(self):
        self.vocab = None
        self.char2idx_map = None
        self.idx2char_map = None

    def preprocess(self,text:str):
        text = text.lower()
        chars = text
        if len(text)>1:
            chars = list(text)
        return chars

    def char2idx(self,text:str):
        chars = self.preprocess(text)
        self.vocab = sorted(set(chars))
        self.char2idx_map  = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char_map = {idx: char for char, idx in self.char2idx_map.items()}
        return self.vocab, self.char2idx_map, self.idx2char_map
    
    def idx2char(self,idx:int):
        return self.idx2char_map[idx]

    def seq_generator(self,text:str,window:int):
        text_pro = self.preprocess(text)
        vocab, char2idx_map, idx2char_map = self.char2idx(text)
        seqs = []
        for idx in range(len(text_pro) - window):
            input = text_pro[idx:idx+window]
            output = text_pro[idx+1:idx+window+1]
            input_idx = [char2idx_map[char] for char in input]
            output_idx = [char2idx_map[char] for char in output]
            seq = (input_idx,output_idx)
            seqs.append(seq)
        return seqs



class LSTMFromScratch(nn.Module):
    def __init__(self,vocab_size:int, embedding_dim:int, hidden_size:int,):
        super().__init__()
        if min(vocab_size, embedding_dim,hidden_size)<=0:
            raise ValueError("All sizes must be positive.")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.W_xh = nn.Parameter(torch.empty(embedding_dim,hidden_size))
        self.W_hh = nn.Parameter(torch.empty(hidden_size,hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.W_hy = nn.Parameter(torch.empty(hidden_size,vocab_size))
        self.b_y = nn.Parameter(torch.zeros(vocab_size))
        nn.init.xavier_uniform_(self.W_xh)
        nn.init.orthogonal_(self.W_hh)
        nn.init.xavier_uniform_(self.W_hy)