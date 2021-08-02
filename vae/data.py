from bpemb import BPEmb
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class TextEmbedding(nn.Module):
    def __init__(self, vectors, config):
        super().__init__()
        self.vecs = nn.Embedding.from_pretrained(torch.tensor(vectors))

    def forward(self, batch_seqs):
        return self.vecs(batch_seqs)
        # ret = []
        # for seq in batch_seqs:
        #     print(seq)
        #     ret.append(self.vecs(seq))
        # return ret

class CoLADataset(Dataset):
    def __init__(self, encoder, data, config):
        super().__init__()
        self.data = []
        self.mask = []
        self.lens = []
        # self.seq_len = config.seq_len
        self.seq_len = 64
        for item in data:
            ids = encoder.encode_ids(item)
            # print(ids)
            data_line, mask_line = torch.zeros(1, self.seq_len, dtype=torch.long), torch.zeros(1, self.seq_len, dtype=torch.long)
            data_line -= 1
            data_line[0, :len(ids)], mask_line[0, :len(ids)] = torch.tensor(ids), torch.ones(len(ids))
            self.data.append(data_line)
            self.mask.append(mask_line)
            self.lens.append(len(ids))
        self.data, self.mask = torch.cat(self.data), torch.cat(self.mask)
        # print(self.data.shape)

    def __len__(self):
        return len(self.lens)

    def __getitem__(self, idx):
        return self.data[idx], self.mask[idx], self.lens[idx]

if __name__ == '__main__':
    bpemb = BPEmb(lang='en', dim=200)
    embedding = TextEmbedding(bpemb.vectors, None)
    data = pd.read_csv('data/CoLA/train.tsv', sep='\t', header=None)[3]
    dataset = CoLADataset(bpemb, data, None)
    loader = DataLoader(dataset,
                            shuffle=True,
                            batch_size=32)
    for item in loader:
        print(item[0][0], item[1][0])
        break
