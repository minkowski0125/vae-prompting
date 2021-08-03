import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from bpemb import BPEmb
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import args
from vae.data import TextEmbedding, CoLADataset
from vae.model import LstmVariationalAutoEncoder

def train():
    args.device = torch.cuda.current_device()
    bpemb = BPEmb(lang='en', dim=200)
    data = pd.read_csv('data/CoLA/train.tsv', sep='\t', header=None)[3]
    dataset = CoLADataset(bpemb, data, args)
    embedding = TextEmbedding(bpemb.vectors, args)

    data_loader = DataLoader(dataset, 
                                batch_size=args.batch_size,
                                shuffle=True)

    model = LstmVariationalAutoEncoder(args)
    model = model.to(args.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters())
    # torch.optim.Adam(params)

    cnt = 0
    for k in range(args.epochs):
        print('epoch', k)
        for batch_ids, batch_mask, batch_lens in data_loader:
            batch_seqs = []
            for i, ids in enumerate(batch_ids):
                batch_seqs.append(embedding(ids[:batch_lens[i]]).to(args.device))
            cnt += 1
            recons, z, mus, logvars = model(batch_seqs)

            batch_ids = batch_ids.to(args.device)
            loss_dic = model.loss_function(batch_ids, batch_lens, recons, mus, logvars, args.kld_weight)
            loss = loss_dic['loss']
            print(loss.item(), end='\r')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if cnt % 100 == 0:
                print('iter', cnt, loss.item())

if __name__ == '__main__':
    # torch.cuda.set_device(0)
    train()