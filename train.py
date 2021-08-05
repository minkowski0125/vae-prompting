import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel

from bpemb import BPEmb
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import args
from vae.data import TextEmbedding, BertEmbedding, CoLADataset
from vae.model import LstmVariationalAutoEncoder, LstmAutoEncoder

def train():
    args.device = torch.cuda.current_device()
    args.latent_dim = args.hidden_dim = args.input_dim = 768
    # bpemb = BPEmb(lang='en', dim=200)
    data = pd.read_csv('data/CoLA/train.tsv', sep='\t', header=None)[3]

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = CoLADataset(tokenizer, data, args)
    # dataset = CoLADataset(bpemb, data, args)

    pretrained_model = AutoModel.from_pretrained('bert-base-uncased').to(args.device)
    embedding = BertEmbedding(pretrained_model, args)

    # embedding = TextEmbedding(bpemb.vectors, args)

    data_loader = DataLoader(dataset, 
                                batch_size=args.batch_size,
                                shuffle=True)

    model = LstmAutoEncoder(args)
    # model = LstmVariationalAutoEncoder(args)
    model = model.to(args.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    # torch.optim.Adam(params)

    cnt = 0
    for k in range(args.epochs):
        print('epoch', k)
        for batch_ids, batch_mask, batch_lens in data_loader:
            batch_seqs = []
            for i, ids in enumerate(batch_ids):
                ids = ids.to(args.device)
                batch_seqs.append(embedding(ids[:batch_lens[i]]).to(args.device))
            cnt += 1
            recons, z = model(batch_seqs)
            # recons, z, mus, logvars = model(batch_seqs)

            batch_ids = batch_ids.to(args.device)
            loss_dic = model.loss_function(batch_ids, batch_seqs, batch_lens, recons)
            # loss_dic = model.loss_function(batch_ids, batch_lens, recons, mus, logvars, args.kld_weight)
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