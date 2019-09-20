import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

import scipy
from scipy import sparse

import torch_geometric.nn as geo_nn

from torch.autograd import gradcheck


def get_batch_adj(adj, tmp_batch_size, args):
    if args.num_node_samples == 0:
        adjs_list = [(adj + i*args.num_genes) for i in range(tmp_batch_size)]
    else:
        adjs_list = [(adj + i*args.num_node_samples) for i in range(tmp_batch_size)]
    adjs_list = np.hstack(adjs_list)
    return adjs_list

def drug2gex_attn_applied(drug_feat, gex_feat, self, args, training):

    #   drug input & gex input requires same dimension
    if args.attn_type == 'multi':
        drug_proj = F.dropout(
                        torch.tanh(
                        self.query_proj(drug_feat)),
                        training = training)
        #   bs x 200
        gex_proj = F.dropout(
                        torch.tanh(
                        self.key_proj(gex_feat)),
                        training = training)
        #   bs x num genes x 200

        #   Multiplicative Attention
        drug_proj  = drug_proj.unsqueeze(1)
        gex_proj = gex_proj.transpose(2,1)
        attn = torch.bmm(drug_proj, gex_proj).squeeze(1) # bs x num_genes 
        attn = F.softmax(attn, dim = -1)

    elif args.attn_type == 'add':
        drug_proj = F.dropout(
                        self.query_proj(drug_feat),
                        training = training)
        #   bs x 200
        gex_proj = F.dropout(
                        self.key_proj(gex_feat),
                        training = training)
        #   bs x num genes x 200

        #   Additive Attention
        drug_proj = drug_proj.unsqueeze(1)
        attn = self.attn_proj(torch.tanh(drug_proj + gex_proj)).squeeze(-1)
        attn = F.softmax(attn, dim = -1)

    gex_attned = torch.mul(attn.unsqueeze(-1), gex_feat) # bs x num_genes x 200
    
    return attn, gex_attned

class GEX_PPI_GAT_cat4_MLP(nn.Module):  
    def __init__(self, ppi_adj, g2v_embedding, args):
        super(GEX_PPI_GAT_cat4_MLP, self).__init__()


        print('num genes : '+str(args.num_genes))
        self.bn1 = nn.BatchNorm1d(args.ecfp_nBits)
        self.bn2 = nn.BatchNorm1d(args.num_genes)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn4 = nn.BatchNorm1d(1)

        self.drug_mlp1 = nn.Linear(args.resimnet_dim, 
                    int(args.resimnet_dim*2/3 + args.drug_embed_dim/3), bias = True)
        self.drug_mlp2 = nn.Linear(int(args.resimnet_dim*2/3 + args.drug_embed_dim/3),
                    int(args.resimnet_dim/3+args.drug_embed_dim*2/3), bias = True)
        self.drug_mlp3 = nn.Linear(int(args.resimnet_dim/3+args.drug_embed_dim*2/3),
                    args.drug_embed_dim, bias = True)

        self.gats = {}
        for i in range(args.num_gcn_hops):
            if i==0:
                self.gats[i] = geo_nn.GATConv(args.gene2vec_dim, args.gcn_hidden_dim1,
                                        heads = args.gat_num_heads)
            else:
                self.gats[i] = geo_nn.GATConv(args.gcn_hidden_dim1 * args.gat_num_heads,
                                        args.gcn_hidden_dim1,
                                        heads = args.gat_num_heads)
        for i in range(args.num_gcn_hops):
            self.add_module('gcn_{}'.format(i), self.gats[i])


        self.pred_emb_dim = args.drug_embed_dim + (args.num_gcn_hops*args.gcn_hidden_dim1*args.gat_num_heads) + 2

        self.pred_mlp1 = nn.Linear(self.pred_emb_dim, int(self.pred_emb_dim*2/3), bias = True)
        self.pred_mlp2 = nn.Linear(int(self.pred_emb_dim*2/3), int(self.pred_emb_dim/3), bias = True)
        self.pred_mlp3 = nn.Linear(int(self.pred_emb_dim/3), args.num_classes, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()


        #   
        self.g2v_embeddings = nn.Embedding(args.num_genes, args.gene2vec_dim)
        if args.g2v_pretrained == True:
            g2v_embedding = F.normalize(torch.from_numpy(g2v_embedding), p = 2)
            self.g2v_embeddings.weight.data.copy_(g2v_embedding)


        self.ppi_adj = ppi_adj # 2 x num edges
        print(self.ppi_adj.shape)

        #   Attention layers
        if args.attn_dim != 0 :
            self.query_proj = nn.Linear(args.drug_embed_dim, args.attn_dim, bias = True)
            self.key_proj = nn.Linear(args.gene2vec_dim, args.attn_dim, bias = True)


        #   Read out MLP
        
        self.readout_mlp1 = nn.Linear(args.num_genes,
               int(args.num_genes*2/3), bias = True)
        self.readout_mlp2 = nn.Linear(int(args.num_genes*2/3), 
                int(args.num_genes/3), bias = True)
        self.readout_mlp3 = nn.Linear(int(args.num_genes/3), 1, bias = True)


    def forward(self, x, adj, get_gex_idxs, device, args, epoch, training = True):
        # x[0] : ecfp, x[1] : gex, x[2] : dosage, x[3] : duration


        self.tmp_batch_size = x[4].shape[0]

        drug_input = x[8].float().to(device)
        gex_input = x[1][:, get_gex_idxs].float().to(device) # bs x n genes x 1
        dose = x[2].float().to(device).view(-1,1)
        duration = x[3].float().to(device).view(-1,1)
        try:
            drug_input = self.bn1(drug_input)
            gex_input = self.bn2(gex_input).unsqueeze(-1)
            dose = self.bn3(dose)
            duration = self.bn4(duration)
        except:
            gex_input = gex_input.unsqueeze(-1)
    
    
        drug_input = F.dropout(drug_input, training = training)
        gex_input = F.dropout(gex_input, training = training)

        drug_emb = F.dropout(self.activ(self.drug_mlp1(drug_input)), training =training)
        drug_emb = F.dropout(self.activ(self.drug_mlp2(drug_emb)), training =training)
        #   #   #   #  get drug embed 
        drug_emb = F.dropout(self.activ(self.drug_mlp3(drug_emb)), training =training)
        self.drug_embed = drug_emb

        gene2vec = self.g2v_embeddings(torch.tensor([i for i in range(args.num_genes)]).to(device))

        #   Get attn btw each gene & drug
        gene2vec = gene2vec.unsqueeze(0).repeat(self.tmp_batch_size, 1, 1)

        #   #   #   #

        gex_embed = gex_input * gene2vec # bs x n genes x vec dims

        gex_embed = gex_embed.view(-1, args.gene2vec_dim) # bs*n genex x vec dims

        #   batch index for sortpooling
        batch_idx = np.hstack(
                    [[i for _ in range(gene2vec.shape[1])] for i in range(self.tmp_batch_size)])
        batch_idx = torch.tensor(batch_idx).to(device)

        ppi_adj = get_batch_adj(self.ppi_adj, self.tmp_batch_size, args)

        ppi_adj = torch.tensor(ppi_adj, dtype = torch.long).to(device)

        gcn_cat_list = []
        for i in range(args.num_gcn_hops):
            gex_embed = F.dropout(self.activ(
                                        self.gats[i](
                                            gex_embed, 
                                            ppi_adj)
                                        ), training = training)

            if args.gat_num_heads >1 :
                #   bs x num nodes x gcn dim * gat num heads
                gex_embed = torch.split(gex_embed,
                                        split_size_or_sections = args.gcn_hidden_dim1,
                                        dim = -1)
                gex_embed = torch.cat(gex_embed, dim = -1)  #bs * num nodes x 256

            #   append each readouts
            gcn_cat_list.append(gex_embed.view(self.tmp_batch_size, -1, 
                        args.gcn_hidden_out*args.gat_num_heads)) # bs x n nodes x 256

        #   get GAT node features for oversmoothing check
        self.gcn_cat_list = gcn_cat_list


        #   Readout + view batchwise

        if args.sort_pool_k == 0:
            read_out = torch.cat(gcn_cat_list, dim = -1) #  bs x n nodes x 728
            read_out = read_out.transpose(2, 1) #   bs x 728 x n nodes
            read_out = F.dropout(self.activ(self.readout_mlp1(read_out)), training = training)
            read_out = F.dropout(self.activ(self.readout_mlp2(read_out)), training = training)
            read_out = F.dropout(self.activ(self.readout_mlp3(read_out)), training = training).squeeze(-1)
        

        #   sort pooling
        else:
            sort_pool_k = int(np.ceil(args.num_genes * args.sort_pool_k))
            batch_output = geo_nn.global_sort_pool(gex_embed, batch_idx, sort_pool_k)
            batch_output = batch_output.view(self.tmp_batch_size, -1, args.gcn_hidden_out)
            ####
            self.attn, batch_output = drug2gex_attn_applied(drug_emb, batch_output, self, args, training)
            ####
            read_out = torch.mean(batch_output, dim = 1)


        total_emb = torch.cat([drug_emb, read_out, dose, duration], dim = -1) 

        total_emb = F.dropout(self.activ(self.pred_mlp1(total_emb)), training = training)
        total_emb = F.dropout(self.activ(self.pred_mlp2(total_emb)), training = training)
        proba = self.pred_mlp3(total_emb)
        
        return proba

