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
#            drug_proj = self.query_proj(drug_feat)
    if args.attn_type == 'multi':
        drug_proj = F.dropout(
#                    F.sigmoid(self.query_proj(drug_feat)), 
#                    torch.sigmoid(self.query_proj(drug_feat)),
                        torch.tanh(
                        self.query_proj(drug_feat)),
                        training = training)
        #   bs x 200
#            gex_proj = self.key_proj(gex_feat)
        gex_proj = F.dropout(
#                F.sigmoid(self.key_proj(gex_feat)),
#                    torch.sigmoid(self.key_proj(gex_feat)),
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
#                    F.sigmoid(self.query_proj(drug_feat)), 
#                    torch.sigmoid(self.query_proj(drug_feat)),
                        self.query_proj(drug_feat),
                        training = training)
        #   bs x 200
#            gex_proj = self.key_proj(gex_feat)
        gex_proj = F.dropout(
#                F.sigmoid(self.key_proj(gex_feat)),
#                    torch.sigmoid(self.key_proj(gex_feat)),
                        self.key_proj(gex_feat),
                        training = training)
        #   bs x num genes x 200

        #   Additive Attention
        drug_proj = drug_proj.unsqueeze(1)
        attn = self.attn_proj(torch.tanh(drug_proj + gex_proj)).squeeze(-1)
        attn = F.softmax(attn, dim = -1)

    gex_attned = torch.mul(attn.unsqueeze(-1), gex_feat) # bs x num_genes x 200
    
    return attn, gex_attned





class Resimnet(nn.Module):
    def __init__(self, args):
        super(Resimnet, self).__init__()

        print('num genes : '+str(args.num_genes))
        self.bn1 = nn.BatchNorm1d(args.resimnet_dim)
        self.bn2 = nn.BatchNorm1d(args.num_genes)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn4 = nn.BatchNorm1d(1)

#        self.drug_mlp = nn.Linear(args.resimnet_dim, drug_embed_dim, bias = True)
#        self.gex_mlp = nn.Linear(args.num_genes, gex_embed_dim, bias = True)
        """
        self.drug_mlp1 = nn.Linear(args.resimnet_dim, 
                    int(0.5*args.resimnet_dim + 0.5*args.drug_embed_dim), bias = True)
        self.drug_mlp2 = nn.Linear(int(0.5*args.resimnet_dim+0.5*args.drug_embed_dim),
                    args.drug_embed_dim, bias = True)
        self.gex_mlp1 = nn.Linear(args.num_genes, 
                int(0.5*args.num_genes + 0.5*args.gex_embed_dim), bias = True)
        self.gex_mlp2 = nn.Linear(int(0.5*args.num_genes+0.5*args.gex_embed_dim), 
                    args.gex_embed_dim, bias = True)
        """
        self.drug_mlp1 = nn.Linear(args.resimnet_dim, 
                    int(args.resimnet_dim*2/3 + args.drug_embed_dim/3), bias = True)
        self.drug_mlp2 = nn.Linear(int(args.resimnet_dim*2/3 + args.drug_embed_dim/3),
                    int(args.resimnet_dim/3+args.drug_embed_dim*2/3), bias = True)
        self.drug_mlp3 = nn.Linear(int(args.resimnet_dim/3+args.drug_embed_dim*2/3),
                    args.drug_embed_dim, bias = True)


        self.gex_mlp1 = nn.Linear(args.num_genes, 
                    int(args.num_genes*2/3 + args.gex_embed_dim/3), bias = True)
        self.gex_mlp2 = nn.Linear(int(args.num_genes*2/3 + args.gex_embed_dim/3),
                    int(args.num_genes/3+args.drug_embed_dim*2/3), bias = True)
        self.gex_mlp3 = nn.Linear(int(args.num_genes/3+args.gex_embed_dim*2/3),
                    args.gex_embed_dim, bias = True)

        self.pred_emb_dim = args.drug_embed_dim + args.gex_embed_dim + 2
        self.pred_mlp1 = nn.Linear(self.pred_emb_dim, int(self.pred_emb_dim*2/3), bias = True)
        self.pred_mlp2 = nn.Linear(int(self.pred_emb_dim*2/3), int(self.pred_emb_dim/3), bias = True)
        self.pred_mlp3 = nn.Linear(int(self.pred_emb_dim/3), args.num_classes, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()


    def forward(self, x, device, args, training = True):
        # x[0] : ecfp, x[1] : gex, x[2] : dosage, x[3] : duration

        #   If using pretrained drug feat:
        drug_input = x[8].float().to(device)

        gex_input = x[1].float().to(device)
        dose = x[2].float().to(device).view(-1,1)
        duration = x[3].float().to(device).view(-1,1)

        try:
            drug_input = self.bn1(drug_input)
            gex_input = self.bn2(gex_input)
            dose = self.bn3(dose)
            duration = self.bn4(duration)
        except ValueError:
            pass

        
        drug_emb = F.dropout(self.activ(self.drug_mlp3(
                    F.dropout(self.activ(self.drug_mlp2(self.activ(self.drug_mlp1(
                            drug_input)))),training = training))), training = training)
        gex_emb = F.dropout(self.activ(self.gex_mlp3(
                    F.dropout(self.activ(self.gex_mlp2(self.activ(self.gex_mlp1(
                            gex_input)))), training = training))), training = training)


        total_emb = torch.cat([drug_emb, gex_emb, dose, duration],
                                dim = -1)

        total_emb = F.dropout(self.activ(self.pred_mlp1(total_emb)), training = training)
        total_emb = F.dropout(self.activ(self.pred_mlp2(total_emb)), training = training)
        proba = self.pred_mlp3(total_emb)
        
        return proba

class Resimnet2(nn.Module): #   Gex only
    def __init__(self, args):
        super(Resimnet2, self).__init__()

        print('num genes : '+str(args.num_genes))
        self.bn2 = nn.BatchNorm1d(args.num_genes)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn4 = nn.BatchNorm1d(1)

#        self.drug_mlp = nn.Linear(args.resimnet_dim, drug_embed_dim, bias = True)
#        self.gex_mlp = nn.Linear(args.num_genes, gex_embed_dim, bias = True)
        self.gex_mlp1 = nn.Linear(args.num_genes, 
                    int(args.num_genes*2/3 + args.gex_embed_dim/3), bias = True)
        self.gex_mlp2 = nn.Linear(int(args.num_genes*2/3 + args.gex_embed_dim/3),
                    int(args.num_genes/3+args.drug_embed_dim*2/3), bias = True)
        self.gex_mlp3 = nn.Linear(int(args.num_genes/3+args.gex_embed_dim*2/3),
                    args.gex_embed_dim, bias = True)
#        self.pred_emb_dim = args.drug_embed_dim + args.gex_embed_dim + 2
        self.pred_emb_dim = args.gex_embed_dim + 2
        self.pred_mlp1 = nn.Linear(self.pred_emb_dim, int(self.pred_emb_dim*2/3), bias = True)
        self.pred_mlp2 = nn.Linear(int(self.pred_emb_dim*2/3), int(self.pred_emb_dim/3), bias = True)
        self.pred_mlp3 = nn.Linear(int(self.pred_emb_dim/3), args.num_classes, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()


    def forward(self, x, device, args, training = True):
        # x[0] : ecfp, x[1] : gex, x[2] : dosage, x[3] : duration

        #   If using pretrained drug feat:
        gex_input = x[1].float().to(device)
        dose = x[2].float().to(device).view(-1,1)
        duration = x[3].float().to(device).view(-1,1)

        try:
            gex_input = self.bn2(gex_input)
            dose = self.bn3(dose)
            duration = self.bn4(duration)
        except ValueError:
            pass

        
        gex_emb = F.dropout(self.activ(self.gex_mlp3(
                    F.dropout(self.activ(self.gex_mlp2(self.activ(self.gex_mlp1(
                            gex_input)))), training = training))), training = training)


#        total_emb = torch.cat([drug_emb, gex_emb, dose, duration],
        total_emb = torch.cat([gex_emb, dose, duration],
                                dim = -1)

        total_emb = F.dropout(self.activ(self.pred_mlp1(total_emb)), training = training)
        total_emb = F.dropout(self.activ(self.pred_mlp2(total_emb)), training = training)
        proba = self.pred_mlp3(total_emb)
        
        return proba


class GEX_PPI_GAT_MLP(nn.Module): 
    def __init__(self, ppi_adj, g2v_embedding, args):
        super(GEX_PPI_GAT_MLP, self).__init__()

        print('num genes : '+str(args.num_genes))

        self.bn1 = nn.BatchNorm1d(args.ecfp_nBits)
        self.bn2 = nn.BatchNorm1d(args.num_genes)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn4 = nn.BatchNorm1d(1)

        """
        self.drug_mlp = nn.Linear(args.resimnet_dim, drug_embed_dim, bias = True)
        self.drug_mlp1 = nn.Linear(args.resimnet_dim, 
                    int(0.5*args.resimnet_dim + 0.5*drug_embed_dim), bias = True)
        self.drug_mlp2 = nn.Linear(int(0.5*args.resimnet_dim+0.5*drug_embed_dim),
                    drug_embed_dim, bias = True)
        """
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
                self.gats[i] = geo_nn.GATConv(args.gcn_hidden_dim1, args.gcn_hidden_dim2,
                                        heads = args.gat_num_heads)
        for i in range(args.num_gcn_hops):
            self.add_module('gcn_{}'.format(i), self.gats[i])


        self.pred_emb_dim = args.drug_embed_dim + args.gcn_hidden_out + 2

#        self.pred_mlp = nn.Linear(self.pred_emb_dim, num_classes, bias = True)
        self.pred_mlp1 = nn.Linear(self.pred_emb_dim, int(self.pred_emb_dim*2/3), bias = True)
        self.pred_mlp2 = nn.Linear(int(self.pred_emb_dim*2/3), int(self.pred_emb_dim/3), bias = True)
        self.pred_mlp3 = nn.Linear(int(self.pred_emb_dim/3), args.num_classes, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()


        #   
        self.g2v_embeddings = nn.Embedding(args.num_genes, args.gene2vec_dim)
#        self.g2v_embeddings.weight.data.copy_(torch.from_numpy(g2v_embedding))
        g2v_embedding = F.normalize(torch.from_numpy(g2v_embedding), p = 2)
        self.g2v_embeddings.weight.data.copy_(g2v_embedding)

        self.ppi_adj = ppi_adj # 2 x num edges
        print(self.ppi_adj.shape)

        #   Attention layers
        self.query_proj = nn.Linear(args.drug_embed_dim, args.attn_dim, bias = True)
#        self.key_proj = nn.Linear(args.gex_embed_dim, args.attn_dim, bias = True)
        self.key_proj = nn.Linear(args.gene2vec_dim, args.attn_dim, bias = True)


    def forward(self, x, adj, get_gex_idxs, device, args, training = True):
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
    
        #torch.as_tensor(exp.float()).detach().to(device

        drug_input = F.dropout(drug_input, training = training)
        gex_input = F.dropout(gex_input, training = training)


#        drug_emb = self.drug_mlp(drug_input) 
#        drug_emb = F.dropout(self.activ(self.drug_mlp(drug_input)), training =training)
        drug_emb = F.dropout(self.activ(self.drug_mlp1(drug_input)), training =training)
        drug_emb = F.dropout(self.activ(self.drug_mlp2(drug_emb)), training =training)
        drug_emb = F.dropout(self.activ(self.drug_mlp3(drug_emb)), training =training)
        


#        gene2vec = self.g2v_embedding.float().to(device) # 1 x n genes x vec dims
        gene2vec = self.g2v_embeddings(torch.tensor([i for i in range(args.num_genes)]).to(device))

        #   Get attn btw each gene & drug
        gene2vec = gene2vec.unsqueeze(0).repeat(self.tmp_batch_size, 1, 1)

        if args.attn_dim != 0:
            self.attn, gene2vec = drug2gex_attn_applied(drug_emb, gene2vec, self, args, training)
        #   #   #   #

        gex_embed = gex_input * gene2vec # bs x n genes x vec dims

        gex_embed = gex_embed.view(-1, args.gene2vec_dim) # bs*n genex x vec dims

        #   batch index for sortpooling
        batch_idx = np.hstack(
                    [[i for _ in range(gene2vec.shape[1])] for i in range(self.tmp_batch_size)])
        batch_idx = torch.tensor(batch_idx).to(device)

        ppi_adj = get_batch_adj(self.ppi_adj, self.tmp_batch_size, args)

        ppi_adj = torch.tensor(ppi_adj, dtype = torch.long).to(device)

        
        for i in range(args.num_gcn_hops):
            if i != 0:
                gex_embed_0 = gex_embed
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
                """
                #   #   #   #
                #   apply attn
                gex_embed_attn = []
                for head in gex_embed:
                    head = head.view(self.tmp_batch_size, -1, args.gcn_hidden_dim1)
                    self.attn, head = drug2gex_attn_applied(drug_emb, head, self, args, training)
                    head = head.view(-1, args.gcn_hidden_dim1)
                    gex_embed_attn.append(head)
                gex_embed = gex_embed_attn
                #   #   #   #
                """
                gex_embed = torch.mean(torch.stack(
                                        gex_embed, dim = -1), dim = -1)
            if i!=0:
                gex_embed += gex_embed_0



        #   Readout + view batchwise

        if args.sort_pool_k == 0:
            batch_output = gex_embed.view(
                                self.tmp_batch_size, -1, args.gcn_hidden_out)
            ####
#            self.attn, batch_output = drug2gex_attn_applied(drug_emb, batch_output, self, args, training)
            ####
            read_out = torch.mean(batch_output, dim = 1)
        

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
        
#        proba = self.pred_mlp(total_emb)
        total_emb = F.dropout(self.activ(self.pred_mlp1(total_emb)), training = training)
        total_emb = F.dropout(self.activ(self.pred_mlp2(total_emb)), training = training)
        proba = self.pred_mlp3(total_emb)
        
        return proba

class GEX_PPI_GAT_cat_MLP(nn.Module):
    def __init__(self, ppi_adj, g2v_embedding, args):
        super(GEX_PPI_GAT_cat_MLP, self).__init__()

#        self.gex_mlp = nn.Linear(args.num_genes, gex_embed_dim, bias = True)

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


#        self.pred_emb_dim = args.drug_embed_dim + args.gcn_hidden_out + 2
        self.pred_emb_dim = args.drug_embed_dim + (args.gcn_hidden_dim1*args.gat_num_heads) + 2

#        self.pred_mlp = nn.Linear(self.pred_emb_dim, num_classes, bias = True)
        self.pred_mlp1 = nn.Linear(self.pred_emb_dim, int(self.pred_emb_dim*2/3), bias = True)
        self.pred_mlp2 = nn.Linear(int(self.pred_emb_dim*2/3), int(self.pred_emb_dim/3), bias = True)
        self.pred_mlp3 = nn.Linear(int(self.pred_emb_dim/3), args.num_classes, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()


        #   
        self.g2v_embeddings = nn.Embedding(args.num_genes, args.gene2vec_dim)
#        self.g2v_embeddings.weight.data.copy_(torch.from_numpy(g2v_embedding))
        if args.g2v_pretrained == True:
            g2v_embedding = F.normalize(torch.from_numpy(g2v_embedding), p = 2)
            self.g2v_embeddings.weight.data.copy_(g2v_embedding)


        self.ppi_adj = ppi_adj # 2 x num edges
        print(self.ppi_adj.shape)

        #   Attention layers
        if args.attn_dim != 0 :
            self.query_proj = nn.Linear(args.drug_embed_dim, args.attn_dim, bias = True)
#        self.key_proj = nn.Linear(args.gex_embed_dim, args.attn_dim, bias = True)
            self.key_proj = nn.Linear(args.gene2vec_dim, args.attn_dim, bias = True)
#            self.attn_proj = nn.Linear(args.attn_dim, 1, bias = True)


    def forward(self, x, adj, get_gex_idxs, device, args, training = True):
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
        drug_emb = F.dropout(self.activ(self.drug_mlp3(drug_emb)), training =training)
        
        gene2vec = self.g2v_embeddings(torch.tensor([i for i in range(args.num_genes)]).to(device))

        #   Get attn btw each gene & drug
        gene2vec = gene2vec.unsqueeze(0).repeat(self.tmp_batch_size, 1, 1)

        if args.attn_dim != 0:
            self.attn, gene2vec = drug2gex_attn_applied(drug_emb, gene2vec, self, args, training)
        #   #   #   #

        gex_embed = gex_input * gene2vec # bs x n genes x vec dims

        gex_embed = gex_embed.view(-1, args.gene2vec_dim) # bs*n genex x vec dims

        #   batch index for sortpooling
        batch_idx = np.hstack(
                    [[i for _ in range(gene2vec.shape[1])] for i in range(self.tmp_batch_size)])
        batch_idx = torch.tensor(batch_idx).to(device)

        ppi_adj = get_batch_adj(self.ppi_adj, self.tmp_batch_size, args)

        ppi_adj = torch.tensor(ppi_adj, dtype = torch.long).to(device)

        
        for i in range(args.num_gcn_hops):
            if i != 0:
                gex_embed_0 = gex_embed
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
                #   #   #   #
                #   apply attn
                """
                gex_embed_attn = []
                for head in gex_embed:
                    head = head.view(self.tmp_batch_size, -1, args.gcn_hidden_dim1)
                    self.attn, head = drug2gex_attn_applied(drug_emb, head, self, args, training)
                    head = head.view(-1, args.gcn_hidden_dim1)
                    gex_embed_attn.append(head)
                gex_embed = gex_embed_attn
                """
                #   #   #   #
#                gex_embed = torch.stack(gex_embed, dim = -1)
                gex_embed = torch.cat(gex_embed, dim = -1)

            if i!=0:
                gex_embed += gex_embed_0

        #   Readout + view batchwise

        if args.sort_pool_k == 0:
            batch_output = gex_embed.view(
                                self.tmp_batch_size, -1, args.gcn_hidden_out*args.gat_num_heads)
            ####
#            self.attn, batch_output = drug2gex_attn_applied(drug_emb, batch_output, self, args, training)
            ####
            read_out = torch.mean(batch_output, dim = 1)
        

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
        
#        proba = self.pred_mlp(total_emb)

        total_emb = F.dropout(self.activ(self.pred_mlp1(total_emb)), training = training)
        total_emb = F.dropout(self.activ(self.pred_mlp2(total_emb)), training = training)
        proba = self.pred_mlp3(total_emb)
        
        return proba

class GEX_PPI_GAT_cat2_MLP(nn.Module):
    def __init__(self, ppi_adj, g2v_embedding, args):
        super(GEX_PPI_GAT_cat2_MLP, self).__init__()

#        self.gex_mlp = nn.Linear(args.num_genes, gex_embed_dim, bias = True)

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


#        self.pred_emb_dim = args.drug_embed_dim + args.gcn_hidden_out + 2
#        self.pred_emb_dim = args.drug_embed_dim + (args.gcn_hidden_dim1*args.gat_num_heads) + 2
        self.pred_emb_dim = args.drug_embed_dim + (args.num_gcn_hops*args.gcn_hidden_dim1*args.gat_num_heads) + 2

#        self.pred_mlp = nn.Linear(self.pred_emb_dim, num_classes, bias = True)
        self.pred_mlp1 = nn.Linear(self.pred_emb_dim, int(self.pred_emb_dim*2/3), bias = True)
        self.pred_mlp2 = nn.Linear(int(self.pred_emb_dim*2/3), int(self.pred_emb_dim/3), bias = True)
        self.pred_mlp3 = nn.Linear(int(self.pred_emb_dim/3), args.num_classes, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()


        #   
        self.g2v_embeddings = nn.Embedding(args.num_genes, args.gene2vec_dim)
#        self.g2v_embeddings.weight.data.copy_(torch.from_numpy(g2v_embedding))
        if args.g2v_pretrained == True:
            g2v_embedding = F.normalize(torch.from_numpy(g2v_embedding), p = 2)
            self.g2v_embeddings.weight.data.copy_(g2v_embedding)


        self.ppi_adj = ppi_adj # 2 x num edges
        print(self.ppi_adj.shape)

        #   Attention layers
        if args.attn_dim != 0 :
            self.query_proj = nn.Linear(args.drug_embed_dim, args.attn_dim, bias = True)
#        self.key_proj = nn.Linear(args.gex_embed_dim, args.attn_dim, bias = True)
            self.key_proj = nn.Linear(args.gene2vec_dim, args.attn_dim, bias = True)
#            self.attn_proj = nn.Linear(args.attn_dim, 1, bias = True)


    def forward(self, x, adj, get_gex_idxs, device, args, training = True):
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
        drug_emb = F.dropout(self.activ(self.drug_mlp3(drug_emb)), training =training)
        
        gene2vec = self.g2v_embeddings(torch.tensor([i for i in range(args.num_genes)]).to(device))

        #   Get attn btw each gene & drug
        gene2vec = gene2vec.unsqueeze(0).repeat(self.tmp_batch_size, 1, 1)

        if args.attn_dim != 0:
            self.attn, gene2vec = drug2gex_attn_applied(drug_emb, gene2vec, self, args, training)
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
                #   #   #   #
                #   apply attn
                """
                gex_embed_attn = []
                for head in gex_embed:
                    head = head.view(self.tmp_batch_size, -1, args.gcn_hidden_dim1)
                    self.attn, head = drug2gex_attn_applied(drug_emb, head, self, args, training)
                    head = head.view(-1, args.gcn_hidden_dim1)
                    gex_embed_attn.append(head)
                gex_embed = gex_embed_attn
                """
                #   #   #   #
#                gex_embed = torch.stack(gex_embed, dim = -1)
                gex_embed = torch.cat(gex_embed, dim = -1)  #bs * num nodes x 256

            #   append each readouts
            read_out = torch.mean(gex_embed.view(self.tmp_batch_size, -1, 
                        args.gcn_hidden_out*args.gat_num_heads), dim = 1) # bs x  256
            gcn_cat_list.append(read_out)


        #   Readout + view batchwise

        if args.sort_pool_k == 0:
#            batch_output = gex_embed.view(
#                                self.tmp_batch_size, -1, args.gcn_hidden_out*args.gat_num_heads)
            ####
#            self.attn, batch_output = drug2gex_attn_applied(drug_emb, batch_output, self, args, training)
            ####
#            read_out = torch.mean(batch_output, dim = 1)
            read_out = torch.cat(gcn_cat_list, dim = -1)
        

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
        
#        proba = self.pred_mlp(total_emb)

        total_emb = F.dropout(self.activ(self.pred_mlp1(total_emb)), training = training)
        total_emb = F.dropout(self.activ(self.pred_mlp2(total_emb)), training = training)
        proba = self.pred_mlp3(total_emb)
        
        return proba

class GEX_PPI_GAT_cat3_MLP(nn.Module):  #   MLP for read out, similar to drug attn
    def __init__(self, ppi_adj, g2v_embedding, args):
        super(GEX_PPI_GAT_cat3_MLP, self).__init__()


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


#        self.pred_emb_dim = args.drug_embed_dim + args.gcn_hidden_out + 2
#        self.pred_emb_dim = args.drug_embed_dim + (args.gcn_hidden_dim1*args.gat_num_heads) + 2
#        self.pred_emb_dim = args.drug_embed_dim + (args.num_gcn_hops*args.gcn_hidden_dim1*args.gat_num_heads) + 2
        self.pred_emb_dim = args.drug_embed_dim + args.num_genes + 2

#        self.pred_mlp = nn.Linear(self.pred_emb_dim, num_classes, bias = True)
        self.pred_mlp1 = nn.Linear(self.pred_emb_dim, int(self.pred_emb_dim*2/3), bias = True)
        self.pred_mlp2 = nn.Linear(int(self.pred_emb_dim*2/3), int(self.pred_emb_dim/3), bias = True)
        self.pred_mlp3 = nn.Linear(int(self.pred_emb_dim/3), args.num_classes, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()


        #   
        self.g2v_embeddings = nn.Embedding(args.num_genes, args.gene2vec_dim)
#        self.g2v_embeddings.weight.data.copy_(torch.from_numpy(g2v_embedding))
        if args.g2v_pretrained == True:
            g2v_embedding = F.normalize(torch.from_numpy(g2v_embedding), p = 2)
            self.g2v_embeddings.weight.data.copy_(g2v_embedding)


        self.ppi_adj = ppi_adj # 2 x num edges
        print(self.ppi_adj.shape)

        #   Attention layers
        if args.attn_dim != 0 :
            self.query_proj = nn.Linear(args.drug_embed_dim, args.attn_dim, bias = True)
#        self.key_proj = nn.Linear(args.gex_embed_dim, args.attn_dim, bias = True)
            self.key_proj = nn.Linear(args.gene2vec_dim, args.attn_dim, bias = True)
#            self.attn_proj = nn.Linear(args.attn_dim, 1, bias = True)


        #   Read out MLP
        
        self.readout_mlp1 = nn.Linear(args.num_gcn_hops*args.gcn_hidden_dim1*args.gat_num_heads,
               int(args.num_gcn_hops*args.gcn_hidden_dim1*args.gat_num_heads*2/3), bias = True)
        self.readout_mlp2 = nn.Linear(int(args.num_gcn_hops*args.gcn_hidden_dim1*args.gat_num_heads*2/3), int(args.num_gcn_hops*args.gcn_hidden_dim1*args.gat_num_heads/3), bias = True)
        self.readout_mlp3 = nn.Linear(int(args.num_gcn_hops*args.gcn_hidden_dim1*args.gat_num_heads/3), 1, bias = True)




    def forward(self, x, adj, get_gex_idxs, device, args, training = True):
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
        drug_emb = F.dropout(self.activ(self.drug_mlp3(drug_emb)), training =training)
        
        gene2vec = self.g2v_embeddings(torch.tensor([i for i in range(args.num_genes)]).to(device))

        #   Get attn btw each gene & drug
        gene2vec = gene2vec.unsqueeze(0).repeat(self.tmp_batch_size, 1, 1)

        if args.attn_dim != 0:
            self.attn, gene2vec = drug2gex_attn_applied(drug_emb, gene2vec, self, args, training)
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
                #   #   #   #
                #   apply attn
                """
                gex_embed_attn = []
                for head in gex_embed:
                    head = head.view(self.tmp_batch_size, -1, args.gcn_hidden_dim1)
                    self.attn, head = drug2gex_attn_applied(drug_emb, head, self, args, training)
                    head = head.view(-1, args.gcn_hidden_dim1)
                    gex_embed_attn.append(head)
                gex_embed = gex_embed_attn
                """
                #   #   #   #
#                gex_embed = torch.stack(gex_embed, dim = -1)
                gex_embed = torch.cat(gex_embed, dim = -1)  #bs * num nodes x 256

            #   append each readouts
            """
            read_out = torch.mean(gex_embed.view(self.tmp_batch_size, -1, 
                        args.gcn_hidden_out*args.gat_num_heads), dim = 1) # bs x  256
            gcn_cat_list.append(read_out)
            """
            gcn_cat_list.append(gex_embed.view(self.tmp_batch_size, -1, 
                        args.gcn_hidden_out*args.gat_num_heads)) # bs x n nodes x 256


        #   Readout + view batchwise

        if args.sort_pool_k == 0:
#            batch_output = gex_embed.view(
#                                self.tmp_batch_size, -1, args.gcn_hidden_out*args.gat_num_heads)
            ####
#            self.attn, batch_output = drug2gex_attn_applied(drug_emb, batch_output, self, args, training)
            ####
#            read_out = torch.mean(batch_output, dim = 1)
            read_out = torch.cat(gcn_cat_list, dim = -1) #  bs x n nodes x 728
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
        
#        proba = self.pred_mlp(total_emb)

        total_emb = F.dropout(self.activ(self.pred_mlp1(total_emb)), training = training)
        total_emb = F.dropout(self.activ(self.pred_mlp2(total_emb)), training = training)
        proba = self.pred_mlp3(total_emb)
        
        return proba

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


#        self.pred_emb_dim = args.drug_embed_dim + args.gcn_hidden_out + 2
#        self.pred_emb_dim = args.drug_embed_dim + (args.gcn_hidden_dim1*args.gat_num_heads) + 2
        self.pred_emb_dim = args.drug_embed_dim + (args.num_gcn_hops*args.gcn_hidden_dim1*args.gat_num_heads) + 2

#        self.pred_mlp = nn.Linear(self.pred_emb_dim, num_classes, bias = True)
        self.pred_mlp1 = nn.Linear(self.pred_emb_dim, int(self.pred_emb_dim*2/3), bias = True)
        self.pred_mlp2 = nn.Linear(int(self.pred_emb_dim*2/3), int(self.pred_emb_dim/3), bias = True)
        self.pred_mlp3 = nn.Linear(int(self.pred_emb_dim/3), args.num_classes, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()


        #   
        self.g2v_embeddings = nn.Embedding(args.num_genes, args.gene2vec_dim)
#        self.g2v_embeddings.weight.data.copy_(torch.from_numpy(g2v_embedding))
        if args.g2v_pretrained == True:
            g2v_embedding = F.normalize(torch.from_numpy(g2v_embedding), p = 2)
            self.g2v_embeddings.weight.data.copy_(g2v_embedding)


        self.ppi_adj = ppi_adj # 2 x num edges
        print(self.ppi_adj.shape)

        #   Attention layers
        if args.attn_dim != 0 :
            self.query_proj = nn.Linear(args.drug_embed_dim, args.attn_dim, bias = True)
#        self.key_proj = nn.Linear(args.gex_embed_dim, args.attn_dim, bias = True)
            self.key_proj = nn.Linear(args.gene2vec_dim, args.attn_dim, bias = True)
#            self.attn_proj = nn.Linear(args.attn_dim, 1, bias = True)


        #   Read out MLP
        
        self.readout_mlp1 = nn.Linear(args.num_genes,
               int(args.num_genes*2/3), bias = True)
        self.readout_mlp2 = nn.Linear(int(args.num_genes*2/3), 
                int(args.num_genes/3), bias = True)
        self.readout_mlp3 = nn.Linear(int(args.num_genes/3), 1, bias = True)


#    def forward(self, x, adj, get_gex_idxs, device, args, training = True):
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
#        self.drug_embed = self.drug_mlp3(drug_emb)
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
            """
            if i != 0:
                gex_embed = self.activ(gex_embed)
            gex_embed = F.dropout(self.gats[i](
                                            gex_embed, 
                                            ppi_adj)
                                        ), training = training)
            """

            if args.gat_num_heads >1 :
                #   bs x num nodes x gcn dim * gat num heads
                gex_embed = torch.split(gex_embed,
                                        split_size_or_sections = args.gcn_hidden_dim1,
                                        dim = -1)
                #   #   #   #
                #   apply attn
                """
                gex_embed_attn = []
                for head in gex_embed:
                    head = head.view(self.tmp_batch_size, -1, args.gcn_hidden_dim1)
                    self.attn, head = drug2gex_attn_applied(drug_emb, head, self, args, training)
                    head = head.view(-1, args.gcn_hidden_dim1)
                    gex_embed_attn.append(head)
                gex_embed = gex_embed_attn
                """
                #   #   #   #
#                gex_embed = torch.stack(gex_embed, dim = -1)
                gex_embed = torch.cat(gex_embed, dim = -1)  #bs * num nodes x 256

            #   append each readouts
            """
            read_out = torch.mean(gex_embed.view(self.tmp_batch_size, -1, 
                        args.gcn_hidden_out*args.gat_num_heads), dim = 1) # bs x  256
            gcn_cat_list.append(read_out)
            """
            gcn_cat_list.append(gex_embed.view(self.tmp_batch_size, -1, 
                        args.gcn_hidden_out*args.gat_num_heads)) # bs x n nodes x 256

        #   get GAT node features for oversmoothing check
        self.gcn_cat_list = gcn_cat_list


        #   Readout + view batchwise

        if args.sort_pool_k == 0:
#            batch_output = gex_embed.view(
#                                self.tmp_batch_size, -1, args.gcn_hidden_out*args.gat_num_heads)
            ####
#            self.attn, batch_output = drug2gex_attn_applied(drug_emb, batch_output, self, args, training)
            ####
#            read_out = torch.mean(batch_output, dim = 1)
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
#        total_emb = torch.cat([read_out, dose, duration], dim = -1) 
        
#        proba = self.pred_mlp(total_emb)

        total_emb = F.dropout(self.activ(self.pred_mlp1(total_emb)), training = training)
        total_emb = F.dropout(self.activ(self.pred_mlp2(total_emb)), training = training)
        proba = self.pred_mlp3(total_emb)
        
        return proba

class GEX_PPI_GAT_cat6_MLP(nn.Module):  
    def __init__(self, ppi_adj, g2v_embedding, args):
        super(GEX_PPI_GAT_cat6_MLP, self).__init__()


        print('num genes : '+str(args.num_genes))
        self.bn1 = nn.BatchNorm1d(args.ecfp_nBits)
        self.bn2 = nn.BatchNorm1d(args.num_genes)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn4 = nn.BatchNorm1d(1)

        """
        self.drug_mlp1 = nn.Linear(args.resimnet_dim, 
                    int(args.resimnet_dim*2/3 + args.drug_embed_dim/3), bias = True)
        self.drug_mlp2 = nn.Linear(int(args.resimnet_dim*2/3 + args.drug_embed_dim/3),
                    int(args.resimnet_dim/3+args.drug_embed_dim*2/3), bias = True)
        self.drug_mlp3 = nn.Linear(int(args.resimnet_dim/3+args.drug_embed_dim*2/3),
                    args.drug_embed_dim, bias = True)
        """
        self.drug_mlp1 = nn.Linear(args.resimnet_dim, 1024, bias = True)
        self.drug_mlp2 = nn.Linear(1024, 512, bias = True)
        self.drug_mlp3 = nn.Linear(512, args.drug_embed_dim, bias = True)
                                  

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


#        self.pred_emb_dim = args.drug_embed_dim + args.gcn_hidden_out + 2
#        self.pred_emb_dim = args.drug_embed_dim + (args.gcn_hidden_dim1*args.gat_num_heads) + 2
        self.pred_emb_dim = args.drug_embed_dim + (args.num_gcn_hops*args.gcn_hidden_dim1*args.gat_num_heads) + 2

#        self.pred_mlp = nn.Linear(self.pred_emb_dim, num_classes, bias = True)
        """
        self.pred_mlp1 = nn.Linear(self.pred_emb_dim, int(self.pred_emb_dim*2/3), bias = True)
        self.pred_mlp2 = nn.Linear(int(self.pred_emb_dim*2/3), int(self.pred_emb_dim/3), bias = True)
        self.pred_mlp3 = nn.Linear(int(self.pred_emb_dim/3), args.num_classes, bias = True)
        """
        self.pred_mlp1 = nn.Linear(self.pred_emb_dim, 512, bias = True)
        self.pred_mlp2 = nn.Linear(512, 256, bias = True)
        self.pred_mlp3 = nn.Linear(256, args.num_classes, bias=True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()


        #   
        self.g2v_embeddings = nn.Embedding(args.num_genes, args.gene2vec_dim)
#        self.g2v_embeddings.weight.data.copy_(torch.from_numpy(g2v_embedding))
        if args.g2v_pretrained == True:
            g2v_embedding = F.normalize(torch.from_numpy(g2v_embedding), p = 2)
            self.g2v_embeddings.weight.data.copy_(g2v_embedding)


        self.ppi_adj = ppi_adj # 2 x num edges
        print(self.ppi_adj.shape)


        #   Attention layers
        if args.attn_dim != 0 :
            self.query_proj = nn.Linear(args.drug_embed_dim, args.attn_dim, bias = True)
#        self.key_proj = nn.Linear(args.gex_embed_dim, args.attn_dim, bias = True)
            self.key_proj = nn.Linear(args.gene2vec_dim, args.attn_dim, bias = True)
#            self.attn_proj = nn.Linear(args.attn_dim, 1, bias = True)


        #   Read out MLP
        
        """
        self.readout_mlp1 = nn.Linear(args.num_genes,
               int(args.num_genes*2/3), bias = True)
        self.readout_mlp2 = nn.Linear(int(args.num_genes*2/3), 
                int(args.num_genes/3), bias = True)
        self.readout_mlp3 = nn.Linear(int(args.num_genes/3), 1, bias = True)
        """
        """
        self.readout_mlp1 = nn.Linear(args.num_genes, 512, bias=True)
        self.readout_mlp2 = nn.Linear(512, 256, bias = True)
        self.readout_mlp3 = nn.Linear(256, 1, bias=True)
        """



#    def forward(self, x, adj, get_gex_idxs, device, args, training = True):
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
        self.drug_embed = self.drug_mlp3(drug_emb)
        drug_emb = F.dropout(self.activ(self.drug_mlp3(drug_emb)), training =training)

        gene2vec = self.g2v_embeddings(torch.tensor([i for i in range(args.num_genes)]).to(device))

        #   #   #   #
        #   Get attn btw each gene & drug
        gene2vec = gene2vec.unsqueeze(0).repeat(self.tmp_batch_size, 1, 1)

        if args.attn_dim != 0:
            self.attn, gene2vec = drug2gex_attn_applied(drug_emb, gene2vec, self, args, training)
        #   #   #   #

        gex_embed = gex_input * gene2vec # bs x n genes x vec dims

        gex_embed = gex_embed.view(-1, args.gene2vec_dim) # bs*n genex x vec dims

        #   batch index for sortpooling
        batch_idx = np.hstack(
                    [[i for _ in range(gene2vec.shape[1])] for i in range(self.tmp_batch_size)])
        batch_idx = torch.tensor(batch_idx).to(device)

        """
        #   #   #   #
        #   ver.1
        #   Randomly sample edges
        num_edges = len(self.ppi_adj[0])
        sample_edge_idxs = torch.multinomial(torch.ones(num_edges).float(),
                                args.num_edge_samples)
        self.ppi_adj = self.ppi_adj[:, sample_edge_idxs]
        #   #   #   #

        #   #   #   #
        #   ver.2
        #   Randomly sample edges & use full graph for inference
        if training:
            num_edges = len(self.ppi_adj[0])
            sample_edge_idxs = torch.multinomial(torch.ones(num_edges).float(),
                                    args.num_edge_samples)
            self.ppi_adj = self.ppi_adj[:, sample_edge_idxs]
        #   #   #   #

        """
        #   #   #   #
        #   Randomly sample nodes
        if training:
#            args.num_genes = args.num_node_samples
            num_nodes = gex_embed.size(1)
            sample_node_idxs = torch.multinomial(torch.ones(num_nodes).float(),
                                    args.num_node_samples, replacement = True)
            to_sample_idxs = {}
            for i, x in enumerate(sample_node_idxs):
                to_sample_idxs[int(x)] = i


            gex_embed = gex_embed.view(self.tmp_batch_size, -1, args.gene2vec_dim) # bs, n genex x vec dims
            gex_embed = gex_embed[:, sample_node_idxs, :]
            gex_embed = gex_embed.view(-1, args.gene2vec_dim) # bs*n genex x vec dims


#            ppi_adj_0 = [to_sample_idxs[x] for x in self.ppi_adj[0] if x in sample_node_idxs]
#            ppi_adj_1 = [to_sample_idxs[x] for x in self.ppi_adj[1] if x in sample_node_idxs]
#            self.ppi_adj = np.vstack([ppi_adj_0, ppi_adj_1])
            ppi_adj_0_idx = [i for i, x in enumerate(self.ppi_adj[0]) if x in sample_node_idxs]
            ppi_adj_1_idx = [i for i, x in enumerate(self.ppi_adj[1]) if x in sample_node_idxs]
            common_idx = list(set(ppi_adj_0_idx)&set(ppi_adj_1_idx))

            self.ppi_adj = self.ppi_adj[:, common_idx]

            ppi_adj_0 = [to_sample_idxs[x] for x in self.ppi_adj[0] if x in sample_node_idxs]
            ppi_adj_1 = [to_sample_idxs[x] for x in self.ppi_adj[1] if x in sample_node_idxs]
            self.ppi_adj = np.vstack([ppi_adj_0, ppi_adj_1])

            gex_embed = gex_embed.to(device)
        #   #   #   #


        ppi_adj = get_batch_adj(self.ppi_adj, self.tmp_batch_size, args)

        ppi_adj = torch.tensor(ppi_adj, dtype = torch.long).to(device)


        gcn_cat_list = []
        for i in range(args.num_gcn_hops):
            """
            gex_embed = F.dropout(self.activ(
                                        self.gats[i](
                                            gex_embed, 
                                            ppi_adj)
                                        ), training = training)
            """
            if i != 0:
                gex_embed = self.activ(gex_embed)
            gex_embed = F.dropout(self.gats[i](
                                            gex_embed, 
                                            ppi_adj)
                                            ,training = training)

            if args.gat_num_heads >1 :
                #   bs x num nodes x gcn dim * gat num heads
                gex_embed = torch.split(gex_embed,
                                        split_size_or_sections = args.gcn_hidden_dim1,
                                        dim = -1)
                #   #   #   #
                #   apply attn
                """
                gex_embed_attn = []
                for head in gex_embed:
                    head = head.view(self.tmp_batch_size, -1, args.gcn_hidden_dim1)
                    self.attn, head = drug2gex_attn_applied(drug_emb, head, self, args, training)
                    head = head.view(-1, args.gcn_hidden_dim1)
                    gex_embed_attn.append(head)
                gex_embed = gex_embed_attn
                """
                #   #   #   #
#                gex_embed = torch.stack(gex_embed, dim = -1)
                gex_embed = torch.cat(gex_embed, dim = -1)  #bs * num nodes x 256

            #   append each readouts
            """
            read_out = torch.mean(gex_embed.view(self.tmp_batch_size, -1, 
                        args.gcn_hidden_out*args.gat_num_heads), dim = 1) # bs x  256
            gcn_cat_list.append(read_out)
            """
            gcn_cat_list.append(gex_embed.view(self.tmp_batch_size, -1, 
                        args.gcn_hidden_out*args.gat_num_heads)) # bs x n nodes x 256

        #   get GAT node features for oversmoothing check
        self.gcn_cat_list = gcn_cat_list


        #   Readout + view batchwise

        if args.sort_pool_k == 0:
#            batch_output = gex_embed.view(
#                                self.tmp_batch_size, -1, args.gcn_hidden_out*args.gat_num_heads)
            ####
#            self.attn, batch_output = drug2gex_attn_applied(drug_emb, batch_output, self, args, training)
            ####
#            read_out = torch.mean(batch_output, dim = 1)

            read_out = torch.cat(gcn_cat_list, dim = -1) #  bs x n nodes x 728
            """
            read_out = read_out.transpose(2, 1) #   bs x 728 x n nodes
            read_out = F.dropout(self.activ(self.readout_mlp1(read_out)), training = training)
            read_out = F.dropout(self.activ(self.readout_mlp2(read_out)), training = training)
            read_out = F.dropout(self.activ(self.readout_mlp3(read_out)), training = training).squeeze(-1)
            """
            read_out = torch.mean(read_out, dim = 1)
        

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
#        total_emb = torch.cat([read_out, dose, duration], dim = -1) 
        
#        proba = self.pred_mlp(total_emb)

        total_emb = F.dropout(self.activ(self.pred_mlp1(total_emb)), training = training)
        total_emb = F.dropout(self.activ(self.pred_mlp2(total_emb)), training = training)
        proba = self.pred_mlp3(total_emb)
        
        return proba


class GEX_PPI_GAT_cat5_MLP(nn.Module):  #   gEX only
    def __init__(self, ppi_adj, g2v_embedding, args):
        super(GEX_PPI_GAT_cat5_MLP, self).__init__()


        print('num genes : '+str(args.num_genes))
        self.bn2 = nn.BatchNorm1d(args.num_genes)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn4 = nn.BatchNorm1d(1)

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


#        self.pred_emb_dim = args.drug_embed_dim + (args.num_gcn_hops*args.gcn_hidden_dim1*args.gat_num_heads) + 2
        self.pred_emb_dim = (args.num_gcn_hops*args.gcn_hidden_dim1*args.gat_num_heads) + 2

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


    def forward(self, x, adj, get_gex_idxs, device, args, training = True):
        # x[0] : ecfp, x[1] : gex, x[2] : dosage, x[3] : duration


        self.tmp_batch_size = x[4].shape[0]

        gex_input = x[1][:, get_gex_idxs].float().to(device) # bs x n genes x 1
        dose = x[2].float().to(device).view(-1,1)
        duration = x[3].float().to(device).view(-1,1)
        try:
            gex_input = self.bn2(gex_input).unsqueeze(-1)
            dose = self.bn3(dose)
            duration = self.bn4(duration)
        except:
            gex_input = gex_input.unsqueeze(-1)
    
    
        gex_input = F.dropout(gex_input, training = training)

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
                #   #   #   #
                #   apply attn
                """
                gex_embed_attn = []
                for head in gex_embed:
                    head = head.view(self.tmp_batch_size, -1, args.gcn_hidden_dim1)
                    self.attn, head = drug2gex_attn_applied(drug_emb, head, self, args, training)
                    head = head.view(-1, args.gcn_hidden_dim1)
                    gex_embed_attn.append(head)
                gex_embed = gex_embed_attn
                """
                #   #   #   #
#                gex_embed = torch.stack(gex_embed, dim = -1)
                gex_embed = torch.cat(gex_embed, dim = -1)  #bs * num nodes x 256

            #   append each readouts
            """
            read_out = torch.mean(gex_embed.view(self.tmp_batch_size, -1, 
                        args.gcn_hidden_out*args.gat_num_heads), dim = 1) # bs x  256
            gcn_cat_list.append(read_out)
            """
            gcn_cat_list.append(gex_embed.view(self.tmp_batch_size, -1, 
                        args.gcn_hidden_out*args.gat_num_heads)) # bs x n nodes x 256


        #   Readout + view batchwise

        if args.sort_pool_k == 0:
#            batch_output = gex_embed.view(
#                                self.tmp_batch_size, -1, args.gcn_hidden_out*args.gat_num_heads)
            ####
#            self.attn, batch_output = drug2gex_attn_applied(drug_emb, batch_output, self, args, training)
            ####
#            read_out = torch.mean(batch_output, dim = 1)
            read_out = torch.cat(gcn_cat_list, dim = -1) #  bs x n nodes x 728
            read_out = read_out.transpose(2, 1) #   bs x 728 x n nodes
            read_out = F.dropout(self.activ(self.readout_mlp1(read_out)), training = training)
            read_out = F.dropout(self.activ(self.readout_mlp2(read_out)), training = training)
            read_out = F.dropout(self.activ(self.readout_mlp3(read_out)), training = training).squeeze(-1)


#        total_emb = torch.cat([drug_emb, read_out, dose, duration], dim = -1) 
        total_emb = torch.cat([read_out, dose, duration], dim = -1) 
        
#        proba = self.pred_mlp(total_emb)

        total_emb = F.dropout(self.activ(self.pred_mlp1(total_emb)), training = training)
        total_emb = F.dropout(self.activ(self.pred_mlp2(total_emb)), training = training)
        proba = self.pred_mlp3(total_emb)
        
        return proba

class GEX_drug_attn_ver2_MLP(nn.Module):
    def __init__(self, g2v_embedding, args):
        super(GEX_drug_attn_ver2_MLP, self).__init__()


        print('num genes : '+str(args.num_genes))
        self.bn1 = nn.BatchNorm1d(args.ecfp_nBits)
        self.bn2 = nn.BatchNorm1d(args.num_genes)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn4 = nn.BatchNorm1d(1)

        """
        self.drug_mlp = nn.Linear(args.resimnet_dim, drug_embed_dim, bias = True)
        self.drug_mlp1 = nn.Linear(args.resimnet_dim, 
                    int(0.5*args.resimnet_dim + 0.5*drug_embed_dim), bias = True)
        self.drug_mlp2 = nn.Linear(int(0.5*args.resimnet_dim+0.5*drug_embed_dim),
                    drug_embed_dim, bias = True)
        """
        self.drug_mlp1 = nn.Linear(args.resimnet_dim, 
                    int(args.resimnet_dim*2/3 + args.drug_embed_dim/3), bias = True)
        self.drug_mlp2 = nn.Linear(int(args.resimnet_dim*2/3 + args.drug_embed_dim/3),
                    int(args.resimnet_dim/3+args.drug_embed_dim*2/3), bias = True)
        self.drug_mlp3 = nn.Linear(int(args.resimnet_dim/3+args.drug_embed_dim*2/3),
                    args.drug_embed_dim, bias = True)
                    
        self.g2v_mlp1 = nn.Linear(args.gene2vec_dim, 150, bias = True)
        self.g2v_mlp2 = nn.Linear(150, 100, bias = True)
        self.g2v_mlp3 = nn.Linear(100, 1, bias = True)

#        self.pred_emb_dim = args.drug_embed_dim + args.gcn_hidden_out + 2
#        self.pred_emb_dim = args.drug_embed_dim + args.gene2vec_dim + 2
        self.pred_emb_dim = args.drug_embed_dim + args.num_genes + 2

#        self.pred_mlp = nn.Linear(self.pred_emb_dim, num_classes, bias = True)
        self.pred_mlp1 = nn.Linear(self.pred_emb_dim, int(self.pred_emb_dim*2/3), bias = True)
        self.pred_mlp2 = nn.Linear(int(self.pred_emb_dim*2/3), int(self.pred_emb_dim/3), bias = True)
        self.pred_mlp3 = nn.Linear(int(self.pred_emb_dim/3), args.num_classes, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()


        #   
        self.g2v_embeddings = nn.Embedding(args.num_genes, args.gene2vec_dim)
#        self.g2v_embeddings.weight.data.copy_(torch.from_numpy(g2v_embedding))
        if args.g2v_pretrained == True:
            g2v_embedding = F.normalize(torch.from_numpy(g2v_embedding), p = 2)
            self.g2v_embeddings.weight.data.copy_(g2v_embedding)

        if args.attn_dim != 0 :
            #   Attention layers
            self.query_proj = nn.Linear(args.drug_embed_dim, args.attn_dim, bias = True)
#        self.key_proj = nn.Linear(args.gex_embed_dim, args.attn_dim, bias = True)
            self.key_proj = nn.Linear(args.gene2vec_dim, args.attn_dim, bias = True)
            #   Additive attention
            self.attn_proj = nn.Linear(args.attn_dim, 1, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()
        self.activ3 = nn.Tanh()

    def forward(self, x, get_gex_idxs, device, args, training = True):
        # x[0] : ecfp, x[1] : gex, x[2] : dosage, x[3] : duration

        self.tmp_batch_size = x[4].shape[0]
        drug_input = x[8].float().to(device)

#        gex_input = self.bn2(x[1][:, get_gex_idxs].float().to(device)).unsqueeze(-1) # bs x n genes x 1
        gex_input = x[1][:, get_gex_idxs].float().to(device) # bs x n genes x 1
        dose = x[2].float().to(device).view(-1,1)
        duration = x[3].float().to(device).view(-1,1)
        try:
            drug_input = self.bn1(drug_input)
            gex_input = self.bn2(gex_input).unsqueeze(-1)
            dose = self.bn3(dose)
            duration = self.bn4(duration)
        except ValueError:
            gex_input = gex_input.unsqueeze(-1)


        drug_input = F.dropout(drug_input, training = training)
        gex_input = F.dropout(gex_input, training = training)

        drug_emb = F.dropout(self.activ(self.drug_mlp1(drug_input)), training =training)
        drug_emb = F.dropout(self.activ(self.drug_mlp2(drug_emb)), training =training)
        drug_emb = F.dropout(self.activ(self.drug_mlp3(drug_emb)), training =training)
        
        gene2vec = self.g2v_embeddings(torch.tensor([i for i in range(args.num_genes)]).to(device)) # n_genes x vec dims

        #   Get attn btw each gene & drug
        gene2vec = gene2vec.unsqueeze(0).repeat(self.tmp_batch_size, 1, 1)

        if args.attn_dim != 0 :
            self.attn, gene2vec = drug2gex_attn_applied(drug_emb, gene2vec, self, args, training)

        gex_embed = gex_input * gene2vec # bs x n genes x vec dims

        #   MLP instead of Read out
        gex_embed = F.dropout(self.activ(self.g2v_mlp1(gex_embed)), training =training)
        gex_embed = F.dropout(self.activ(self.g2v_mlp2(gex_embed)), training =training)
        read_out = F.dropout(self.activ(self.g2v_mlp3(gex_embed)), training =training).squeeze(-1)

        #   Readout + view batchwise

#        read_out = torch.mean(gex_embed, dim = 1)
        

        total_emb = torch.cat([drug_emb, read_out, dose, duration], dim = -1) 
        
#        proba = self.pred_mlp(total_emb)

        total_emb = F.dropout(self.activ(self.pred_mlp1(total_emb)), training = training)
        total_emb = F.dropout(self.activ(self.pred_mlp2(total_emb)), training = training)
        proba = self.pred_mlp3(total_emb)
        
        return proba

class GEX_drug_attn_ver3_MLP(nn.Module):    #   gene wise mlp first
    def __init__(self, g2v_embedding, args):
        super(GEX_drug_attn_ver3_MLP, self).__init__()


        print('num genes : '+str(args.num_genes))
        self.bn1 = nn.BatchNorm1d(args.ecfp_nBits)
        self.bn2 = nn.BatchNorm1d(args.num_genes)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn4 = nn.BatchNorm1d(1)

        """
        self.drug_mlp = nn.Linear(args.resimnet_dim, drug_embed_dim, bias = True)
        self.drug_mlp1 = nn.Linear(args.resimnet_dim, 
                    int(0.5*args.resimnet_dim + 0.5*drug_embed_dim), bias = True)
        self.drug_mlp2 = nn.Linear(int(0.5*args.resimnet_dim+0.5*drug_embed_dim),
                    drug_embed_dim, bias = True)
        """
        self.drug_mlp1 = nn.Linear(args.resimnet_dim, 
                    int(args.resimnet_dim*2/3 + args.drug_embed_dim/3), bias = True)
        self.drug_mlp2 = nn.Linear(int(args.resimnet_dim*2/3 + args.drug_embed_dim/3),
                    int(args.resimnet_dim/3+args.drug_embed_dim*2/3), bias = True)
        self.drug_mlp3 = nn.Linear(int(args.resimnet_dim/3+args.drug_embed_dim*2/3),
                    args.drug_embed_dim, bias = True)
                    
        """
        self.g2v_mlp1 = nn.Linear(args.gene2vec_dim, 150, bias = True)
        self.g2v_mlp2 = nn.Linear(150, 100, bias = True)
        self.g2v_mlp3 = nn.Linear(100, 1, bias = True)
        """
        self.readout_mlp1 = nn.Linear(args.gene2vec_dim, int(args.gene2vec_dim*2/3), bias = True)
        self.readout_mlp2 = nn.Linear(int(args.gene2vec_dim*2/3), int(args.gene2vec_dim/3), bias = True)
        self.readout_mlp3 = nn.Linear(int(args.gene2vec_dim/3), 1, bias = True)

#        self.pred_emb_dim = args.drug_embed_dim + args.gcn_hidden_out + 2
#        self.pred_emb_dim = args.drug_embed_dim + args.gene2vec_dim + 2
#        self.pred_emb_dim = args.drug_embed_dim + args.num_genes + 2
        self.pred_emb_dim = args.drug_embed_dim + args.gene2vec_dim + 2


#        self.pred_mlp = nn.Linear(self.pred_emb_dim, num_classes, bias = True)
        self.pred_mlp1 = nn.Linear(self.pred_emb_dim, int(self.pred_emb_dim*2/3), bias = True)
        self.pred_mlp2 = nn.Linear(int(self.pred_emb_dim*2/3), int(self.pred_emb_dim/3), bias = True)
        self.pred_mlp3 = nn.Linear(int(self.pred_emb_dim/3), args.num_classes, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()


        #   
        self.g2v_embeddings = nn.Embedding(args.num_genes, args.gene2vec_dim)
#        self.g2v_embeddings.weight.data.copy_(torch.from_numpy(g2v_embedding))
        if args.g2v_pretrained == True:
            g2v_embedding = F.normalize(torch.from_numpy(g2v_embedding), p = 2)
            self.g2v_embeddings.weight.data.copy_(g2v_embedding)

        if args.attn_dim != 0 :
            #   Attention layers
            self.query_proj = nn.Linear(args.drug_embed_dim, args.attn_dim, bias = True)
#        self.key_proj = nn.Linear(args.gex_embed_dim, args.attn_dim, bias = True)
            self.key_proj = nn.Linear(args.gene2vec_dim, args.attn_dim, bias = True)
            #   Additive attention
            self.attn_proj = nn.Linear(args.attn_dim, 1, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()
        self.activ3 = nn.Tanh()




    def forward(self, x, get_gex_idxs, device, args, training = True):
        # x[0] : ecfp, x[1] : gex, x[2] : dosage, x[3] : duration

        self.tmp_batch_size = x[4].shape[0]
        drug_input = x[8].float().to(device)

#        gex_input = self.bn2(x[1][:, get_gex_idxs].float().to(device)).unsqueeze(-1) # bs x n genes x 1
        gex_input = x[1][:, get_gex_idxs].float().to(device) # bs x n genes x 1
        dose = x[2].float().to(device).view(-1,1)
        duration = x[3].float().to(device).view(-1,1)
        try:
            drug_input = self.bn1(drug_input)
            gex_input = self.bn2(gex_input).unsqueeze(-1)
            dose = self.bn3(dose)
            duration = self.bn4(duration)
        except ValueError:
            gex_input = gex_input.unsqueeze(-1)


        drug_input = F.dropout(drug_input, training = training)
        gex_input = F.dropout(gex_input, training = training)

        drug_emb = F.dropout(self.activ(self.drug_mlp1(drug_input)), training =training)
        drug_emb = F.dropout(self.activ(self.drug_mlp2(drug_emb)), training =training)
        drug_emb = F.dropout(self.activ(self.drug_mlp3(drug_emb)), training =training)
        
        gene2vec = self.g2v_embeddings(torch.tensor([i for i in range(args.num_genes)]).to(device)) # n_genes x vec dims

        #   Get attn btw each gene & drug
        gene2vec = gene2vec.unsqueeze(0).repeat(self.tmp_batch_size, 1, 1)

        #   #   #   #

        if args.attn_dim != 0 :
            self.attn, gene2vec = drug2gex_attn_applied(drug_emb, gene2vec, self, args, training)

        gex_embed = gex_input * gene2vec # bs x n genes x vec dims
        #   #   #   #

        #   MLP instead of Read out
        gex_embed = gex_embed.transpose(0, 2, 1)
        gex_embed = F.dropout(self.activ(self.readout_mlp1(gex_embed)), training =training)
        gex_embed = F.dropout(self.activ(self.readout_mlp2(gex_embed)), training =training)
        read_out = F.dropout(self.activ(self.readout_mlp3(gex_embed)), training =training).squeeze(-1)
        

        """
        gex_embed = F.dropout(self.activ(self.g2v_mlp1(gex_embed)), training =training)
        gex_embed = F.dropout(self.activ(self.g2v_mlp2(gex_embed)), training =training)
        read_out = F.dropout(self.activ(self.g2v_mlp3(gex_embed)), training =training).squeeze(-1)
        """

        #   Readout + view batchwise

#        read_out = torch.mean(gex_embed, dim = 1)
        

        total_emb = torch.cat([drug_emb, read_out, dose, duration], dim = -1) 
        
#        proba = self.pred_mlp(total_emb)

        total_emb = F.dropout(self.activ(self.pred_mlp1(total_emb)), training = training)
        total_emb = F.dropout(self.activ(self.pred_mlp2(total_emb)), training = training)
        proba = self.pred_mlp3(total_emb)
        
        return proba


class GEX_drug_attn_MLP(nn.Module):
    def __init__(self, g2v_embedding, drug_embed_dim, gex_embed_dim, num_classes, args):
        super(GEX_drug_attn_MLP, self).__init__()

        self.g2v_embedding = torch.tensor(g2v_embedding, dtype = torch.float, 
                                        requires_grad = True).unsqueeze(0)

        self.bn1 = nn.BatchNorm1d(args.resimnet_dim)
        self.bn2 = nn.BatchNorm1d(args.num_genes)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn4 = nn.BatchNorm1d(1)

        self.drug_mlp1 = nn.Linear(args.resimnet_dim, 512, bias = True)
        self.drug_mlp2 = nn.Linear(512, drug_embed_dim, bias = True)

        self.pred_emb_dim = args.drug_embed_dim + 200 + 2

        self.pred_mlp1 = nn.Linear(self.pred_emb_dim, 100, bias = True)
        self.pred_mlp2 = nn.Linear(100, 50, bias = True)
        self.pred_mlp3 = nn.Linear(50, num_classes, bias = True)
                

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()
        self.activ3 = nn.Tanh()

        #   Attention layers
        self.query_proj = nn.Linear(args.drug_embed_dim, args.attn_dim, bias = True)
#        self.query_proj = nn.Linear(args.drug_embed_dim, args.gene2vec_dim, bias = True)
        self.key_proj = nn.Linear(args.gex_embed_dim, args.attn_dim, bias = True)
       


    def forward(self, x, gene2vec, get_gex_idxs, device, args, training = True):
        # x[0] : ecfp, x[1] : gex, x[2] : dosage, x[3] : duration

        self.tmp_batch_size = x[4].shape[0]

        drug_input = x[8].float().to(device)
        #torch.as_tensor(exp.float()).detach().to(device
        gex_input = x[1][:, get_gex_idxs].float().to(device) # bs x n genes x 1
        dose = x[2].float().to(device).view(-1,1)
        duration = x[3].float().to(device).view(-1,1)
        try:
            drug_input = self.bn1(drug_input)
            gex_input = self.bn2(gex_input).unsqueeze(-1)

            dose = self.bn3(dose)
            duration = self.bn4(duration)
        except ValueError:
            gex_input = gex_input.unsqueeze(-1)
#        drug_input = self.bn1(x[0].float().to(device))

        #   dropout for inputs
        drug_input = F.dropout(drug_input, training = training)
        gex_input = F.dropout(gex_input, training = training)

        drug_emb = F.dropout(self.drug_mlp2(
                    self.activ(self.drug_mlp1(drug_input))), training = training) 

        gene2vec = self.g2v_embedding.float().to(device) # 1 x n genes x vec dims

        gex_emb = gex_input * gene2vec # bs x n genes x vec dims

#        gex_embed = gex_embed.view(-1, args.gene2vec_dim) # bs*n genex x vec dims

        
        def drug2gex_attn_applied(drug_feat, gex_feat, self, args, training):

            #   drug input & gex input requires same dimension
            drug_proj = F.sigmoid(self.query_proj(drug_feat))
            #   bs x 200
            gex_proj = F.sigmoid(self.key_proj(gex_feat))
            #   bs x num genes x 200

            drug_proj  = drug_proj.unsqueeze(1)
            gex_proj = gex_proj.transpose(2,1)
            attn = torch.bmm(drug_proj, gex_proj).squeeze(1) # bs x num_genes 
            attn = F.softmax(attn, dim = -1)
#            attn = F.sigmoid(attn)
            
#            attn = F.tanh(attn)

#            print(attn[0])


            gex_attned = torch.mul(attn.unsqueeze(-1), gex_feat) # bs x num_genes x 200
            
            return attn, gex_attned
         

#        gex_emb = drug2gex_attn_applied(drug_emb, gex_emb, self, args, training)
       
#        read_out = torch.mean(gex_emb, dim = 1)


#        ver 5
        #   drug input & gex input requires same dimension

        attn, _ = drug2gex_attn_applied(drug_emb, gex_emb, self, args, training) # bs x n genes
        self.attn = attn
        read_out = torch.mul(attn.unsqueeze(-1), gex_emb) # bs x n genes x 200 dim
        read_out = torch.mean(read_out, dim = 1) # bs x 200


        total_emb = torch.cat([drug_emb, read_out, dose, duration], dim = -1) 
        
#        proba = self.pred_mlp(total_emb)
        total_emb = F.dropout(self.activ(self.pred_mlp1(total_emb)))
        total_emb = F.dropout(self.activ(self.pred_mlp2(total_emb)))
        proba = self.pred_mlp3(total_emb)

        return proba

"""










class Drug_Pretrain(nn.Module):
    def __init__(self, drug_embed_dim, num_classes, args):
        super(Drug_Pretrain, self).__init__()

        self.bn1 = nn.BatchNorm1d(args.ecfp_nBits)

        self.drug_mlp1 = nn.Linear(args.resimnet_dim, 512, bias = True)
        self.drug_mlp2 = nn.Linear(512, drug_embed_dim, bias = True)
               
        self.pred_mlp1 = nn.Linear(drug_embed_dim, 128, bias = True)
        self.pred_mlp2 = nn.Linear(128, num_classes, bias = True)

        self.activ = nn.ReLU()

    def forward(self, x, device, args, training = True):

        drug_input = self.bn1(x.float().to(device))

        self.drug_hidden = F.dropout(self.drug_mlp2(
                            F.dropout(self.activ(self.drug_mlp1(drug_input)),
                                training = training)),
                                    training = training)
        pred = self.pred_mlp2(F.dropout(self.activ(self.pred_mlp1(self.drug_hidden)),
                                training = training))

        return pred

        

class GEX_PPI_MLP(nn.Module):
    def __init__(self, ppi_adj, g2v_embedding, drug_embed_dim, gex_embed_dim, num_classes, args):
        super(GEX_PPI_MLP, self).__init__()

#        self.gex_mlp = nn.Linear(args.num_genes, gex_embed_dim, bias = True)

        self.bn1 = nn.BatchNorm1d(args.ecfp_nBits)
        self.bn2 = nn.BatchNorm1d(args.num_genes)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn4 = nn.BatchNorm1d(1)

        self.drug_mlp = nn.Linear(args.ecfp_nBits, drug_embed_dim, bias = True)
        
        self.gcns = []
        for i in range(args.num_gcn_hops):
            if i==0:
                self.gcns.append(geo_nn.GCNConv(args.gene2vec_dim, args.gcn_hidden_dim1,
                                        heads = args.gat_num_heads))
            else:
                self.gcns.append(geo_nn.GCNConv(args.gcn_hidden_dim1, args.gcn_hidden_dim2,
                                        heads = args.gat_num_heads))
        for i in range(args.num_gcn_hops):
            self.add_module('gcn_{}'.format(i), self.gcns[i])

        self.pred_emb_dim = args.drug_embed_dim + args.gcn_hidden_out + 2

        self.pred_mlp = nn.Linear(self.pred_emb_dim, num_classes, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()


        #   
        #self.g2v_embeddings = nn.Parameters(torch.tensor(g2v_embeddings))
        self.g2v_embedding = torch.tensor(g2v_embedding).unsqueeze(0)
#        self.ppi_adj = ppi_adj
        self.ppi_adj = ppi_adj # 2 x num edges
        print(self.ppi_adj.shape)

        #   Attention layers
        self.query_proj = nn.Linear(args.drug_embed_dim, args.attn_dim, bias = True)
        self.key_proj = nn.Linear(args.gex_embed_dim, args.attn_dim, bias = True)


    def forward(self, x, adj, gene2vec, get_gex_idxs, device, args, training = True):
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
    
        #torch.as_tensor(exp.float()).detach().to(device

#        drug_emb = self.drug_mlp(drug_input) 
        drug_emb = F.dropout(self.activ(self.drug_mlp(drug_input)), training =training)


        gene2vec = self.g2v_embedding.float().to(device) # 1 x n genes x vec dims

        gex_embed = gex_input * gene2vec # bs x n genes x vec dims

        gex_embed = gex_embed.view(-1, args.gene2vec_dim) # bs*n genex x vec dims

        #   batch index for sortpooling
        batch_idx = np.hstack(
                    [[i for _ in range(gene2vec.shape[1])] for i in range(self.tmp_batch_size)])
        batch_idx = torch.tensor(batch_idx).to(device)

#        ppi_adj = torch.tensor(self.ppi_adj).unsqueeze(0) # 1 x n genes x n genes

        id_mat = get_batch_adj(self.id_mat, self.tmp_batch_size, args)
        ppi_adj = get_batch_adj(self.ppi_adj, self.tmp_batch_size, args)
        id_mat = torch.tensor(id_mat, dtype = torch.long).to(device)
        ppi_adj = torch.tensor(ppi_adj, dtype = torch.long).to(device)


        for i in range(args.num_gcn_hops):

            if i != 0:
                gex_embed_0 = gex_embed
            gex_embed = F.dropout(self.activ(
                                        self.gcns[i](
                                            gex_embed, 
                                            ppi_adj)
                                        ), training = training)

            if args.gat_num_heads >1 :
                #   bs x num nodes x gcn dim * gat num heads
                gex_embed = torch.split(gex_embed,
                                        split_size_or_sections = args.gcn_hidden_dim1,
                                        dim = -1)
                gex_embed = torch.mean(torch.stack(
                                        gex_embed, dim = -1), dim = -1)

            if i!=0:
                gex_embed += gex_embed_0

        #   Readout + view batchwise

        if args.sort_pool_k == 0:
            batch_output = gex_embed.view(
                                self.tmp_batch_size, -1, args.gcn_hidden_out)
            ####
            self.attn, batch_output = drug2gex_attn_applied(drug_emb, batch_output, self, args, training)
            ####
            read_out = torch.mean(batch_output, dim = 1)
        

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
        
        proba = self.pred_mlp(total_emb)
        
        return proba


class GEX_kegg_GAT_MLP(nn.Module):
    def __init__(self, ppi_adj, g2v_embeddings, drug_embed_dim, gex_embed_dim, num_classes, args):
        super(GEX_kegg_GAT_MLP, self).__init__()

#        self.gex_mlp = nn.Linear(args.num_genes, gex_embed_dim, bias = True)

#        self.bn1 = nn.BatchNorm1d(args.ecfp_nBits)
        self.bn1 = nn.BatchNorm1d(args.resimnet_dim)
        self.bn2 = nn.BatchNorm1d(args.num_genes)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn4 = nn.BatchNorm1d(1)

        self.drug_mlp = nn.Linear(args.resimnet_dim, drug_embed_dim, bias = True)
        
#       self.gcn1 = geo_nn.GCNConv(args.smiles_emb_dim, args.gcn_hidden_dim1)


        self.gats_pos = []
        self.gats_neg = []
        for i in range(args.num_gcn_hops):
            if i==0:
                self.gats_pos.append(geo_nn.GATConv(args.gene2vec_dim, args.gcn_hidden_dim1,
                                        heads = args.gat_num_heads))
                self.gats_neg.append(geo_nn.GATConv(args.gene2vec_dim, args.gcn_hidden_dim1,
                                        heads = args.gat_num_heads))
            else:
                self.gats_pos.append(geo_nn.GATConv(args.gcn_hidden_dim1, args.gcn_hidden_dim2,
                                        heads = args.gat_num_heads))
                self.gats_neg.append(geo_nn.GATConv(args.gcn_hidden_dim1, args.gcn_hidden_dim2,
                                        heads = args.gat_num_heads))
        for i in range(args.num_gcn_hops):
            self.add_module('gat_pos_{}'.format(i), self.gats_pos[i])
            self.add_module('gat_neg_{}'.format(i), self.gats_neg[i])


        self.pred_emb_dim = args.drug_embed_dim + args.gcn_hidden_out + 2

        self.pred_mlp = nn.Linear(self.pred_emb_dim, num_classes, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()


        #   
        self.g2v_embeddings = nn.Parameter(torch.tensor(g2v_embeddings, dtype = torch.float32).cuda(),
                                            requires_grad = True).unsqueeze(0)
#        self.register_parameter('g2v_embeddings', self.g2v_embeddings)
#        self.add_module('g2v_embeddings', self.g2v_embeddings)
        
        #self.g2v_embedding = torch.tensor(g2v_embedding).unsqueeze(0)
        self.ppi_adj_pos = ppi_adj[0]
        self.ppi_adj_neg = ppi_adj[1]

        #   Attention layers
        self.query_proj = nn.Linear(args.drug_embed_dim, args.attn_dim, bias = True)
        self.key_proj = nn.Linear(args.gex_embed_dim, args.attn_dim, bias = True)


    def forward(self, x, adj, gene2vec, get_gex_idxs, device, args, training = True):
        # x[0] : ecfp, x[1] : gex, x[2] : dosage, x[3] : duration
        # x[4] : label , x[5] : drug name , x[6] : cell_line name, x[7] : smiles
        # x[8] : Resimnet Feature

        # no eye, 2 layer

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
            
        #   dropout for inputs
        drug_input = F.dropout(drug_input, training = training)
        gex_input = F.dropout(gex_input, training = training)

        drug_emb = F.dropout(self.activ(self.drug_mlp(drug_input)), training =training)
#        drug_emb = F.dropout(self.drug_mlp(drug_input), training = training)

#        gene2vec = self.g2v_embeddings.float().to(device) # 1 x n genes x vec dims
        gene2vec = self.g2v_embeddings

        gex_embed = gex_input * gene2vec # bs x n genes x vec dims
        self.gex_embed = gex_embed

        gex_embed = gex_embed.view(-1, args.gene2vec_dim) # bs*n genex x vec dims

        #   batch index for sortpooling
        #   batch index for sortpooling
        batch_idx = np.hstack(
                    [[i for _ in range(gene2vec.shape[1])] for i in range(self.tmp_batch_size)])
        batch_idx = torch.tensor(batch_idx).to(device)



#        id_mat = get_batch_adj(self.id_mat, self.tmp_batch_size)
        ppi_adj_pos = get_batch_adj(self.ppi_adj_pos, self.tmp_batch_size, args)
        ppi_adj_neg = get_batch_adj(self.ppi_adj_neg, self.tmp_batch_size, args)
        
#        id_mat = torch.tensor(id_mat, dtype = torch.long).to(device)
        ppi_adj_pos = torch.tensor(ppi_adj_pos, dtype = torch.long).to(device)
        ppi_adj_neg = torch.tensor(ppi_adj_neg, dtype = torch.long).to(device)

        ####
        for i in range(args.num_gcn_hops):
            if i == 0:
                gex_embed_pos = gex_embed
                gex_embed_neg = gex_embed
            else:
                gex_embed_0_pos = gex_embed_pos
                gex_embed_0_neg = gex_embed_neg
            

            gex_embed_pos = F.dropout(self.activ(
                                        self.gats_pos[i](
                                            gex_embed_pos, 
                                            ppi_adj_pos)
                                        ), training = training)
            gex_embed_neg = F.dropout(self.activ(
                                        self.gats_neg[i](
                                            gex_embed_neg, 
                                            ppi_adj_neg)
                                        ), training = training)

            if args.gat_num_heads >1 :
                #   bs x num nodes x gcn dim * gat num heads
                gex_embed_pos = torch.split(gex_embed_pos,
                                        split_size_or_sections = args.gcn_hidden_dim1,
                                        dim = -1)
                gex_embed_pos = torch.mean(torch.stack(
                                        gex_embed_pos, dim = -1), dim = -1)
                gex_embed_neg = torch.split(gex_embed_neg,
                                        split_size_or_sections = args.gcn_hidden_dim1,
                                        dim = -1)
                gex_embed_neg = torch.mean(torch.stack(
                                        gex_embed_neg, dim = -1), dim = -1)

            if i!=0:
                gex_embed_pos += gex_embed_0_pos
                gex_embed_neg += gex_embed_0_neg
        ####
        #   Readout + view batchwise


        if args.sort_pool_k == 0.0:
            batch_output_pos = gex_embed_pos.view(self.tmp_batch_size, -1, args.gcn_hidden_out)

            batch_output_neg = gex_embed_neg.view(self.tmp_batch_size, -1, args.gcn_hidden_out)
            ###
            attn_pos, batch_output_pos = drug2gex_attn_applied(
                                    drug_emb, batch_output_pos, self, args, training)
            attn_neg, batch_output_neg = drug2gex_attn_applied(
                                    drug_emb, batch_output_neg, self, args, training)
            self.attn = torch.cat([attn_pos, attn_neg], dim = -1)

            ###
            read_out = torch.mean(torch.cat(
                                [batch_output_pos, batch_output_neg], dim = 1), dim = 1)
            batch_output = torch.cat(
                            [gex_embed_pos.view(
                                self.tmp_batch_size, -1, args.gcn_hidden_out),
                            gex_embed_neg.view(
                                self.tmp_batch_size, -1, args.gcn_hidden_out)],
                            dim = 1)

            read_out = torch.mean(batch_output, dim = 1)
        

        #   sort pooling
        else:
            sort_pool_k = int(np.ceil(args.num_genes * args.sort_pool_k))
            batch_output_pos = geo_nn.global_sort_pool(
                                gex_embed_pos, batch_idx, sort_pool_k)
            batch_output_pos = batch_output_pos.view(self.tmp_batch_size, -1, args.gcn_hidden_out)

            batch_output_neg = geo_nn.global_sort_pool(
                                gex_embed_neg, batch_idx, sort_pool_k)
            batch_output_neg = batch_output_neg.view(self.tmp_batch_size, -1, args.gcn_hidden_out)

            ###
            attn_pos, batch_output_pos = drug2gex_attn_applied(
                                    drug_emb, batch_output_pos, self, args, training)
            attn_neg, batch_output_neg = drug2gex_attn_applied(
                                    drug_emb, batch_output_neg, self, args, training)
            self.attn = torch.cat([attn_pos, attn_neg], dim = -1)

            ###
            read_out = torch.mean(torch.cat(
                                [batch_output_pos, batch_output_neg], dim = 1), dim = 1)


        total_emb = torch.cat([drug_emb, read_out, dose, duration], dim = -1) 
        
        proba = self.pred_mlp(total_emb)
#        total_emb = F.dropout(self.activ(self.pred_mlp1(total_emb)))
#        total_emb = F.dropout(self.activ(self.pred_mlp2(total_emb)))
#        proba = self.pred_mlp3(total_emb)
        
        return proba


class GEX_kegg_MLP(nn.Module):
    def __init__(self, ppi_adj, g2v_embedding, drug_embed_dim, gex_embed_dim, num_classes, args):
        super(GEX_kegg_MLP, self).__init__()

#        self.gex_mlp = nn.Linear(args.num_genes, gex_embed_dim, bias = True)

        self.bn1 = nn.BatchNorm1d(args.ecfp_nBits)
        self.bn2 = nn.BatchNorm1d(args.num_genes)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn4 = nn.BatchNorm1d(1)

        self.drug_mlp = nn.Linear(args.ecfp_nBits, drug_embed_dim, bias = True)
        
        self.gcns_pos = []
        self.gcns_neg = []
        for i in range(args.num_gcn_hops):
            if i==0:
                self.gcns_pos.append(geo_nn.GCNConv(args.gene2vec_dim, args.gcn_hidden_dim1,
                                        heads = args.gat_num_heads))
                self.gcns_neg.append(geo_nn.GCNConv(args.gene2vec_dim, args.gcn_hidden_dim1,
                                        heads = args.gat_num_heads))
            else:
                self.gcns_pos.append(geo_nn.GCNConv(args.gcn_hidden_dim1, args.gcn_hidden_dim2,
                                        heads = args.gat_num_heads))
                self.gcns_neg.append(geo_nn.GCNConv(args.gcn_hidden_dim1, args.gcn_hidden_dim2,
                                        heads = args.gat_num_heads))
        for i in range(args.num_gcn_hops):
            self.add_module('gcn_pos_{}'.format(i), self.gcns_pos[i])
            self.add_module('gcn_neg_{}'.format(i), self.gcns_neg[i])

        self.pred_emb_dim = args.drug_embed_dim + args.gcn_hidden_out + 2

        self.pred_mlp = nn.Linear(self.pred_emb_dim, num_classes, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()


        #   
        #self.g2v_embeddings = nn.Parameters(torch.tensor(g2v_embeddings))
        self.g2v_embedding = torch.tensor(g2v_embedding).unsqueeze(0)
        self.ppi_adj_pos = ppi_adj[0]
        self.ppi_adj_neg = ppi_adj[1]

        #   Attention layers
        self.query_proj = nn.Linear(args.drug_embed_dim, args.attn_dim, bias = True)
        self.key_proj = nn.Linear(args.gex_embed_dim, args.attn_dim, bias = True)


        self.id_mat = torch.tensor([sparse.coo_matrix(sparse.eye(args.num_genes)).row,
                                    sparse.coo_matrix(sparse.eye(args.num_genes)).col],
                                    dtype = torch.long
                                    )


    def forward(self, x, adj, gene2vec, get_gex_idxs, device, args, training = True):
        # x[0] : ecfp, x[1] : gex, x[2] : dosage, x[3] : duration
        # x[4] : label , x[5] : drug name , x[6] : cell_line name, x[7] : smiles
        # x[8] : Resimnet Feature

        # no eye, 2 layer

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
            
        #   dropout for inputs
        drug_input = F.dropout(drug_input, training = training)
        gex_input = F.dropout(gex_input, training = training)

#        drug_emb = self.drug_mlp(drug_input) 
        drug_emb = F.dropout(self.activ(self.drug_mlp(drug_input)), training =training)

        gene2vec = self.g2v_embedding.float().to(device) # 1 x n genes x vec dims

        gex_embed = gex_input * gene2vec # bs x n genes x vec dims

        gex_embed = gex_embed.view(-1, args.gene2vec_dim) # bs*n genex x vec dims

        #   batch index for sortpooling
        #   batch index for sortpooling
        batch_idx = np.hstack(
                    [[i for _ in range(gene2vec.shape[1])] for i in range(self.tmp_batch_size)])
        batch_idx = torch.tensor(batch_idx).to(device)

#        ppi_adj = torch.tensor(self.ppi_adj).unsqueeze(0) # 1 x n genes x n genes


        id_mat = get_batch_adj(self.id_mat, self.tmp_batch_size, args)
        ppi_adj_pos = get_batch_adj(self.ppi_adj_pos, self.tmp_batch_size, args)
        ppi_adj_neg = get_batch_adj(self.ppi_adj_neg, self.tmp_batch_size, args)
        
        id_mat = torch.tensor(id_mat, dtype = torch.long).to(device)
        ppi_adj_pos = torch.tensor(ppi_adj_pos, dtype = torch.long).to(device)
        ppi_adj_neg = torch.tensor(ppi_adj_neg, dtype = torch.long).to(device)

        ####
        for i in range(args.num_gcn_hops):
            if i == 0:
                gex_embed_pos = gex_embed
                gex_embed_neg = gex_embed
            else:
                gex_embed_0_pos = gex_embed_pos
                gex_embed_0_neg = gex_embed_neg
            

            gex_embed_pos = F.dropout(self.activ(
                                        self.gcns_pos[i](
                                            gex_embed_pos, 
                                            ppi_adj_pos)
                                        ), training = training)
            gex_embed_neg = F.dropout(self.activ(
                                        self.gcns_neg[i](
                                            gex_embed_neg, 
                                            ppi_adj_neg)
                                        ), training = training)

            if args.gat_num_heads >1 :
                #   bs x num nodes x gcn dim * gat num heads
                gex_embed_pos = torch.split(gex_embed_pos,
                                        split_size_or_sections = args.gcn_hidden_dim1,
                                        dim = -1)
                gex_embed_pos = torch.mean(torch.stack(
                                        gex_embed_pos, dim = -1), dim = -1)
                gex_embed_neg = torch.split(gex_embed_neg,
                                        split_size_or_sections = args.gcn_hidden_dim1,
                                        dim = -1)
                gex_embed_neg = torch.mean(torch.stack(
                                        gex_embed_neg, dim = -1), dim = -1)

            if i!=0:
                gex_embed_pos += gex_embed_0_pos
                gex_embed_neg += gex_embed_0_neg
        ####

        #   Readout + view batchwise

        if args.sort_pool_k == 0.0:
            batch_output_pos = gex_embed_pos.view(self.tmp_batch_size, -1, args.gcn_hidden_out)

            batch_output_neg = gex_embed_neg.view(self.tmp_batch_size, -1, args.gcn_hidden_out)
            ###
            attn_pos, batch_output_pos = drug2gex_attn_applied(
                                    drug_emb, batch_output_pos, self, args, training)
            attn_neg, batch_output_neg = drug2gex_attn_applied(
                                    drug_emb, batch_output_neg, self, args, training)
            self.attn = torch.cat([attn_pos, attn_neg], dim = -1)

            ###
            read_out = torch.mean(torch.cat(
                                [batch_output_pos, batch_output_neg], dim = 1), dim = 1)
            batch_output = torch.cat(
                            [gex_embed_pos.view(
                                self.tmp_batch_size, -1, args.gcn_hidden_out),
                            gex_embed_neg.view(
                                self.tmp_batch_size, -1, args.gcn_hidden_out)],
                            dim = 1)

            read_out = torch.mean(batch_output, dim = 1)
        

        #   sort pooling
        else:
            sort_pool_k = int(np.ceil(args.num_genes * args.sort_pool_k))
            batch_output_pos = geo_nn.global_sort_pool(
                                gex_embed_pos, batch_idx, sort_pool_k)
            batch_output_pos = batch_output_pos.view(self.tmp_batch_size, -1, args.gcn_hidden_out)

            batch_output_neg = geo_nn.global_sort_pool(
                                gex_embed_neg, batch_idx, sort_pool_k)
            batch_output_neg = batch_output_neg.view(self.tmp_batch_size, -1, args.gcn_hidden_out)


            ###
            attn_pos, batch_output_pos = drug2gex_attn_applied(
                                    drug_emb, batch_output_pos, self, args, training)
            attn_neg, batch_output_neg = drug2gex_attn_applied(
                                    drug_emb, batch_output_neg, self, args, training)
            self.attn = torch.cat([attn_pos, attn_neg], dim =-1)

            ###
            read_out = torch.mean(torch.cat(
                                [batch_output_pos, batch_output_neg], dim = 1), dim = 1)


        total_emb = torch.cat([drug_emb, read_out, dose, duration], dim = -1) 
        
        proba = self.pred_mlp(total_emb)
        
        return proba


class GEX_PPI_GAT_cat2_MLP(nn.Module):
    def __init__(self, ppi_adj, g2v_embedding, args):
        super(GEX_PPI_GAT_cat2_MLP, self).__init__()

#        self.gex_mlp = nn.Linear(args.num_genes, gex_embed_dim, bias = True)

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


#        self.pred_emb_dim = args.drug_embed_dim + args.gcn_hidden_out + 2
        self.pred_emb_dim = args.drug_embed_dim + (args.gcn_hidden_dim1*args.gat_num_heads) + 2

#        self.pred_mlp = nn.Linear(self.pred_emb_dim, num_classes, bias = True)
        self.pred_mlp1 = nn.Linear(self.pred_emb_dim, int(self.pred_emb_dim*2/3), bias = True)
        self.pred_mlp2 = nn.Linear(int(self.pred_emb_dim*2/3), int(self.pred_emb_dim/3), bias = True)
        self.pred_mlp3 = nn.Linear(int(self.pred_emb_dim/3), args.num_classes, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()


        #   
        self.g2v_embeddings = nn.Embedding(args.num_genes, args.gene2vec_dim)
#        self.g2v_embeddings.weight.data.copy_(torch.from_numpy(g2v_embedding))
        g2v_embedding = F.normalize(torch.from_numpy(g2v_embedding), p = 2)
        self.g2v_embeddings.weight.data.copy_(g2v_embedding)


        self.ppi_adj = ppi_adj # 2 x num edges
        print(self.ppi_adj.shape)

        #   Attention layers
        if args.attn_dim != 0 :
            self.query_proj = nn.Linear(args.drug_embed_dim, args.attn_dim, bias = True)
#        self.key_proj = nn.Linear(args.gex_embed_dim, args.attn_dim, bias = True)
#            self.key_proj = nn.Linear(args.gene2vec_dim, args.attn_dim, bias = True)
            self.key_proj = nn.Linear(args.gcn_hidden_dim1, args.attn_dim, bias = True)

#            self.attn_proj = nn.Linear(args.attn_dim, 1, bias = True)


    def forward(self, x, adj, gene2vec, get_gex_idxs, device, args, training = True):
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
        drug_emb = F.dropout(self.activ(self.drug_mlp3(drug_emb)), training =training)
        
        gene2vec = self.g2v_embeddings(torch.tensor([i for i in range(args.num_genes)]).to(device))

        #   Get attn btw each gene & drug
        gene2vec = gene2vec.unsqueeze(0).repeat(self.tmp_batch_size, 1, 1)

#        if args.attn_dim != 0:
#            self.attn, gene2vec = drug2gex_attn_applied(drug_emb, gene2vec, self, args, training)
        #   #   #   #

        gex_embed = gex_input * gene2vec # bs x n genes x vec dims

        gex_embed = gex_embed.view(-1, args.gene2vec_dim) # bs*n genex x vec dims

        #   batch index for sortpooling
        batch_idx = np.hstack(
                    [[i for _ in range(gene2vec.shape[1])] for i in range(self.tmp_batch_size)])
        batch_idx = torch.tensor(batch_idx).to(device)

        ppi_adj = get_batch_adj(self.ppi_adj, self.tmp_batch_size, args)

        ppi_adj = torch.tensor(ppi_adj, dtype = torch.long).to(device)

        
        for i in range(args.num_gcn_hops):
            if i != 0:
                gex_embed_0 = gex_embed
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
                #   #   #   #
                #   apply attn
                gex_embed_attn = []
                for head in gex_embed:
                    head = head.view(self.tmp_batch_size, -1, args.gcn_hidden_dim1)
                    self.attn, head = drug2gex_attn_applied(drug_emb, head, self, args, training)
                    head = head.view(-1, args.gcn_hidden_dim1)
                    gex_embed_attn.append(head)
                gex_embed = gex_embed_attn
                #   #   #   #
#                gex_embed = torch.stack(gex_embed, dim = -1)
                gex_embed = torch.cat(gex_embed, dim = -1)

            if i!=0:
                gex_embed += gex_embed_0

        #   Readout + view batchwise

        if args.sort_pool_k == 0:
            batch_output = gex_embed.view(
                                self.tmp_batch_size, -1, args.gcn_hidden_out*args.gat_num_heads)
            ####
#            self.attn, batch_output = drug2gex_attn_applied(drug_emb, batch_output, self, args, training)
            ####
            read_out = torch.mean(batch_output, dim = 1)
        

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
        
#        proba = self.pred_mlp(total_emb)

        total_emb = F.dropout(self.activ(self.pred_mlp1(total_emb)), training = training)
        total_emb = F.dropout(self.activ(self.pred_mlp2(total_emb)), training = training)
        proba = self.pred_mlp3(total_emb)
        
        return proba

class GEX_drug_attn_MLP_multi(nn.Module):
    def __init__(self, g2v_embedding, drug_embed_dim, gex_embed_dim, num_classes, args):
        super(GEX_drug_attn_MLP_multi, self).__init__()

#        self.gex_mlp = nn.Linear(args.num_genes, gex_embed_dim, bias = True)
        self.g2v_embedding = torch.tensor(g2v_embedding).unsqueeze(0)
#        self.ppi_adj = ppi_adj
        args.gex_embed_dim = self.g2v_embedding.size()[-1]

        self.bn1 = nn.BatchNorm1d(args.ecfp_nBits)
        self.bn2 = nn.BatchNorm1d(args.num_genes)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn4 = nn.BatchNorm1d(1)

        self.drug_mlp = nn.Linear(args.ecfp_nBits, drug_embed_dim, bias = True)
        
        self.gex_mlp = nn.Linear(args.num_genes, gex_embed_dim, bias = True)

        self.read_out_mlp1 = nn.Linear(args.gex_embed_dim, 100)
        self.read_out_mlp2 = nn.Linear(100, 1)
        self.pred_emb_dim = args.drug_embed_dim + 200 + 2

        self.pred_mlp = nn.Linear(self.pred_emb_dim, num_classes, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()


        #   
        #self.g2v_embeddings = nn.Parameters(torch.tensor(g2v_embeddings))
        #   Attention layers
        self.query_proj = nn.Linear(args.drug_embed_dim, args.attn_dim, bias = True)
        self.key_proj = nn.Linear(args.gex_embed_dim, args.attn_dim, bias = True)
        self.add_proj = nn.Linear(args.attn_dim, args.num_multi_heads, bias = True)


    def forward(self, x, gene2vec, get_gex_idxs, device, args, training = True):
        # x[0] : ecfp, x[1] : gex, x[2] : dosage, x[3] : duration

        self.tmp_batch_size = x[4].shape[0]
        drug_input = x[8].float().to(device)

#        gex_input = self.bn2(x[1][:, get_gex_idxs].float().to(device)).unsqueeze(-1) # bs x n genes x 1
        gex_input = x[1][:, get_gex_idxs].float().to(device) # bs x n genes x 1
        dose = x[2].float().to(device).view(-1,1)
        duration = x[3].float().to(device).view(-1,1)
        try:
            drug_input = self.bn1(drug_input)
            gex_input = self.bn2(gex_input).unsqueeze(-1)
            dose = self.bn3(dose)
            duration = self.bn4(duration)
        except ValueError:
            gex_input = gex_input.unsqueeze(-1)

#        drug_emb = self.drug_mlp(drug_input) 
        drug_emb = F.dropout(self.activ(self.drug_mlp(drug_input)), training =training)


        gene2vec = self.g2v_embedding.float().to(device) # 1 x n genes x vec dims

        gex_emb = gex_input * gene2vec # bs x n genes x vec dims

        
        def drug2gex_add_attn_applied(drug_feat, gex_feat, self, args, training):

            #   drug input & gex input requires same dimension
            drug_proj = self.activ(self.query_proj(drug_feat))
            #   bs x 200
            gex_proj = self.activ(self.key_proj(gex_feat))
            #   bs x num genes x 200

            drug_proj  = drug_proj.unsqueeze(1).repeat(1, gex_proj.size()[1], 1) # bs x n genes x 200
            gex_proj = gex_proj
#            attn = torch.bmm(drug_proj, gex_proj).squeeze(1) # bs x num_genes 
            attn = torch.add(drug_proj, gex_proj) # bs x num_genes x 200
            attn = self.activ(self.add_proj(attn)) #bs x num_genes x num multi heads
            attn = F.softmax(attn, dim = 1)
#            print(attn[0])
            attn = torch.chunk(attn, args.num_multi_heads, dim=-1) # num heads * bs x num_genes x1
            attn = [x.repeat(1,1,args.gex_embed_dim) for x in attn]
            #   bs x num_genes x vec dims


            gex_attned = [torch.mul(x, gex_feat) for x in attn]
            # num multi heads * bs x num_genes x 200
                
            
            return attn, gex_attned#   num multi heads * list of bs x num_genes x vec dims
         

#        gex_emb = drug2gex_attn_applied(drug_emb, gex_emb, self, args, training)
       
#        read_out = torch.mean(gex_emb, dim = 1)


#        ver 5
        #   drug input & gex input requires same dimension

        attn, gex_attned = drug2gex_add_attn_applied(drug_emb, gex_emb, self, args, training) # bs x n genes
        self.attn = attn
#        read_out = torch.mul(attn.unsqueeze(-1), gex_emb) # bs x n genes x 200 dim
        read_out = torch.sum(torch.stack(gex_attned), dim=0)
        read_out = torch.mean(read_out, dim = 1) # bs x 200


        #   Dimension mismatch issue - needs configuration
        try:
            total_emb = torch.cat([drug_emb, read_out, dose, duration], dim = -1) 
        except:
            print(drug_emb.size())
            print(read_out.size())
            print(dose.size())
            print(duration.size())
        
        proba = self.pred_mlp(total_emb)

        return proba


class GEX_drug_attn_MLP(nn.Module):
    def __init__(self, g2v_embedding, drug_embed_dim, gex_embed_dim, num_classes, args):
        super(GEX_drug_attn_MLP, self).__init__()

        self.g2v_embedding = torch.tensor(g2v_embedding, dtype = torch.float, 
                                        requires_grad = True).unsqueeze(0)

        self.bn1 = nn.BatchNorm1d(args.resimnet_dim)
        self.bn2 = nn.BatchNorm1d(args.num_genes)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn4 = nn.BatchNorm1d(1)

        self.drug_mlp1 = nn.Linear(args.resimnet_dim, 512, bias = True)
        self.drug_mlp2 = nn.Linear(512, drug_embed_dim, bias = True)

        self.pred_emb_dim = args.drug_embed_dim + 200 + 2

        self.pred_mlp1 = nn.Linear(self.pred_emb_dim, 100, bias = True)
        self.pred_mlp2 = nn.Linear(100, 50, bias = True)
        self.pred_mlp3 = nn.Linear(50, num_classes, bias = True)
                

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()
        self.activ3 = nn.Tanh()

        #   Attention layers
        self.query_proj = nn.Linear(args.drug_embed_dim, args.attn_dim, bias = True)
#        self.query_proj = nn.Linear(args.drug_embed_dim, args.gene2vec_dim, bias = True)
        self.key_proj = nn.Linear(args.gex_embed_dim, args.attn_dim, bias = True)
       


    def forward(self, x, gene2vec, get_gex_idxs, device, args, training = True):
        # x[0] : ecfp, x[1] : gex, x[2] : dosage, x[3] : duration

        self.tmp_batch_size = x[4].shape[0]

        drug_input = x[8].float().to(device)
        #torch.as_tensor(exp.float()).detach().to(device
        gex_input = x[1][:, get_gex_idxs].float().to(device) # bs x n genes x 1
        dose = x[2].float().to(device).view(-1,1)
        duration = x[3].float().to(device).view(-1,1)
        try:
            drug_input = self.bn1(drug_input)
            gex_input = self.bn2(gex_input).unsqueeze(-1)

            dose = self.bn3(dose)
            duration = self.bn4(duration)
        except ValueError:
            gex_input = gex_input.unsqueeze(-1)
#        drug_input = self.bn1(x[0].float().to(device))

        #   dropout for inputs
        drug_input = F.dropout(drug_input, training = training)
        gex_input = F.dropout(gex_input, training = training)

        drug_emb = F.dropout(self.drug_mlp2(
                    self.activ(self.drug_mlp1(drug_input))), training = training) 

        gene2vec = self.g2v_embedding.float().to(device) # 1 x n genes x vec dims

        gex_emb = gex_input * gene2vec # bs x n genes x vec dims

#        gex_embed = gex_embed.view(-1, args.gene2vec_dim) # bs*n genex x vec dims

        
        def drug2gex_attn_applied(drug_feat, gex_feat, self, args, training):

            #   drug input & gex input requires same dimension
            drug_proj = F.sigmoid(self.query_proj(drug_feat))
            #   bs x 200
            gex_proj = F.sigmoid(self.key_proj(gex_feat))
            #   bs x num genes x 200

            drug_proj  = drug_proj.unsqueeze(1)
            gex_proj = gex_proj.transpose(2,1)
            attn = torch.bmm(drug_proj, gex_proj).squeeze(1) # bs x num_genes 
            attn = F.softmax(attn, dim = -1)
#            attn = F.sigmoid(attn)
            
#            attn = F.tanh(attn)

#            print(attn[0])


            gex_attned = torch.mul(attn.unsqueeze(-1), gex_feat) # bs x num_genes x 200
            
            return attn, gex_attned
         

#        gex_emb = drug2gex_attn_applied(drug_emb, gex_emb, self, args, training)
       
#        read_out = torch.mean(gex_emb, dim = 1)


#        ver 5
        #   drug input & gex input requires same dimension

        attn, _ = drug2gex_attn_applied(drug_emb, gex_emb, self, args, training) # bs x n genes
        self.attn = attn
        read_out = torch.mul(attn.unsqueeze(-1), gex_emb) # bs x n genes x 200 dim
        read_out = torch.mean(read_out, dim = 1) # bs x 200


        total_emb = torch.cat([drug_emb, read_out, dose, duration], dim = -1) 
        
#        proba = self.pred_mlp(total_emb)
        total_emb = F.dropout(self.activ(self.pred_mlp1(total_emb)))
        total_emb = F.dropout(self.activ(self.pred_mlp2(total_emb)))
        proba = self.pred_mlp3(total_emb)

        return proba

class GEX_drug_attn_MLP_ver3(nn.Module):
    def __init__(self, g2v_embedding, drug_embed_dim, gex_embed_dim, num_classes, args):
        super(GEX_drug_attn_MLP_ver3, self).__init__()

#        self.gex_mlp = nn.Linear(args.num_genes, gex_embed_dim, bias = True)
        self.g2v_embedding = torch.tensor(g2v_embedding).unsqueeze(0)
#        self.ppi_adj = ppi_adj
        args.gex_embed_dim = self.g2v_embedding.size()[-1]

        self.bn1 = nn.BatchNorm1d(args.ecfp_nBits)
        self.bn2 = nn.BatchNorm1d(args.num_genes)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn4 = nn.BatchNorm1d(1)

        self.drug_mlp = nn.Linear(args.ecfp_nBits, drug_embed_dim, bias = True)
        
        self.gex_mlp = nn.Linear(args.num_genes, gex_embed_dim, bias = True)

#        self.pred_emb_dim = args.gcn_hidden_dim2
#        self.pred_emb_dim = args.drug_embed_dim + args.gex_embed_dim + 2
        self.pred_emb_dim = args.drug_embed_dim + args.num_genes + 2
#        self.pred_emb_dim = args.drug_embed_dim + args.drug_embed_dim + 2

        self.pred_mlp = nn.Linear(self.pred_emb_dim, num_classes, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()


        #   
        #self.g2v_embeddings = nn.Parameters(torch.tensor(g2v_embeddings))
        #   Attention layers
        self.query_proj = nn.Linear(args.drug_embed_dim, args.attn_dim, bias = True)
        self.key_proj = nn.Linear(args.gex_embed_dim, args.attn_dim, bias = True)


    def forward(self, x, gene2vec, get_gex_idxs, device, args, training = True):
        # x[0] : ecfp, x[1] : gex, x[2] : dosage, x[3] : duration

        self.tmp_batch_size = x[4].shape[0]
        #torch.as_tensor(exp.float()).detach().to(device
        drug_input = self.bn1(x[8].float().to(device))

        drug_emb = self.drug_mlp(drug_input) 


        gex_input = self.bn2(x[1][:, get_gex_idxs].float().to(device)).unsqueeze(-1) # bs x n genes x 1

        gene2vec = self.g2v_embedding.float().to(device) # 1 x n genes x vec dims

        gex_emb = gex_input * gene2vec # bs x n genes x vec dims

#        gex_embed = gex_embed.view(-1, args.gene2vec_dim) # bs*n genex x vec dims

        def drug2gex_attn_applied(drug_feat, gex_feat, self, args, training):

            #   drug input & gex input requires same dimension
            drug_proj = self.query_proj(drug_feat)
            #   bs x 200
            gex_proj = self.key_proj(gex_feat)
            #   bs x num genes x 200

        
            drug_proj  = drug_proj.unsqueeze(1)
            gex_proj = gex_proj.transpose(2,1)
            attn = torch.bmm(drug_proj, gex_proj).squeeze(1) # bs x num_genes 
            attn = F.softmax(attn, dim = -1)
            print(attn[0])

            gex_attned = torch.mul(attn.unsqueeze(-1), gex_feat) # bs x num_genes x 200
            
            return attn, gex_attned

#        gex_emb = drug2gex_attn_applied(drug_emb, gex_emb, self, args, training)
       
#        read_out = torch.mean(gex_emb, dim = 1)

#        ver 3
        attn, _ = drug2gex_attn_applied(drug_emb, gex_emb, self, args, training) # bs x n genes
        self.attn = attn
        read_out =  self.activ(self.read_out_mlp2(
                    self.activ(self.read_out_mlp1(gex_emb)))).squeeze(-1) # bs x n genes
        read_out = torch.mul(attn, read_out) # bs x num_genes x 200
        
        dose = x[2].float().to(device).view(-1,1)
        duration = x[3].float().to(device).view(-1,1)
        dose = self.bn3(dose)
        duration = self.bn4(duration)

        total_emb = torch.cat([drug_emb, read_out, dose, duration], dim = -1) 
        
        proba = self.pred_mlp(total_emb)

        return proba

class Drug_MLP(nn.Module):
    def __init__(self, drug_embed_dim, gex_embed_dim, num_classes, args):
        super(Drug_MLP, self).__init__()

        self.bn1 = nn.BatchNorm1d(args.ecfp_nBits)
        self.bn2 = nn.BatchNorm1d(args.num_genes)

        self.drug_mlp = nn.Linear(args.ecfp_nBits, gex_embed_dim, bias = True)
        self.gex_mlp = nn.Linear(args.num_genes, gex_embed_dim, bias = True)

        self.pred_emb_dim = drug_embed_dim + gex_embed_dim 
        self.pred_mlp = nn.Linear(self.pred_emb_dim, num_classes, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()
        

    def forward(self, x, device, args, training = True):
        # x[0] : ecfp, x[1] : gex, x[2] : dosage, x[3] : duration

        #torch.as_tensor(exp.float()).detach().to(device
        drug_input = x[1].float().to(device)
        gex_input = x[0].float().to(device)
#        drug_input = x[8].float().to(device)
#        gex_input = x[0].float().to(device)

        

        drug_emb = F.dropout(self.activ(self.drug_mlp(self.bn1(
                    drug_input))), training = training)

        gex_emb = F.dropout(self.activ(self.gex_mlp(self.bn2(
                    gex_input))), training = training)

        
        total_emb = torch.cat([drug_emb, gex_emb],
                                dim = -1)

        #proba = self.activ2(self.pred_mlp(total_emb))
        proba = self.pred_mlp(total_emb)
        
        return proba
        

class GCN_GEX_MLP(nn.Module):
    def __init__(self, drug_embed_dim, gex_embed_dim, num_classes, 
                    atom_dict, args):
        super(GCN_GEX_MLP, self).__init__()

        self.bn1 = nn.BatchNorm1d(args.ecfp_nBits)
        self.bn2 = nn.BatchNorm1d(args.num_genes)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn4 = nn.BatchNorm1d(1)

        self.drug_mlp = nn.Linear(args.ecfp_nBits, drug_embed_dim, bias = True)
        self.gex_mlp = nn.Linear(args.num_genes, gex_embed_dim, bias = True)
#        self.pred_emb_dim = drug_embed_dim + gex_embed_dim + 2
        self.pred_emb_dim = args.gcn_hidden_dim2 + gex_embed_dim + 2
        self.pred_mlp = nn.Linear(self.pred_emb_dim, num_classes, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()

        self.batch_norm = nn.BatchNorm1d(num_features = args.smiles_emb_dim +
                                                        args.num_genes +
                                                        2)

        self.atom_embed = nn.Embedding(args.num_atom_symbols+1,
                                        args.smiles_emb_dim,
                                        padding_idx = args.atom_pad_idx)

        self.atom_dict = atom_dict

        self.gcn1 = geo_nn.GCNConv(args.smiles_emb_dim, args.gcn_hidden_dim1)
        self.gcn2 = geo_nn.GCNConv(args.gcn_hidden_dim1, args.gcn_hidden_dim2)
        self.gcn3 = geo_nn.GCNConv(args.gcn_hidden_dim2, args.gcn_hidden_out)

        
    def forward(self, x, device, args, dropout = 0.5, training = True):
        # x[0] : ecfp, x[1] : gex, x[2] : dosage, x[3] : duration
        # x[4] : label


        self.tmp_batch_size = len(x[4])


        def get_batch_adj(adjs):
            block_adjs = scipy.linalg.block_diag(*tuple(adjs))
            #   total nodes in batch x total nodes in batch
            
            #   get block-wise sparse adj mats
            block_adjs = np.array(sparse.find(block_adjs)[:2]) # start node array x end node array
            return block_adjs

        drug_adj = get_batch_adj(x[8])
        
        drug_adj = torch.tensor(drug_adj, dtype = torch.long).to(device)

        #   ???

        drug_atom_idxs = torch.tensor(np.vstack(x[9]).T, dtype = torch.long).to(device)

        ####

        drug_atoms_emb = self.atom_embed(drug_atom_idxs)
        drug_atoms_emb = drug_atoms_emb.view(-1, args.smiles_emb_dim)

        drug_gcn1_output = F.dropout(self.activ(
                                        self.gcn1(
                                            drug_atoms_emb, 
                                            drug_adj)
                                        ), training = training)

        drug_gcn2_output = F.dropout(self.activ(self.gcn2(
                                            drug_gcn1_output, 
                                            drug_adj)), training = training)

#        drug_gcn2_output = torch.add(drug_gcn2_output, drug_gcn1_output)

#        drug_gcn3_output = F.dropout(self.activ(self.gcn3(
#                                            drug_gcn2_output, 
#                                            drug_adj)), training = training)

#        drug_gcn_output = torch.add(drug_gcn3_output, drug_gcn2_output)
#        drug_gcn_output = drug_gcn3_output + drug_gcn2_output + drug_gcn1_output
        drug_gcn_output = drug_gcn2_output + drug_gcn1_output

        #   Readout + view batchwise

        batch_output = drug_gcn_output.view(
                            self.tmp_batch_size, -1, args.gcn_hidden_out)
        read_out = torch.sum(batch_output, dim = 1)
    


        #   Resimnet workframe

        gex_input = self.bn2(x[1].float().to(device))
        dose = self.bn3(x[2].float().to(device).view(-1,1))
        duration = self.bn4(x[3].float().to(device).view(-1,1))

        

        #drug_emb = F.dropout(self.activ(self.drug_mlp(drug_input)), training = training)
        drug_emb = read_out
        gex_emb = F.dropout(self.activ(self.gex_mlp(gex_input)), training = training)

        total_emb = torch.cat([drug_emb, gex_emb, dose, duration],
                                dim = -1)
        
#        proba = self.activ2(self.pred_mlp(total_emb))
        proba = self.pred_mlp(total_emb)
        
        return proba

class Resimnet_attn(nn.Module):
    def __init__(self, drug_embed_dim, gex_embed_dim, num_classes, args):
        super(Resimnet_attn, self).__init__()

        self.bn1 = nn.BatchNorm1d(args.ecfp_nBits)
        self.bn2 = nn.BatchNorm1d(args.num_genes)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn4 = nn.BatchNorm1d(1)

        self.drug_mlp = nn.Linear(args.ecfp_nBits, drug_embed_dim, bias = True)
        self.gex_mlp = nn.Linear(args.num_genes, gex_embed_dim, bias = True)
        self.pred_emb_dim = drug_embed_dim + gex_embed_dim + 2
        self.pred_mlp = nn.Linear(self.pred_emb_dim, num_classes, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()
        self.activ3 = nn.Sigmoid()

        self.batch_norm = nn.BatchNorm1d(num_features = args.ecfp_nBits +
                                                        args.num_genes +
                                                        2)
        
        #   Attention layers
#        self.query_vec = torch.tensor(torch.empty(args.attn_dim), 
#                            requires_grad = True).to(torch.device(args.device))
        self.query_vec = nn.init.xavier_normal_(torch.empty(1,args.attn_dim))
        self.query_vec = nn.Parameter(self.query_vec)

        self.key_proj = nn.Linear(args.gex_embed_dim, args.attn_dim, bias = True)

        self.attn = Attention(args.attn_dim)

    def forward(self, x, device, args, training = True):
        # x[0] : ecfp, x[1] : gex, x[2] : dosage, x[3] : duration

        self.tmp_batch_size = x[4].shape[0]

        #drug_input = x[0].float().to(device)

        #   If using pretrained drug feat:
        drug_input = x[8].float().to(device)

        gex_input = x[1].float().to(device)
        dose = x[2].float().to(device).view(-1,1)
        duration = x[3].float().to(device).view(-1,1)

#        drug_input = self.bn1(drug_input)
#        gex_input = self.bn2(gex_input)
        dose = self.bn3(dose)
        duration = self.bn4(duration)

        

        drug_emb = F.dropout(self.activ(self.drug_mlp(
                            drug_input)),training = training)
        gex_emb = F.dropout(self.activ(self.gex_mlp(
                            gex_input)), training = training)

        #   Get attn applied
        def get_attn_applied(drug_feat, gex_feat, query_vec, self, args, training):
            query = query_vec

            #   drug input & gex input requires same dimension
            drug_proj = F.dropout(self.activ(self.key_proj(drug_feat)), training)
            gex_proj = F.dropout(self.activ(self.key_proj(gex_feat)), training)

#            query = query.expand(self.tmp_batch_size, -1)
            query = torch.unsqueeze(query.repeat(self.tmp_batch_size, 1), dim = -1)
            
            drug_attn = torch.bmm(drug_proj.unsqueeze(1), query).squeeze(-1) # bs x 1
            gex_attn = torch.bmm(gex_proj.unsqueeze(1), query).squeeze(-1) # bs x 1

            attn = torch.cat((drug_attn, gex_attn), dim = -1) # bs x 2
            attn = F.softmax(attn, dim = -1)

            drug_attned = torch.mul(drug_feat, drug_feat)
            gex_attned = torch.mul(gex_feat, gex_feat)
            
            return drug_attned, gex_attned


        total_emb = torch.cat([drug_emb, gex_emb, dose, duration],
                                dim = -1)
        

#        proba = self.activ2(self.pred_mlp(total_emb))
        proba = self.pred_mlp(total_emb)
        
        return proba

class GCN_GEX_attn_MLP(nn.Module):
    def __init__(self, drug_embed_dim, gex_embed_dim, num_classes, 
                    atom_dict, args):
        super(GCN_GEX_attn_MLP, self).__init__()

        self.bn1 = nn.BatchNorm1d(args.ecfp_nBits)
        self.bn2 = nn.BatchNorm1d(args.num_genes)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn4 = nn.BatchNorm1d(1)

        self.drug_mlp = nn.Linear(args.ecfp_nBits, drug_embed_dim, bias = True)
        self.gex_mlp = nn.Linear(args.num_genes, gex_embed_dim, bias = True)
#        self.pred_emb_dim = drug_embed_dim + gex_embed_dim + 2
        self.pred_emb_dim = args.gcn_hidden_dim2 + gex_embed_dim + 2
        self.pred_mlp = nn.Linear(self.pred_emb_dim, num_classes, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()
        self.activ3 = nn.Sigmoid()

        self.batch_norm = nn.BatchNorm1d(num_features = args.smiles_emb_dim +
                                                        args.num_genes +
                                                        2)

        self.atom_embed = nn.Embedding(args.num_atom_symbols+1,
                                        args.smiles_emb_dim,
                                        padding_idx = args.atom_pad_idx)

        self.atom_dict = atom_dict

        self.gcn1 = geo_nn.GCNConv(args.smiles_emb_dim, args.gcn_hidden_dim1)
        self.gcn2 = geo_nn.GCNConv(args.gcn_hidden_dim1, args.gcn_hidden_dim2)
        self.gcn3 = geo_nn.GCNConv(args.gcn_hidden_dim2, args.gcn_hidden_out)

        #   Attention layers
#        self.query_vec = torch.tensor(torch.empty(args.attn_dim), 
#                            requires_grad = True).to(torch.device(args.device))
        self.query_vec = nn.init.xavier_normal_(torch.empty(1,args.attn_dim))
#        self.query_vec = self.query_vec.clone().detach().requires_grad_(True).to(torch.device(args.device))
        self.query_vec = nn.Parameter(self.query_vec)
        self.key_proj = nn.Linear(args.gex_embed_dim, args.attn_dim)
        


        
    def forward(self, x, device, args, dropout = 0.5, training = True):
        # x[0] : ecfp, x[1] : gex, x[2] : dosage, x[3] : duration
        # x[4] : label


        self.tmp_batch_size = len(x[4])

        def get_batch_adj(adjs):
            block_adjs = scipy.linalg.block_diag(*tuple(adjs))
            #   total nodes in batch x total nodes in batch
            
            #   get block-wise sparse adj mats
            block_adjs = np.array(sparse.find(block_adjs)[:2]) # start node array x end node array
            return block_adjs

        drug_adj = get_batch_adj(x[8])
        
        drug_adj = torch.tensor(drug_adj, dtype = torch.long).to(device)

        #   ???

        drug_atom_idxs = torch.tensor(np.vstack(x[9]).T, dtype = torch.long).to(device)

        #   GCN framework

        drug_atoms_emb = self.atom_embed(drug_atom_idxs)
        drug_atoms_emb = drug_atoms_emb.view(-1, args.smiles_emb_dim)

        drug_gcn1_output = F.dropout(self.activ(
                                        self.gcn1(
                                            drug_atoms_emb, 
                                            drug_adj)
                                        ), training = training)

        drug_gcn2_output = F.dropout(self.activ(self.gcn2(
                                            drug_gcn1_output, 
                                            drug_adj)), training = training)
#        drug_gcn2_output = torch.add(drug_gcn2_output, drug_gcn1_output)

        drug_gcn3_output = F.dropout(self.activ(self.gcn3(
                                            drug_gcn2_output, 
                                            drug_adj)), training = training)

        drug_gcn_output = drug_gcn3_output + drug_gcn2_output + drug_gcn1_output

        #   Readout + view batchwise

        batch_output = drug_gcn_output.view(
                            self.tmp_batch_size, -1, args.gcn_hidden_out)
        read_out = torch.sum(batch_output, dim = 1)
    


        #   Resimnet workframe

        gex_input = self.bn2(x[1].float().to(device))


        dose = self.bn3(x[2].float().to(device).view(-1,1))
        duration = self.bn4(x[3].float().to(device).view(-1,1))

        #drug_emb = F.dropout(self.activ(self.drug_mlp(drug_input)), training = training)
        drug_emb = read_out
        gex_emb = F.dropout(self.activ(self.gex_mlp(gex_input)), training = training)

        #   Get attn applied
        def get_attn_applied(drug_feat, gex_feat, query_vec, self, args, training = True):
            query = query_vec

            #   drug input & gex input requires same dimension
            drug_proj = F.dropout(self.activ(self.key_proj(drug_feat)), training)
            gex_proj = F.dropout(self.activ(self.key_proj(gex_feat)), training)

#            query = query.expand(self.tmp_batch_size, -1)
            query = torch.unsqueeze(query.repeat(self.tmp_batch_size, 1), dim = -1)
            
            drug_attn = torch.bmm(drug_proj.unsqueeze(1), query).squeeze(-1) # bs x 1
            gex_attn = torch.bmm(gex_proj.unsqueeze(1), query).squeeze(-1) # bs x 1

            attn = torch.cat((drug_attn, gex_attn), dim = -1) # bs x 2
            attn = F.softmax(attn, dim = -1)

            drug_attned = torch.mul(drug_feat, drug_feat)
            gex_attned = torch.mul(gex_feat, gex_feat)
            
            return drug_attned, gex_attned

        drug_emb, gex_emb = get_attn_applied(drug_emb, gex_emb, self.query_vec, 
                                            self, args, training)

        total_emb = torch.cat([drug_emb, gex_emb, dose, duration],
                                dim = -1)
        
#        proba = self.activ2(self.pred_mlp(total_emb))
        proba = self.pred_mlp(total_emb)
        
        return proba




class GCN_GEX_MLP(nn.Module):
    def __init__(self, drug_embed_dim, gex_embed_dim, num_classes, 
                    atom_dict, args):
        super(GCN_GEX_MLP, self).__init__()

        self.bn1 = nn.BatchNorm1d(args.ecfp_nBits)
        self.bn2 = nn.BatchNorm1d(args.num_genes)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn4 = nn.BatchNorm1d(1)

        self.drug_mlp = nn.Linear(args.ecfp_nBits, drug_embed_dim, bias = True)
        self.gex_mlp = nn.Linear(args.num_genes, gex_embed_dim, bias = True)
#        self.pred_emb_dim = drug_embed_dim + gex_embed_dim + 2
        self.pred_emb_dim = args.gcn_hidden_dim2 + gex_embed_dim + 2
        self.pred_mlp = nn.Linear(self.pred_emb_dim, num_classes, bias = True)

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()

        self.batch_norm = nn.BatchNorm1d(num_features = args.smiles_emb_dim +
                                                        args.num_genes +
                                                        2)

        self.atom_embed = nn.Embedding(args.num_atom_symbols+1,
                                        args.smiles_emb_dim,
                                        padding_idx = args.atom_pad_idx)

        self.atom_dict = atom_dict

        self.gcn1 = geo_nn.GCNConv(args.smiles_emb_dim, args.gcn_hidden_dim1)
        self.gcn2 = geo_nn.GCNConv(args.gcn_hidden_dim1, args.gcn_hidden_dim2)
        self.gcn3 = geo_nn.GCNConv(args.gcn_hidden_dim2, args.gcn_hidden_out)

        
    def forward(self, x, device, args, dropout = 0.5, training = True):
        # x[0] : ecfp, x[1] : gex, x[2] : dosage, x[3] : duration
        # x[4] : label


        self.tmp_batch_size = len(x[4])


        def get_batch_adj(adjs):
            block_adjs = scipy.linalg.block_diag(*tuple(adjs))
            #   total nodes in batch x total nodes in batch
            
            #   get block-wise sparse adj mats
            block_adjs = np.array(sparse.find(block_adjs)[:2]) # start node array x end node array
            return block_adjs

        drug_adj = get_batch_adj(x[8])
        
        drug_adj = torch.tensor(drug_adj, dtype = torch.long).to(device)

        #   ???

        drug_atom_idxs = torch.tensor(np.vstack(x[9]).T, dtype = torch.long).to(device)

        ####

        drug_atoms_emb = self.atom_embed(drug_atom_idxs)
        drug_atoms_emb = drug_atoms_emb.view(-1, args.smiles_emb_dim)

        drug_gcn1_output = F.dropout(self.activ(
                                        self.gcn1(
                                            drug_atoms_emb, 
                                            drug_adj)
                                        ), training = training)

        drug_gcn2_output = F.dropout(self.activ(self.gcn2(
                                            drug_gcn1_output, 
                                            drug_adj)), training = training)

#        drug_gcn2_output = torch.add(drug_gcn2_output, drug_gcn1_output)

        drug_gcn3_output = F.dropout(self.activ(self.gcn3(
                                            drug_gcn2_output, 
                                            drug_adj)), training = training)

#        drug_gcn_output = torch.add(drug_gcn3_output, drug_gcn2_output)
        drug_gcn_output = drug_gcn3_output + drug_gcn2_output + drug_gcn1_output

        #   Readout + view batchwise

        batch_output = drug_gcn_output.view(
                            self.tmp_batch_size, -1, args.gcn_hidden_out)
        read_out = torch.sum(batch_output, dim = 1)
    


        #   Resimnet workframe

        gex_input = self.bn2(x[1].float().to(device))
        dose = self.bn3(x[2].float().to(device).view(-1,1))
        duration = self.bn4(x[3].float().to(device).view(-1,1))

        

        #drug_emb = F.dropout(self.activ(self.drug_mlp(drug_input)), training = training)
        drug_emb = read_out
        gex_emb = F.dropout(self.activ(self.gex_mlp(gex_input)), training = training)

        total_emb = torch.cat([drug_emb, gex_emb, dose, duration],
                                dim = -1)
        
#        proba = self.activ2(self.pred_mlp(total_emb))
        proba = self.pred_mlp(total_emb)
        
        return proba






"""   

"""
#        ver 1
read_out =  self.activ(self.read_out_mlp2(
            self.activ(self.read_out_mlp1(gex_emb)))).squeeze(-1) # bs x n genes
dose = x[2].float().to(device).view(-1,1)
duration = x[3].float().to(device).view(-1,1)
dose = self.bn3(dose)
duration = self.bn4(duration)

total_emb = torch.cat([drug_emb, read_out, dose, duration], dim = -1) 

proba = self.pred_mlp(total_emb)
"""
""" 
#        ver 2
read_out =  self.activ(self.read_out_mlp2(
            self.activ(self.read_out_mlp1(gex_emb)))).squeeze(-1) # bs x n genes

read_out = self.activ(self.read_out_mlp4(
            self.activ(self.read_out_mlp3(read_out)))) # bs x drug embed dim
            

dose = x[2].float().to(device).view(-1,1)
duration = x[3].float().to(device).view(-1,1)
dose = self.bn3(dose)
duration = self.bn4(duration)

total_emb = torch.cat([drug_emb, read_out, dose, duration], dim = -1) 

proba = self.pred_mlp(total_emb)
"""
"""
#        ver 3
attn, _ = drug2gex_attn_applied(drug_emb, gex_emb, self, args, training) # bs x n genes
read_out =  self.activ(self.read_out_mlp2(
            self.activ(self.read_out_mlp1(gex_emb)))).squeeze(-1) # bs x n genes
read_out = torch.mul(attn, read_out) # bs x num_genes x 200

dose = x[2].float().to(device).view(-1,1)
duration = x[3].float().to(device).view(-1,1)
dose = self.bn3(dose)
duration = self.bn4(duration)

total_emb = torch.cat([drug_emb, read_out, dose, duration], dim = -1) 

proba = self.pred_mlp(total_emb)
"""

"""
#        ver 4
attn, _ = drug2gex_attn_applied(drug_emb, gex_emb, self, args, training) # bs x n genes
read_out =  self.activ(self.read_out_mlp2(
            self.activ(self.read_out_mlp1(gex_emb)))).squeeze(-1) # bs x n genes
read_out = torch.mul(attn, read_out) # bs x num_genes x 200
read_out = self.activ(self.read_out_mlp4(
            self.activ(self.read_out_mlp3(read_out)))) # bs x drug embed dim

dose = x[2].float().to(device).view(-1,1)
duration = x[3].float().to(device).view(-1,1)
dose = self.bn3(dose)
duration = self.bn4(duration)

total_emb = torch.cat([drug_emb, read_out, dose, duration], dim = -1) 

proba = self.pred_mlp(total_emb)
"""
