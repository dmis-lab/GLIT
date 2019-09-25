import numpy as np
import pandas as pd
import pickle
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import gradcheck

from torch_geometric.data import Data, DataListLoader

import networkx as nx
from scipy import sparse

from sklearn import preprocessing
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score

from model import *
import utils2
from utils2 import *

import argparse

import os

#   #   #   #   #   #   
seed = 44
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#   #   #   #   #   #   

def Get_DataLoader(drug_label, input_list, args):
    #   Drug Label as df
    #   Input List as list
    train_drug = []
    valid_drug = []
    test_drug = []
    try:
        if args.dataset_ver == 4:
            drug_list_path = 'data/drug_list_92742_70138_'+str(args.seed)+'ver4.pkl'
        with open(drug_list_path, 'rb') as f:
            train_drug, valid_drug, test_drug = pickle.load(f)

    except:
        print('Drug List is not created yet')
        print('data ver :' + str(args.dataset_ver))


    train_list = []
    valid_list = []
    test_list = []

    for i, x in enumerate(input_list):
        if x[5] in train_drug:
            train_list.append(x)
        elif x[5] in valid_drug:
            valid_list.append(x)
        elif x[5] in test_drug:
            test_list.append(x)
    print('train set len : ' + str(len(train_list)))
    print('valid set len : ' + str(len(valid_list)))
    print('test set len : ' + str(len(test_list)))

    train_loader = DataLoader(dataset = train_list,
                            batch_size = args.batch_size,
                            shuffle = True)
    try:
        valid_loader = DataLoader(dataset = valid_list,
                                batch_size = args.batch_size,
                                shuffle = True)
    except:
        pass

    test_loader = DataLoader(dataset = test_list,
                            batch_size = args.batch_size,
                            shuffle = True)
    return train_loader, valid_loader, test_loader

def TrainAndTest(train_loader, valid_loader, test_loader, model, optimizer, 
        loss_fn, scheduler, device, ppi_adj, get_gex_idxs, n_sample, args):

    ####
    for epoch in range(args.n_epochs):
        #   Train set

        train_losses = []
        train_pred = []
        train_proba = []
        train_label = []

        for x in train_loader:

            if ('PPI' in args.model) or ('kegg' in args.model):
                proba = model(x, ppi_adj,  get_gex_idxs,
                                device, args, epoch, training = True)
            elif 'attn' in args.model:
                proba = model(x, get_gex_idxs,
                                device, args, training = True)
            else:
                proba = model(x, device, args, training = True)

            label = x[4].to(device)
    
            #   divide loss for each sample for num of drugs
            num_samples = x[9].float().to(device)
            
            loss = loss_fn(proba, label)

            #   Loss propagation divided by num of samples per drug

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if ('PPI' in args.model) or ('kegg' in args.model):
                proba = model(x, ppi_adj, get_gex_idxs,
                                device, args, epoch, training = True)
            elif 'attn' in args.model:
                proba = model(x, get_gex_idxs,
                                device, args, training = True)
            else:
                proba = model(x, device, args, training = True)

            proba = F.softmax(proba, dim = -1).detach().cpu()
            pred_bin = np.argmax(proba, axis = -1)


            train_losses.append(loss.detach().cpu())

            train_proba.append(proba)
            train_pred.append(pred_bin)
            train_label.append(label.detach().cpu().numpy())

        #   Valid set
        
        valid_pred = []
        valid_proba = []
        valid_label = []
        valid_losses = []
        valid_drug_names = []

        for x in valid_loader:

            if ('PPI' in args.model) or ('kegg' in args.model):
                proba = model(x, ppi_adj, get_gex_idxs,
#                                device, args, training = False)
                                device, args, epoch, training = False)
            elif 'attn' in args.model:
                proba = model(x, get_gex_idxs,
                                device, args, training = True)
            else:
                proba = model(x, device, args, training = False)

            label = x[4].to(device)
            drug_name = x[5]

            #   divide loss for each sample for num of drugs
            num_samples = x[9].float().to(device)
            if args.loss_alpha != 0.:
                loss = loss_fn(proba, label, num_samples, args.loss_alpha, args, device)
            else:
                loss = loss_fn(proba, label)

            valid_losses.append(loss.detach().cpu())

            proba = F.softmax(proba, dim = -1).detach().cpu().numpy()
            pred_bin = np.argmax(proba, axis = -1)
            valid_pred.append(pred_bin)
            valid_proba.append(proba)
            valid_label.append(label.detach().cpu().numpy())
            valid_drug_names.extend(drug_name)
        
        train_pred = np.hstack(train_pred)
        train_proba = np.vstack(train_proba)
        train_label = np.hstack(train_label)
        valid_pred = np.hstack(valid_pred)
        valid_proba = np.vstack(valid_proba)
        valid_label = np.hstack(valid_label)

        print(str(n_sample) + ' sample '+str(epoch) + ' Epochs')
        print("Train F1 Score")
        print('train_losses : ' + str(np.mean(train_losses)))
        if args.num_classes != 2:
            print(f1_score(train_label, train_pred, average = 'macro'))
            print(f1_score(train_label, train_pred, average = 'micro'))
            print(f1_score(train_label, train_pred, average = 'weighted'))
        else:
            print(f1_score(train_label, train_pred))
        print("Valid F1 Score")
        print('valid_losses : ' + str(np.mean(valid_losses)))
        if args.num_classes != 2:
            print(f1_score(valid_label, valid_pred, average = 'macro'))
            print(f1_score(valid_label, valid_pred, average = 'micro'))
            print(f1_score(valid_label, valid_pred, average = 'weighted'))
        else:
            Print_Scores(valid_label, valid_pred, valid_proba)


        #   check attention values
        if ('PPI' in args.model) or ('attn' in args.model):
            if args.attn_dim != 0:
                print(model.attn)
                with open('attn_values.pkl', 'wb') as f:
                    pickle.dump(model.attn.detach().cpu().numpy(), f)
             
        scheduler.step()


    #   Test set

    test_pred = []
    test_proba = []
    test_label = []
    test_drug_names = []


    for i, x in enumerate(test_loader):
        epoch = None
        if ('PPI' in args.model) or ('kegg' in args.model):
            proba = model(x, ppi_adj, get_gex_idxs, 
                            device, args, epoch, training = False)
        elif 'attn' in args.model:
            proba = model(x,  get_gex_idxs,
                            device, args, training = False)
        else:
            proba = model(x, device, args, training = False)
        label = x[4]
        drug_name = x[5]

        proba = F.softmax(proba, dim = -1).detach().cpu()

        pred_bin = np.argmax(proba, axis = -1)
        test_pred.append(pred_bin)
        test_proba.append(proba)
        test_label.append(label)
        test_drug_names.extend(drug_name)

        if i==0:
            try:
                print('Attn scores')
                print(model.attn.detach().cpu().numpy())
            except:
                pass

        #   Check oversmoothing issues
        gat_node_vecs = model.gcn_cat_list # 3 x bs x n nodes x feat dim
        with open('gat_node_vecs.pkl', 'wb') as f:
            pickle.dump(gat_node_vecs, f)
        
    test_pred = np.hstack(test_pred)
    test_proba = np.vstack(test_proba)
    test_label = np.hstack(test_label)

    print("Test Score")
    if args.num_classes != 2:
        print(f1_score(test_label, test_pred, average = 'macro'))
        print(f1_score(test_label, test_pred, average = 'micro'))
        print(f1_score(test_label, test_pred, average = 'weighted'))
    else:
        Print_Scores(test_label, test_pred, test_proba)
    print("Test Preds")
    print(test_pred[:10])
    print("Test Probas")
    print(test_proba[:10])
    print("Test Labels")
    print(test_label[:10])

    return valid_pred, valid_proba, valid_label, valid_drug_names,\
            test_pred, test_proba, test_label, test_drug_names, model
    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   
def ValidAndTest(valid_loader, test_loader, model, optimizer, 
        loss_fn, scheduler, device, ppi_adj, get_gex_idxs, args):

    ####
    epoch = None
    #   Valid set
    
    valid_pred = []
    valid_proba = []
    valid_label = []
    valid_losses = []
    valid_drug_names = []

    for x in valid_loader:

        proba = model(x, ppi_adj, get_gex_idxs,
                        device, args, epoch, training = False)

        label = x[4].to(device)
        drug_name = x[5]

        num_samples = x[9].float().to(device)

        proba = F.softmax(proba, dim = -1).detach().cpu().numpy()
        pred_bin = np.argmax(proba, axis = -1)
        valid_pred.append(pred_bin)
        valid_proba.append(proba)
        valid_label.append(label.detach().cpu().numpy())
        valid_drug_names.extend(drug_name)
        
    valid_pred = np.hstack(valid_pred)
    valid_proba = np.vstack(valid_proba)
    valid_label = np.hstack(valid_label)

    #   Test set

    test_pred = []
    test_proba = []
    test_label = []
    test_drug_names = []



    for i, x in enumerate(test_loader):
        proba = model(x, ppi_adj, get_gex_idxs, 
                        device, args, epoch, training = False)

        label = x[4]
        drug_name = x[5]

        proba = F.softmax(proba, dim = -1).detach().cpu()
        pred_bin = np.argmax(proba, axis = -1)
        test_pred.append(pred_bin)
        test_proba.append(proba)
        test_label.append(label)
        test_drug_names.extend(drug_name)
        
    test_pred = np.hstack(test_pred)
    test_proba = np.vstack(test_proba)
    test_label = np.hstack(test_label)

    print("Test Score")
    if args.num_classes == 2:
        print(Print_Scores(test_label, test_pred, test_proba))
    else:
        pass

    return valid_pred, valid_proba, valid_label, valid_drug_names,\
            test_pred, test_proba, test_label, test_drug_names, 


def Get_Models(ppi_adj, g2v_embedding, args, device):

    if args.model == 'GEX_PPI_GAT_cat4_MLP':
        return GEX_PPI_GAT_cat4_MLP(ppi_adj, g2v_embedding, args).to(device)
    
def main(args):
    #   #   #   #   #   #   
    seed = 44
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #   #   #   #   #   #   

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device[-1]


    #   Get Input List
    """
    x[0] = ecfp
    x[1] = l1000 gex
    x[2] = dosage
    x[3] = duration
    x[4] = label
    x[5] = pert_iname
    x[6] = cell_id
    x[7] = smiles string
    """

    if args.gex_feat == 'l1000':
        with open('data/labeled_list_woAmbi_92742_70138.pkl', 'rb') as f:
            input_list = pickle.load(f)

    args.num_genes = len(input_list[0][1])


    #   For 2 class analysis
    if args.num_classes == 2:
        for i, x in enumerate(input_list):
            if input_list[i][4] == 2:
                input_list[i][4] = 1

    drug_label = pd.read_csv('data/drug_label_92742_70138.tsv',
                                    delimiter = '\t',
                                    index_col = 0) 

    #   gene expressions in samples are sorted in gene info index order!

    gene_info = get_gene_info(args)
    #   index : gene num, 'pr_gene_sympol' : gene symbol


    random.shuffle(input_list)

    from sklearn.utils import shuffle
    drug_label = shuffle(drug_label, random_state = args.seed).copy()
    

#  use ecfp feature
    for i, x in enumerate(input_list):
        input_list[i].append(get_ecfp_fingerprints(x[7], args))
    args.resimnet_dim = args.ecfp_nBits

    if 'PPI' in args.model:
        gene2vecdict, gene_info, ppi_adj, ppi_nx, g2v_embedding, get_gex_idxs, args = get_ppi_features(gene_info, args)
        
    elif 'attn' in args.model:

        gene2vecdict, gene_info, g2v_embedding, get_gex_idxs, args = get_drug_attn_features(gene_info, args)
        ppi_adj = None
    else:
        gene2vecdict = None
        ppi_adj = None
        g2v_embedding = None
        get_gex_idxs = None


#   Try Focal loss approach : add num samples of the drug
    drug_info_nums = pd.read_csv('data/drug_label_92742_70138.tsv',
                                delimiter = '\t',
                                index_col = 0)
    for i, x in enumerate(input_list):
        pert_iname = input_list[i][5]
        num_samples = drug_info_nums.loc[pert_iname, 'num_samples']
        input_list[i].append(num_samples) # x[9]
    #   add alpha as parameters : later. 

    #   Scores over 5 random samples
    valid_avg_scores = []
    valid_drug_scores = []
    avg_scores = []
    drug_scores = []



    for n_sample in range(5):

        #   dataset separation by drugs
        train_loader, valid_loader, test_loader =  Get_DataLoader(drug_label, input_list, args)

        device = torch.device(args.device)

        #   Choose Model

        model = Get_Models(ppi_adj, g2v_embedding, args, device)

        def init_normal(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

        model.apply(init_normal)
        print(model.parameters)
        

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr = args.learning_rate,
                                    weight_decay = args.weight_decay)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                        gamma = 0.95)


#        model_path = 'models/'+str(args.model)+str(args.num_gcn_hops)+'_'+str(args.drug_feat)+'_'+str(args.gex_feat)+'_'+str(args.learning_rate)+'_'+str(args.weight_decay)+'_'+str(args.n_epochs)+'_'+str(args.attn_dim)+'_'+str(args.loss_alpha)+'_'+str(args.g2v_pretrained)+'_'+str(args.seed)+'_ver'+str(args.dataset_ver) 
        model_path = 'models/'+str(args.model)+str(args.num_gcn_hops)+'_'+str(args.drug_feat)+'_'+str(args.gex_feat)+'_'+str(args.learning_rate)+'_'+str(args.weight_decay)+'_'+str(args.n_epochs)+'_'+str(args.g2v_pretrained)+'_'+str(args.seed)+'_ver'+str(args.dataset_ver) 
            

        if args.eval == True:
            model.load_state_dict(torch.load(model_path))
            valid_pred, valid_proba, valid_label, valid_drug_names,\
                    test_pred, test_proba, test_label, test_drug_names = ValidAndTest(
                        valid_loader, test_loader, model, optimizer, 
                        loss_fn, scheduler, device, ppi_adj, get_gex_idxs, args)
            

        else:
            valid_pred, valid_proba, valid_label, valid_drug_names,\
                    test_pred, test_proba, test_label, test_drug_names, model = TrainAndTest(
                        train_loader, valid_loader, test_loader, model, optimizer, 
                        loss_fn, scheduler, device, ppi_adj, get_gex_idxs, n_sample, args)


        valid_drug_labels_avg, valid_drug_preds_avg, valid_drug_probas_avg, \
                valid_drug_names_avg = Get_Drugwise_Preds(
                        valid_label, valid_pred, valid_proba, valid_drug_names)


        test_drug_labels_avg, test_drug_preds_avg, test_drug_probas_avg, \
                test_drug_names_avg = Get_Drugwise_Preds(
                        test_label, test_pred, test_proba, test_drug_names)
            
        #   save model state_dict
        if (args.save_model == True) & (args.eval == False):
            torch.save(model.state_dict(), model_path)

        with open(model_path+'_preds.pkl', 'wb') as f:
            pickle.dump([test_drug_preds_avg, test_drug_probas_avg, 
                    test_drug_labels_avg, test_drug_names_avg], f)

        valid_avg_scores.append(Return_Scores(valid_label, valid_pred, valid_proba))
        valid_drug_scores.append(Return_Scores(valid_drug_labels_avg, valid_drug_preds_avg, valid_drug_probas_avg))
        avg_scores.append(Return_Scores(test_label, test_pred, test_proba))
        drug_score = Return_Scores(test_drug_labels_avg, test_drug_preds_avg, test_drug_probas_avg)
        drug_scores.append(Return_Scores(test_drug_labels_avg, test_drug_preds_avg, test_drug_probas_avg))

        args.seed += 1
        #   #   #   #   #   iter ends

    valid_avg_mean = np.mean(valid_avg_scores, axis = 0)
    valid_avg_std = np.std(valid_avg_scores, axis = 0)
    valid_drug_mean = np.mean(valid_drug_scores, axis = 0)
    valid_drug_std = np.std(valid_drug_scores, axis = 0)

    avg_mean = np.mean(avg_scores, axis = 0)
    avg_std = np.std(avg_scores, axis = 0)
    drug_mean = np.mean(drug_scores, axis = 0)
    drug_std = np.std(drug_scores, axis = 0)

    print("Total valid Avg Scores")
    print(valid_avg_mean)
    print("Total valid Avg Std")
    print(valid_avg_std)
    print("Total valid Drug Scores")
    print(valid_drug_mean)
    print("Total valid Drug Std")
    print(valid_drug_std)

    print("Total Avg Scores")
    print(avg_mean)
    print("Total Avg Std")
    print(avg_std)
    print("Total Drug Scores")
    print(drug_mean)
    print("Total Drug Std")
    print(drug_std)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type = str, default = 'GEX_PPI_GAT_cat4_MLP')


    parser.add_argument('--gex_feat', type = str, default = 'l1000')
    parser.add_argument('--drug_feat', type = str, default = 'ecfp')

    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--num_classes', type = int, default = 2)
    parser.add_argument('--n_epochs', type = int, default = 20)
    parser.add_argument('--seed', type = int, default = 44)
    parser.add_argument('--device', type = str, default = 'cuda')


    #   features
    parser.add_argument('--ecfp_nBits', type = int, default = 2048)
    parser.add_argument('--ecfp_radius', type = int, default = 2)
    parser.add_argument('--resimnet_dim', type = int, default = 300)
    parser.add_argument('--num_genes', type = int, default = 300)
    parser.add_argument('--gex_embed_dim', type = int, default = 256)
    #   TR net
    parser.add_argument('--drug_embed_dim', type = int, default = 256)
    parser.add_argument('--max_adj_len', type = int, default = 0)


    parser.add_argument('--gcn_hidden_dim1', type = int, default = 64)
    parser.add_argument('--gcn_hidden_dim2', type = int, default = 64)
    parser.add_argument('--gcn_hidden_out', type = int, default = 64)
    parser.add_argument('--adj_max_len', type = int, default = 0)
    
    parser.add_argument('--gene2vec_dim', type = int, default = 200)

    parser.add_argument('--sort_pool_k', type = float, default = 0.)
    parser.add_argument('--gat_num_heads', type = int, default = 4)

    #   Attention

    parser.add_argument('--learning_rate', type = float, default = 0.001)
    parser.add_argument('--weight_decay', type = float, default = 1e-9)
    parser.add_argument('--gcnds', type = int, default = 64)
    parser.add_argument('--num_gcn_hops', type = int, default = 3)

    parser.add_argument('--g2v_pretrained', type = bool, default = True)

    parser.add_argument('--network_name', type = str, default = 'omnipath')
    parser.add_argument('--undir_graph', type = bool, default = False)

    parser.add_argument('--dataset_ver', type = int, default = 4)
    parser.add_argument('--save_model', type = bool, default = True)
    parser.add_argument('--eval', type = bool, default = False)



    args = parser.parse_args()


    if args.eval == True:
        args.save_model == False

    gcnd = args.gcnds
    
    args.gcn_hidden_dim1 = gcnd
    args.gcn_hidden_dim2 = gcnd
    args.gcn_hidden_out = gcnd

    main(args)
    print('Model : '+ args.model)
    print('GEX feature : '+ args.gex_feat)
    print('Drug feature : '+ args.drug_feat)
    print('num classes : '+ str(args.num_classes))
    print('num epochs : '+str(args.n_epochs))
    print('batch size : '+ str(args.batch_size))
    print('learning rate : ' + str(args.learning_rate))
    print('weight decay : ' + str(args.weight_decay))
    if 'GCN' or 'PPI' or 'kegg' in args.model:
        print('gcn hidden dim : '+ str(args.gcn_hidden_dim1))
        print('network name : ' + str(args.network_name))
        print('Undirected Graph : ' + str(args.network_name))
        print('GCN/GAT num hops: '+str(args.num_gcn_hops))
    
    print('drug embed dim : '+str(args.drug_embed_dim))
    print('gex embed dim : '+str(args.gex_embed_dim))
    print('drug embed dim : '+str(args.drug_embed_dim))

    if ('GAT' in args.model)&('GCN' in args.model):
        print('gat num heads : ' + str(args.gat_num_heads))

    print('use g2v pretrained : '+str(args.g2v_pretrained))
    print('dataset ver : '+str(args.dataset_ver))

    print('=    '*8)
    

