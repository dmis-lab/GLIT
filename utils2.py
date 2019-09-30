import numpy as np
import pandas as pd
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import gradcheck


import networkx as nx
from scipy import sparse
from sklearn import preprocessing

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score,\
         recall_score, precision_score, matthews_corrcoef
from sklearn.preprocessing import LabelBinarizer


import argparse
import os

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


def Get_Drugwise_Preds(test_labels, test_preds, test_probas, test_drug_names):

    test_drug_preds = {}
    test_drug_probas = {}
    test_drug_labels = {}
    for i, x in enumerate(test_drug_names):
        if x not in test_drug_probas.keys():
            test_drug_probas[x] = test_probas[i]
        else:
            avg_list = test_drug_probas[x]
            avg_list = np.vstack((avg_list, test_probas[i]))
            test_drug_probas[x] = avg_list
        if x not in test_drug_labels.keys():
            test_drug_labels[x] = test_labels[i]
    
    test_drug_preds_avg = []
    test_drug_probas_avg = []
    test_drug_labels_avg = []
    test_drug_names_avg = []
    for i, x in enumerate(test_drug_probas.keys()):
        test_drug_probas_avg.append(np.mean(test_drug_probas[x], axis = 0))
        test_drug_preds_avg.append(np.argmax(np.mean(test_drug_probas[x], axis = 0)))
        test_drug_labels_avg.append(test_drug_labels[x])
        test_drug_names_avg.append(x)
    return test_drug_labels_avg, test_drug_preds_avg, test_drug_probas_avg, test_drug_names_avg

def Get_Specificity(label, pred):
    tn, fn = 1e-10, 1e-10
    for i, x in enumerate(label):
        if (x == 0) & (pred[i] == 0):
            tn += 1
        elif (x == 0) & (pred[i] != 0):
            fn += 1
    return tn / (tn + fn)



def Get_CCR_Score(label, pred):
    
    if type(label) == list:
        label = np.asarray(label)
    if len(label.shape) > 1:
        label = label[0]
    if type(pred) == list:
        pred = np.asarray(pred)
    if len(pred.shape) > 1:
        pred = pred[0]

    N_0 = len([x for x in label if x == 0])+1e-10
    N_1 = len([x for x in label if x == 1])+1e-10
    T_n = len([i for i in range(len(label)) if (label[i] == pred[i]) & (pred[i] == 0)])
    T_p = len([i for i in range(len(label)) if (label[i] == pred[i]) & (pred[i] == 1)])
    return 0.5*(T_n/N_0 + T_p/N_1)


def Print_Scores(labels, preds, probas):
    labels = np.asarray(labels)
    if len(labels.shape) > 1:
        labels = labels[0]
    preds = np.asarray(preds)
    probas = np.asarray(probas)
    print('ccr : '+str(Get_CCR_Score(labels, preds)))
    print('accu : '+str(accuracy_score(labels, preds)))
    print('f1 : '+str(f1_score(labels, preds)))
    try : 
        print('auc : '+str(roc_auc_score(labels, 
                            [x[1] for x in probas])))
        print('aupr : '+str(average_precision_score(labels,
                            [x[1] for x in probas])))
    except:
        print('auc : 0.0')
        print('aupr : 0.0')
    print('MCC : '+str(matthews_corrcoef(labels, preds)))
    print('precision : '+str(precision_score(labels, preds)))
    print('recall : '+str(recall_score(labels, preds)))
    print('specificity : '+str(Get_Specificity(labels, preds)))

def Return_Scores(labels, preds, probas):
    labels = np.asarray(labels)
    if len(labels.shape) > 1:
        labels = labels[0]
    preds = np.asarray(preds)
    probas = np.asarray(probas)
    try:
        return [Get_CCR_Score(labels, preds), 
                accuracy_score(labels, preds), 
                f1_score(labels, preds),
                roc_auc_score(labels, 
                            [x[1] for x in probas]),
                average_precision_score(labels,
                            [x[1] for x in probas]),
                matthews_corrcoef(labels, preds),
                precision_score(labels, preds),
                recall_score(labels, preds),
                Get_Specificity(labels, preds)
                ]
    except:
        return[Get_CCR_Score(labels, preds), 
                accuracy_score(labels, preds), 
                f1_score(labels, preds),
                0,
                0,
                matthews_corrcoef(labels, preds),
                precision_score(labels, preds),
                recall_score(labels, preds),
                Get_Specificity(labels, preds)
                ]


def get_gene_info(args):
    if args.gex_feat == 'l1000':
        gene_info = pd.read_csv('data/GSE92742/GSE92742_Broad_LINCS_gene_info.txt',
                           delimiter = '\t',
                           index_col = 0)
        gene_info = gene_info.loc[gene_info.loc[:, 'pr_is_lm'] == 1].copy()
    elif args.gex_feat == 'ptgs_total':
        gene_info = pd.read_csv('data/GSE92742/GSE92742_Broad_LINCS_gene_info_ptgs_total.txt',
                           delimiter = '\t',
                           index_col = 0)
    elif args.gex_feat == 'ptgs_core':
        gene_info = pd.read_csv('data/GSE92742/GSE92742_Broad_LINCS_gene_info_ptgs_core.txt',
                           delimiter = '\t',
                           index_col = 0)
    else:
        gene_info = pd.read_csv('data/GSE92742/GSE92742_Broad_LINCS_gene_info.txt',
                           delimiter = '\t',
                           index_col = 0)
    return gene_info


def get_ecfp_fingerprints(smiles, args):

    #   Smiles issue with 0 -> create new input list
    imol = Chem.MolFromSmiles(smiles)
    fps = AllChem.GetMorganFingerprintAsBitVect(imol,
                                        radius = args.ecfp_radius,
                                        nBits = args.ecfp_nBits,
                                        useChirality = True)
    
    return np.array(fps)

"""
def get_drug_attn_features(gene_info, args):
    with open('data/gene2vec_dim_200_iter_9_dict.pkl', 'rb') as f:
        gene2vecdict = pickle.load(f)

    symb2id_dict = {}
    id2symb_dict = {}
    id2order_dict = {}  #   gene id - index order dict
    order2id_dict = {}  #   gene id - index order dict
    for i, x in enumerate(gene_info.index):
        symb2id_dict[gene_info.loc[x, 'pr_gene_symbol']] = x
        id2symb_dict[x] = gene_info.loc[x, 'pr_gene_symbol']
        id2order_dict[x] = i
        order2id_dict[i] = x

    get_gex_idxs = [i for i,x in enumerate(gene_info.index) if 
                    id2symb_dict[x] in gene2vecdict.keys()]
    args.num_genes = len(get_gex_idxs)
    gene_info = gene_info.loc[[x for x in gene_info.index if 
                            id2symb_dict[x] in gene2vecdict.keys()]].copy()

    g2v_embedding = np.vstack([gene2vecdict[x] for x in gene_info.loc[:,'pr_gene_symbol']])

    return gene2vecdict, gene_info, g2v_embedding, get_gex_idxs, args
"""

def get_ppi_features(gene_info, args):

    symb2id_dict = {}
    id2symb_dict = {}
    id2order_dict = {}  #   gene id - index order dict
    order2id_dict = {}  #   gene id - index order dict
    for i, x in enumerate(gene_info.index):
        symb2id_dict[gene_info.loc[x, 'pr_gene_symbol']] = x
        id2symb_dict[x] = gene_info.loc[x, 'pr_gene_symbol']
        id2order_dict[x] = i
        order2id_dict[i] = x

    #   get gene2vec dict
    with open('data/gene2vec_dim_200_iter_9_dict.pkl', 'rb') as f:
        gene2vecdict = pickle.load(f)

    gene_info = gene_info.loc[[x for x in gene_info.index if id2symb_dict[x] in gene2vecdict.keys()]]

    #   get ppi/kegg pathway networkx

    if args.network_name == 'biogrid' :
        biogrid = pd.read_csv('data/BIOGRID-ALL-3.5.169.tab.graphonly.txt', 
                             delimiter = '\t')
        biogridppi = nx.Graph()
        genes = list(set(biogrid.iloc[:,2].values) & set(biogrid.iloc[:,3]))
        biogridppi.add_nodes_from(genes)
        for i, x  in enumerate(biogrid.index):
            biogridppi.add_edge(biogrid.iloc[i,2], biogrid.iloc[i,3])
        biogridppi.remove_node(np.nan)
        ppi_nx = biogridppi

    elif args.network_name == 'omnipath' : 
        with open('data/Omnipath_190806_nx_DiGraph.pkl', 'rb') as f:
            omnipath = pickle.load(f)

        #   Get weakly connected components
        wcc = list([x for x in nx.weakly_connected_components(omnipath)][0])
        omnipath = nx.subgraph(omnipath, wcc)

        ppi_nx = omnipath



    #   get common genes

    common_genes = set(gene_info.loc[:,'pr_gene_symbol'].values)&set(gene2vecdict.keys()) & set(ppi_nx.nodes)

    ppi_nx = nx.subgraph(ppi_nx, list(common_genes)).copy()

    common_symbols = [x for x in gene_info.loc[:, 'pr_gene_symbol'].values.tolist()\
                        if x in common_genes]

    #   remove isolated nodes
    ppi_nx = ppi_nx.subgraph(common_symbols).copy()
    isol_nodes = nx.isolates(ppi_nx)
    common_symbols = [x for x in common_symbols if x not in isol_nodes]

    #   sort gene ids into fixed order
    common_orders = np.sort([id2order_dict[symb2id_dict[x]] for x in common_symbols])

    common_ids = [order2id_dict[x] for x in common_orders]

    common_symbols = [id2symb_dict[x] for x in common_ids]

    ppi_nx = ppi_nx.subgraph(common_symbols)

    #   to undirected graph (for gcn)
    if args.undir_graph == True:
        ppi_nx = ppi_nx.to_undirected()

    ppi_adj = nx.to_numpy_matrix(ppi_nx, nodelist = common_symbols)

    g2v_embedding = np.vstack([gene2vecdict[x] for x in common_symbols])

    get_gex_idxs = common_orders

    #   get adj as COO row x col form
    ppi_adj = np.array([sparse.coo_matrix(ppi_adj).row, 
                        sparse.coo_matrix(ppi_adj).col])

    with open('ppi_'+args.gex_feat+'_feats.pkl', 'wb') as f:
        pickle.dump((ppi_adj, g2v_embedding, get_gex_idxs), f)

    args.num_genes = len(get_gex_idxs)

    print('num PPI extracted genes : '+str(args.num_genes))
        
    return gene2vecdict, gene_info, ppi_adj, ppi_nx, g2v_embedding, get_gex_idxs, args


