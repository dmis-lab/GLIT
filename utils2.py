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

#   Create Loss Alpha as parameter
def loss_fn(proba, label, num_samples, alpha, args, device):
    losses = torch.zeros(1).to(device)
#        num_samples = torch.pow(num_samples, alpha)
    num_samples = torch.pow(num_samples, torch.tensor(alpha).to(device))

    for c in range(args.num_classes):
        label_c = torch.where(label == torch.tensor(c).to(device).expand_as(label), 
                            torch.ones(1).to(device).expand_as(label)
                            , torch.zeros(1).to(device).expand_as(label))
        proba_c = F.log_softmax(proba, dim = -1)[:, c]
#                    print(-1*label_c*proba_c)
        if label_c.size()[0] >1:
            loss = torch.mean(-1*label_c*proba_c/num_samples)
        else:
            loss = -1*label_c*proba_c
        losses += loss
    return losses

def get_multi_binary_loss(proba, label, loss_fn, device):

    bin_label = torch.tensor([[1,1,1,0,0],
                              [0,1,1,1,0],
                              [0,0,1,1,1]])
    binned_label = bin_label[label].to(device)

    loss = 0

    proba = F.sigmoid(proba)
    proba_minus = torch.tensor(1.).expand_as(proba).to(device) - proba
    proba = torch.cat([proba, proba_minus], dim = -1)


    proba_split = torch.split(proba, 5, dim = -2)
    binned_label_split = torch.split(binned_label, 5, dim = -1)
    print(proba_split)
    print(proba_split[0].size())
    print(binned_label_split)
    print(binned_label_split[0].size())
    for i in range(5):
        print(loss_fn(proba_split[i], binned_label_split[i]).size())
        loss += loss_fn(proba_split[i], binned_label_split[i])

    return loss


def globally_normalize_bipartite_adjacency(adjacencies, verbose=False, symmetric=True):
    """ Globally Normalizes set of bipartite adjacency matrices """

    if verbose:
        print('Symmetrically normalizing bipartite adj')
    # degree_u and degree_v are row and column sums of adj+I

    adj_tot = np.sum(adj for adj in adjacencies)
    degree_u = np.asarray(adj_tot.sum(1)).flatten()
    degree_v = np.asarray(adj_tot.sum(0)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree_u[degree_u == 0.] = np.inf
    degree_v[degree_v == 0.] = np.inf

    degree_u_inv_sqrt = 1. / np.sqrt(degree_u)
    degree_v_inv_sqrt = 1. / np.sqrt(degree_v)
    degree_u_inv_sqrt_mat = sparse.diags([degree_u_inv_sqrt], [0])
    degree_v_inv_sqrt_mat = sparse.diags([degree_v_inv_sqrt], [0])

    degree_u_inv = degree_u_inv_sqrt_mat.dot(degree_u_inv_sqrt_mat)

    if symmetric:
        adj_norm = [degree_u_inv_sqrt_mat.dot(adj).dot(degree_v_inv_sqrt_mat) for adj in adjacencies]

    else:
        adj_norm = [degree_u_inv.dot(adj) for adj in adjacencies]

    return adj_norm


def get_atom_symbol_bonds(drug_label, args):
    atoms = []
    bonds = []
    for i, x in enumerate(drug_label.loc[:,'SMILES'].values.tolist()):    
        imol = Chem.MolFromSmiles(x)
        ato = set([atom.GetSymbol() for atom in imol.GetAtoms()])
        bon = set([str(bond.GetBondType()) for bond in imol.GetBonds()])
        for at in ato:
            if at not in atoms:
                atoms.append(at)
        for bo in bon:
            if bo not in bonds:
                bonds.append(bo)
    
    args.num_atom_symbols = len(atoms)
    args.num_bond_types = len(bonds)
    atom_dict = {}
    bond_dict = {}
    
    atom_dict[args.atom_pad_symbol] = args.atom_pad_idx
#    bond_dict[args.bond_pad_symbol] = args.bond_pad_idx

    for i, x in enumerate(atoms):
        atom_dict[x] = i+1

    bond_dict = {'SINGLE' : 1., 'AROMATIC' : 1.5, 'DOUBLE' : 2., 'TRIPLE' : 3}
       

    return atom_dict, bond_dict, args

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

def get_chem_feat(smiles):
    imol = Chem.MolFromSmiles(smiles)
    iadjtmp = Chem.rdmolops.GetAdjacencyMatrix(imol)
            
    norm_adj = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(iadjtmp)
    atoms = [atom.GetSymbol() for atom in imol.GetAtoms()]
        
    return norm_adj, atoms

def get_ecfp_fingerprints(smiles, args):

    #   Smiles issue with 0 -> create new input list
    imol = Chem.MolFromSmiles(smiles)
    fps = AllChem.GetMorganFingerprintAsBitVect(imol,
                                        radius = args.ecfp_radius,
                                        nBits = args.ecfp_nBits,
                                        useChirality = True)
    
    return np.array(fps)

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                       'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                       'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',    # H?
                                       'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                       'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
#                     one_of_k_encoding_unk(atom.GetBondTypeAsDouble(),[1.0, 1.5, 2.0])+
                    [atom.GetIsAromatic()])

def get_drug_gcn_feat(smiles, atom_dict, bond_dict):
    #   create gcn feature (adj, atom index) from a sample

    def get_chem_feat(smiles):
        imol = Chem.MolFromSmiles(smiles)
#        iadjtmp = Chem.rdmolops.GetAdjacencyMatrix(imol)
        iadjtmp = Chem.rdmolops.GetAdjacencyMatrix(imol, useBO = True) 
        #   1. : Single
        #   1.5 : Aromatic
        #   2.0 : Double
        #   3.0 : Triple


        #norm_adj = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(iadjtmp)
        atoms = [atom.GetSymbol() for atom in imol.GetAtoms()]
            
        bonds = [bond.GetBondType() for bond in imol.GetBonds()]
                
        #return norm_adj, atoms
        return iadjtmp, atoms, bonds

    drug_feats  = get_chem_feat(smiles)

    #   drug adj mat : pad 0 to max len in batch
    drug_adj = drug_feats[0]
    #   #   #   #   #   #   #   #   
    drug_atoms = drug_feats[1]
    drug_atoms = [atom_dict[atom] for atom in drug_atoms]
    #drug_atoms = [atom_features(atom) for atom in drug_atoms]
    #   #   #   #   #   #   #   #   
#    drug_bonds = drug_feats[2]
    drug_bonds = [str(bond).split('.')[-1] for bond in drug_feats[2]]
    drug_bonds = [bond_dict[bond] for bond in drug_bonds]

    return drug_adj, drug_atoms, drug_bonds

def adj_padding(adj, args):
    max_len = args.max_adj_len
    adj_pad = np.zeros((max_len,max_len))
    adj_pad[:adj.shape[0], :adj.shape[1]] = adj

    return adj_pad

def atom_padding(atom, args):
    #get list of atom strings with diff shapes

    atom.extend([args.atom_pad_idx for _ in range(args.max_adj_len - len(atom))])
        
    return atom

def get_max_adj_len(input_list, args):
    for i, x in enumerate(input_list):
        imol = Chem.MolFromSmiles(x[1])
        iadjtmp = Chem.rdmolops.GetAdjacencyMatrix(imol)
        if args.max_adj_len < iadjtmp.shape[0] :
            args.max_adj_len = iadjtmp.shape[0]
    return args



def get_GCN_features(x, atom_dict, bond_dict, args):
    #   x : smiles string

    drug_adj, drug_atoms, drug_bonds = get_drug_gcn_feat(x, atom_dict, bond_dict)
    #   padding for adj mat and atom list
    drug_adj = adj_padding(drug_adj, args)
#    drug_atoms = atom_padding(drug_atoms, args)

    #   get adj mats into coo matrix by its values

    #   1. : Single
    #   1.5 : Aromatic
    #   2.0 : Double
    #   3.0 : Triple
    
    drug_adj = np.vstack([sparse.coo_matrix(drug_adj).row,
                          sparse.coo_matrix(drug_adj).col,
                          sparse.coo_matrix(drug_adj).data])
    #   Returned drug adj mat is 3 x num edges

    return drug_atoms, drug_adj

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

#    ppi_nx.remove_nodes_from(set(ppi_nx.nodes) - common_genes)
    ppi_nx = nx.subgraph(ppi_nx, list(common_genes)).copy()

    common_symbols = [x for x in gene_info.loc[:, 'pr_gene_symbol'].values.tolist()\
                        if x in common_genes]

    #   remove isolated nodes
    ppi_nx = ppi_nx.subgraph(common_symbols).copy()
#    non_isol_nodes = set(ppi_nx.nodex) - set(nx.isolates(ppi_nx))
    isol_nodes = nx.isolates(ppi_nx)
    common_symbols = [x for x in common_symbols if x not in isol_nodes]
#    ppi_nx = ppi_nx.subgraph(non_isol_nodes) 

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

    print(g2v_embedding.shape)

    #   get adj as COO row x col form
    ppi_adj = np.array([sparse.coo_matrix(ppi_adj).row, 
                        sparse.coo_matrix(ppi_adj).col])

    with open('ppi_'+args.gex_feat+'_feats.pkl', 'wb') as f:
        pickle.dump((ppi_adj, g2v_embedding, get_gex_idxs), f)

    args.num_genes = len(get_gex_idxs)

    print('num PPI extracted genes : '+str(args.num_genes))
        
    return gene2vecdict, gene_info, ppi_adj, ppi_nx, g2v_embedding, get_gex_idxs, args

def To2class_pred(test_label, test_preds, test_probas):
    test_2class_labels = []
    test_2class_preds = []
    test_2class_probas = []
    
    for i, x in enumerate(test_label):
        if x == 2:
            test_2class_labels.append(1)
        else:
            test_2class_labels.append(x)


    test_probas = np.asarray(test_probas)
    test_2class_probas = torch.softmax(torch.tensor([test_probas[:, 0], test_probas[:, 2]]).transpose(1,0), dim = 0)
    test_2class_preds = np.argmax(test_2class_probas, axis = -1)
    return test_2class_labels, test_2class_preds, test_2class_probas

def multi_roc_auc_score(y_test, y_proba, args, average="macro"):
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    if args.num_classes != 2:
        lb.fit([i for i in range(args.num_classes)])
        y_test = lb.transform(y_test)
    return roc_auc_score(y_test, y_proba, average=average)

def multi_aupr_score(y_test, y_proba, args, average="macro"):
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    if args.num_classes != 2:
        lb.fit([i for i in range(args.num_classes)])
        y_test = lb.transform(y_test)
    return average_precision_score(y_test, y_proba, average=average)

def Print_3class_Scores(test_labels, test_preds, test_probas, test_drug_names, args):

    test_drug_labels_avg, test_drug_preds_avg, test_drug_probas_avg, \
    test_drug_names_avg = Get_Drugwise_Preds(test_labels, test_preds, \
        test_probas, test_drug_names)

    print('=    '*10)
    print("Average Test F1 Score")
    print('macro : '+ str(f1_score(test_labels, test_preds, average = 'macro')))
    print('micro : '+ str(f1_score(test_labels, test_preds, average = 'micro')))
    print('weighted : '+ str(f1_score(test_labels, test_preds, average = 'weighted')))


    print("Average Test AUROC Score")
    print('macro : '+ str(multi_roc_auc_score(
                        test_labels, test_probas, args, average = 'macro')))
    print('micro : '+ str(multi_roc_auc_score(
                        test_labels, test_probas, args, average = 'micro')))
    print('weighted : '+ str(multi_roc_auc_score(
                        test_labels, test_probas, args, average = 'weighted')))
    print("Average Test AUPR Score")
    print('macro : '+ str(multi_aupr_score(
                        test_labels, test_probas, args, average = 'macro')))
    print('micro : '+ str(multi_aupr_score(
                        test_labels, test_probas, args, average = 'micro')))
    print('weighted : '+ str(multi_aupr_score(
                        test_labels, test_probas, args, average = 'weighted')))


    print('=    '*10)
    print("Drug_wise Test F1 Score")
    print('macro : '+ str(f1_score(test_drug_labels_avg, 
                            test_drug_preds_avg, average = 'macro')))
    print('micro : '+ str(f1_score(test_drug_labels_avg,
                            test_drug_preds_avg, average = 'micro')))
    print('weighted : '+ str(f1_score(test_drug_labels_avg,
                            test_drug_preds_avg, average = 'weighted')))

    print("Drug_wise Test AUROC Score")
    print('macro : '+ str(multi_roc_auc_score(test_drug_labels_avg, 
                            test_drug_probas_avg, args, average = 'macro')))
    print('micro : '+ str(multi_roc_auc_score(test_drug_labels_avg,
                            test_drug_probas_avg, args, average = 'micro')))
    print('weighted : '+ str(multi_roc_auc_score(test_drug_labels_avg,
                            test_drug_probas_avg, args, average = 'weighted')))
    print("Drug_wise Test AUPR Score")
    print('macro : '+ str(multi_aupr_score(test_drug_labels_avg, 
                            test_drug_probas_avg, args, average = 'macro')))
    print('micro : '+ str(multi_aupr_score(test_drug_labels_avg,
                            test_drug_probas_avg, args, average = 'micro')))
    print('weighted : '+ str(multi_aupr_score(test_drug_labels_avg,
                            test_drug_probas_avg, args, average = 'weighted')))

    print('=    '*10)

def Return_3class_Scores(test_labels, test_preds, test_probas, test_drug_names, args):

    test_drug_labels_avg, test_drug_preds_avg, test_drug_probas_avg, \
    test_drug_names_avg = Get_Drugwise_Preds(test_labels, test_preds, \
        test_probas, test_drug_names)

    avg_f1_macro = f1_score(test_labels, test_preds, average = 'macro')
    avg_f1_micro = f1_score(test_labels, test_preds, average = 'micro')
    avg_f1_weighted = f1_score(test_labels, test_preds, average = 'weighted')
    avg_auc_macro = multi_roc_auc_score(
                        test_labels, test_probas, args, average = 'macro')
    avg_auc_micro = multi_roc_auc_score(
                        test_labels, test_probas, args, average = 'micro')
    avg_auc_weighted = multi_roc_auc_score(
                        test_labels, test_probas, args, average = 'weighted')
    avg_aupr_macro = multi_aupr_score(
                        test_labels, test_probas, args, average = 'macro')
    avg_aupr_micro = multi_aupr_score(
                        test_labels, test_probas, args, average = 'micro')
    avg_aupr_weighted = multi_aupr_score(
                        test_labels, test_probas, args, average = 'weighted')
    drug_f1_macro = f1_score(test_drug_labels_avg,
                            test_drug_preds_avg, average = 'macro')
    drug_f1_micro = f1_score(test_drug_labels_avg,
                            test_drug_preds_avg, average = 'micro')
    drug_f1_weighted = f1_score(test_drug_labels_avg,
                            test_drug_preds_avg, average = 'weighted')
    drug_auc_macro = multi_roc_auc_score(test_drug_labels_avg, 
                            test_drug_probas_avg, args, average = 'macro')
    drug_auc_micro = multi_roc_auc_score(test_drug_labels_avg,
                            test_drug_probas_avg, args, average = 'micro')
    drug_auc_weighted = multi_roc_auc_score(test_drug_labels_avg,
                            test_drug_probas_avg, args, average = 'weighted')
    drug_aupr_macro = multi_aupr_score(test_drug_labels_avg,
                            test_drug_probas_avg, args, average = 'macro')
    drug_aupr_micro = multi_aupr_score(test_drug_labels_avg,
                            test_drug_probas_avg, args, average = 'micro')
    drug_aupr_weighted = multi_aupr_score(test_drug_labels_avg,
                            test_drug_probas_avg, args, average = 'weighted')

    return [avg_f1_macro, avg_f1_micro, avg_f1_weighted,
                            avg_auc_macro, avg_auc_micro, avg_auc_weighted,
                            avg_aupr_macro, avg_aupr_micro, avg_aupr_weighted,
                            drug_f1_macro, drug_f1_micro, drug_f1_weighted,
                            drug_auc_macro, drug_auc_micro, drug_auc_weighted,
                            drug_aupr_macro, drug_aupr_micro, drug_aupr_weighted]

