import numpy as np
import pandas as pd
import pickle
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import gradcheck

from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score

from utils2 import *


def TrainAndTest(train_loader, valid_loader, test_loader, model, optimizer, 
        loss_fn, scheduler, device, ppi_adj, get_gex_idxs, n_sample, args):

    for epoch in range(args.n_epochs):
        #   Train set

        train_losses = []
        train_pred = []
        train_proba = []
        train_label = []

        for x in train_loader:
            proba = model(x, ppi_adj,  get_gex_idxs,
                            device, args, epoch, training = True)

            label = x[4].to(device)
            
            loss = loss_fn(proba, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            proba = model(x, ppi_adj, get_gex_idxs,
                            device, args, epoch, training = True)

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
                                device, args, epoch, training = False)
            elif 'attn' in args.model:
                proba = model(x, get_gex_idxs,
                                device, args, training = True)
            else:
                proba = model(x, device, args, training = False)

            label = x[4].to(device)
            drug_name = x[5]

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
        print(f1_score(train_label, train_pred))

        print("Valid F1 Score")
        print('valid_losses : ' + str(np.mean(valid_losses)))

        Print_Scores(valid_label, valid_pred, valid_proba)

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


def ValidAndTest(valid_loader, test_loader, model, optimizer, 
        loss_fn, scheduler, device, ppi_adj, get_gex_idxs, args):

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
            test_pred, test_proba, test_label, test_drug_names
