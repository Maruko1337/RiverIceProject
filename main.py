from __future__ import division, print_function
import time
import pandas as pd
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import cv2
import pickle
import optuna
import sklearn.metrics as sk_metrics
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import networkx as nx
from utils import *
from models import *
from constants import *

from torch_geometric.data import DataLoader


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--LR', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--WD', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--HIDDEN_SIZE', type=int, default=2, help='Number of HIDDEN_SIZE units.')
    parser.add_argument('--DROPOUT', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    return parser.parse_args()

def setup_cuda(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    return args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for test_year in TEST_YEARS:
    def objective(trial):
        args = parse_arguments()
        args = setup_cuda(args)
        print(f"test years are {TEST_YEARS}")
        
        print("Running experiment for year:", test_year)

        nNodes = 2131 if AOI == "lake" else 1181

        # Model and optimizer
        if MODEL_NAME == "ResGCN":
            model = ResGCN(nfeat=N_FEATURES, nhid=HIDDEN_SIZE, nclass=N_CLASS, dropout=DROPOUT)
        elif MODEL_NAME == "GAT":
            model = GAT(nfeat=N_FEATURES, nhid=HIDDEN_SIZE, nclass=N_CLASS, dropout=DROPOUT)
        else:
            model = GCN(nfeat=N_FEATURES, nhid=HIDDEN_SIZE, nclass=N_CLASS, dropout=DROPOUT, nNodes=nNodes)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        for param in model.parameters():
            print(param.requires_grad)
        
        if args.cuda: 
            model.cuda()
            print(f"cuda is avaiable")

        def train_model(path, train_files):
            model.train()
            train_loss_list, train_acc_list, train_f1_list = [], [], []
            need_break = False
            for file_name in train_files:
                if not file_name.startswith(test_year):
                    file_path = os.path.join(path, file_name)
                    print("Training on:", file_path)
                    if TO_MASK:
                        adj, features, labels, mask = load_data(file_path, mask = True)
                        if AUGMENT:
                            #-------------------
                            data = convert_to_data_object(adj, features, labels, mask)
                            data_list = [data]
                            criterion = torch.nn.CrossEntropyLoss()

                            train_loader = DataLoader(data_list, batch_size=1, shuffle=True)  # Adjust batch_size as needed
                            for data in train_loader:
                                data = data.to(device)
                                
                                # Apply augmentations
                                data = add_gaussian_noise(data)
                                data = feature_dropout(data)
                                data = feature_scaling(data)
                                
                                optimizer.zero_grad()
                                out = model(data.x, data.adj)
                                loss = criterion(out[data.train_mask], data.y[data.train_mask])
                                loss.backward()
                                optimizer.step()
                                
                                train_acc = accuracy(out[data.train_mask], data.y[data.train_mask])
                                train_f1 = f1_score(out[data.train_mask], data.y[data.train_mask])
                                
                                train_loss_list.append(loss.detach().cpu().item())
                                
                                train_acc_list.append(train_acc)
                                train_f1_list.append(train_f1)
                            continue
                            #-----------------------------
                        
                        mask_rate = sum(mask) / len(mask) * 100
                        # print(f"label rate = {mask_rate}")
                        mask = torch.tensor(mask, dtype=torch.bool)
                        # print(f"label length = {len(labels)}, mask len = {len(mask)}")
                        # print(f"label mask = 1: {labels[mask]}")
                    else:
                        adj, features, labels = load_data(file_path, mask=False)
                    if args.cuda:
                        features, adj, labels = features.cuda(), adj.cuda(), labels.cuda()
                    optimizer.zero_grad()

                    # print(f"features is {features}")
                    

                    # features = features.to(torch.float32).requires_grad_()
                    # adj = adj.to(torch.float32).requires_grad_()
                    
                    output = model(features, adj)
                    if TO_MASK:
                        masked_output = output[mask]
                        masked_labels = labels[mask]
                        # print(f"masked_output = {masked_output}, masked_label = {masked_labels}")
                        # print(f"output = {output}, label = {labels}")

                        
                        weights = calculate_weights(masked_labels)
                        
                        # ----------------------------------
                        
                        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
                        loss_BCE = nn.BCEWithLogitsLoss(pos_weight=weights)
                        preds = masked_output   

                        # Convert masked_labels to float
                        preds = preds.float()
                        
                        one_hot_labels = torch.zeros(masked_labels.size(0), 2, device = 'cuda')  # Create a tensor of zeros
                        one_hot_labels.scatter_(1, masked_labels.unsqueeze(1), 1)  # Ensure masked_labels is [1542, 1]
                        
                        one_hot_labels = one_hot_labels.float()
                        
                        train_loss = loss_BCE(preds, one_hot_labels)
                        
                        
                        #-------------------------------------------
                        print(f"loss is {train_loss}")
                        print(f"weights = {weights}")
                        # print(f"masked output is {masked_output}, masked labels is {masked_labels}, weights is {weights}, loss test is {loss_test}")
                        train_acc = accuracy(masked_output, masked_labels)
                        train_f1 = f1_score(masked_output, masked_labels)
                    else:
                        weights = calculate_weights(labels)
                        train_loss = F.cross_entropy(output, labels, weight=weights)
                        # print(f"masked output is {masked_output}, masked labels is {masked_labels}, weights is {weights}, loss test is {loss_test}")
                        train_acc = accuracy(output, labels)
                        train_f1 = f1_score(output, labels)

                    if torch.isnan(train_loss):
                        print(f"date {file_name} need break---------------------------")
                        break
                    else:
                        print(f"date {file_name} ---------------------------")
                    train_loss.backward()
                    
                    optimizer.step()
                    
                    
                    
                    train_loss_list.append(train_loss.detach().cpu().item())
                    train_acc_list.append(train_acc)
                    train_f1_list.append(train_f1)
                    
                    if need_break:
                        print(f"date {file_name} need break---------------------------")
                        break
            
            print(f"train loss list is {train_loss_list}")
            return np.mean(train_loss_list), np.mean(train_acc_list), np.mean(train_f1_list)

        def cross_validation(path, n_splits=4):
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
            val_loss_list, val_acc_list, val_f1_list = [], [], []
            train_loss_list, train_acc_list, train_f1_list = [], [], []

            for train_idx, val_idx in kf.split(os.listdir(path)):
                train_files = [os.listdir(path)[i] for i in train_idx]
                val_files = [os.listdir(path)[i] for i in val_idx]
                train_loss, train_acc, train_f1 = train_model(path, train_files)
                
                # train_loss_tensor = torch.tensor(train_loss, requires_grad=True) 
                # train_loss_tensor.backward()
                # optimizer.step()
                
                val_loss, val_acc, val_f1 = validate(path, val_files)
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                train_f1_list.append(train_f1)
                val_loss_list.append(val_loss)
                val_acc_list.append(val_acc)
                val_f1_list.append(val_f1)

            return np.mean(train_loss_list), np.mean(train_acc_list), np.mean(train_f1_list), np.mean(val_loss_list), np.mean(val_acc_list), np.mean(val_f1_list)

        def validate(path, val_files):
            model.eval()
            val_loss_list, val_acc_list, val_f1_list = [], [], []
            for file_name in val_files:
                if not file_name.startswith(test_year):
                    file_path = os.path.join(path, file_name)
                    
                    if TO_MASK:
                        adj, features, labels, mask = load_data(file_path, mask = True)
                        mask_rate = sum(mask) / len(mask) * 100
                        # print(f"label rate = {mask_rate}")
                        mask = torch.tensor(mask, dtype=torch.bool)
                        # print(f"label length = {len(labels)}, mask len = {len(mask)}")
                        # print(f"label mask = 1: {labels[mask]}")
                    else:
                        adj, features, labels = load_data(file_path, mask=False)
                    if args.cuda:
                        features, adj, labels = features.cuda(), adj.cuda(), labels.cuda()
                    # optimizer.zero_grad()
                    output = model(features, adj)
                    if TO_MASK:
                        output = output[mask]
                        labels = labels[mask]
                        
                    weights = calculate_weights(labels)
                    
                    # val_loss = F.cross_entropy(output, labels, weight=weights)
                    
                    #---------------------
                    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
                    loss_BCE = nn.BCEWithLogitsLoss(pos_weight=weights)
                    preds = output   

                    # Convert masked_labels to float
                    preds = preds.float()
                    
                    one_hot_labels = torch.zeros(labels.size(0), 2, device = 'cuda')  # Create a tensor of zeros
                    one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)  # Ensure masked_labels is [1542, 1]
                    
                    one_hot_labels = one_hot_labels.float()
                    
                    val_loss = loss_BCE(preds, one_hot_labels)
                    
                    #------------------
                    val_acc = accuracy(output, labels)
                    val_f1 = f1_score(output, labels)
                    val_loss_list.append(val_loss.item())
                    val_acc_list.append(val_acc)
                    val_f1_list.append(val_f1)
            return np.mean(val_loss_list), np.mean(val_acc_list), np.mean(val_f1_list)

        def test(path, last=False):
            model.eval()
            loss_list, acc_list, f1_list = [], [], []
            for file_name in os.listdir(path):
                if file_name.startswith(test_year):
                    file_path = os.path.join(path, file_name)
                    
                    if TO_MASK:
                        adj, features, labels, mask = load_data(file_path, mask = True)
                        mask_rate = sum(mask) / len(mask) * 100
                        # print(f"label rate = {mask_rate}")
                        mask = torch.tensor(mask, dtype=torch.bool)
                        # print(f"label length = {len(labels)}, mask len = {len(mask)}")
                        # print(f"label mask = 1: {labels[mask]}")
                    else:
                        adj, features, labels = load_data(file_path, mask=False)
                    if args.cuda:
                        features, adj, labels = features.cuda(), adj.cuda(), labels.cuda()
                    # optimizer.zero_grad()
                    output = model(features, adj)
                    # print(f"output is {output}, features is {features}, adj is {adj}")
                    
                    if TO_MASK:
                        masked_output = output[mask]
                        masked_labels = labels[mask]
                        
                        weights = calculate_weights(masked_labels)
                        
                        
                        #---------------------
                        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
                        loss_BCE = nn.BCEWithLogitsLoss(pos_weight=weights)
                        preds = masked_output   

                        # Convert masked_labels to float
                        preds = preds.float()
                        
                        one_hot_labels = torch.zeros(masked_labels.size(0), 2, device = 'cuda')  # Create a tensor of zeros
                        one_hot_labels.scatter_(1, masked_labels.unsqueeze(1), 1)  # Ensure masked_labels is [1542, 1]
                        
                        one_hot_labels = one_hot_labels.float()
                        
                        loss_test = loss_BCE(preds, one_hot_labels)
                        
                        #------------------
                    
                        # loss_test = F.cross_entropy(masked_output, masked_labels, weight=weights)
                        # print(f"masked output is {masked_output}, masked labels is {masked_labels}, weights is {weights}, loss test is {loss_test}")
                        acc_test = accuracy(masked_output, masked_labels)
                        f1_test = f1_score(masked_output, masked_labels)
                    else:
                        weights = calculate_weights(labels)
                        loss_test = F.cross_entropy(output, labels, weight=weights)
                        # print(f"masked output is {masked_output}, masked labels is {masked_labels}, weights is {weights}, loss test is {loss_test}")
                        acc_test = accuracy(output, labels)
                        f1_test = f1_score(output, labels)
                    loss_list.append(loss_test)
                    acc_list.append(acc_test)
                    f1_list.append(f1_test)
                    if last:
                        visualize(file_name, adj, features, output, labels, AOI, MODEL_NAME)
                        
            

            
            # Assuming acc_list, loss_list, f1_list are lists of PyTorch tensors
            # acc_list_cpu = [acc_item.cpu().item() for acc_item in acc_list]
            # loss_list_cpu = [loss_item.cpu().item() for loss_item in loss_list]
            # f1_list_cpu = [f1_item.cpu().item() for f1_item in f1_list]
            loss_list = [loss.detach().cpu().numpy() for loss in loss_list]
            # return np.mean(loss_list_cpu), np.mean(acc_list_cpu), np.mean(f1_list_cpu)
            return np.mean(loss_list), np.mean(acc_list), np.mean(f1_list)
        if AOI == "lake":
            path = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/lake_stlawrence/new_label/graphs"
        else:
            path = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/7relabel_graphs"
        
        # Training process
        loss_train_list, acc_train_list, f1_train_list = [], [], []
        loss_val_list, acc_val_list, f1_val_list = [], [], []
        loss_test_list, acc_test_list, f1_test_list = [], [], []

        for epoch in range(args.epochs):
            mean_train_loss, mean_train_acc, mean_train_f1, mean_val_loss, mean_val_acc, mean_val_f1 = cross_validation(path)
            loss_train_list.append(mean_train_loss)
            acc_train_list.append(mean_train_acc)
            f1_train_list.append(mean_train_f1)
            loss_val_list.append(mean_val_loss)
            acc_val_list.append(mean_val_acc)
            f1_val_list.append(mean_val_f1)
            if epoch == args.epochs - 1:
                loss_test, acc_test, f1_test = test(path, last = True)
            else:
                loss_test, acc_test, f1_test = test(path)
            loss_test_list.append(loss_test)
            acc_test_list.append(acc_test)
            f1_test_list.append(f1_test)
            print(f"Epoch {epoch+1}: Test Loss={loss_test}, Test Accuracy={acc_test}, Test F1={f1_test}")
            print(f"Epoch {epoch+1}: train Loss={mean_train_loss}, train Accuracy={mean_train_acc}, train F1={mean_train_f1}")
        # Plotting results
        total_epochs = range(1, len(loss_train_list) + 1)
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.plot(total_epochs, loss_train_list, 'b', label='Training loss')
        plt.plot(total_epochs, loss_test_list, 'r', label='Test loss')
        plt.plot(total_epochs, loss_val_list, 'g', label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training/Validation/Testing Loss')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(total_epochs, acc_train_list, 'b', label='Training acc')
        plt.plot(total_epochs, acc_test_list, 'r', label='Test acc')
        plt.plot(total_epochs, acc_val_list, 'g', label='Validation acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training/Validation/Testing Accuracy')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(total_epochs, f1_train_list, 'b', label='Training f1')
        plt.plot(total_epochs, f1_test_list, 'r', label='Test f1')
        plt.plot(total_epochs, f1_val_list, 'g', label='Validation f1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Training/Validation/Testing F1 Score')
        plt.legend()

        plt.savefig(f"output_figures/{test_year}_results.png")
        plt.show()

        return mean_val_loss

    


# Save the current stdout
# original_stdout = sys.stdout

# # Specify the file path where you want to save the printed output
# output_file_path = "output_file.txt"

# # Open the file in write mode
# with open(output_file_path, "w") as f:
#     # Redirect stdout to the file
#     sys.stdout = f

#     if __name__ == "__main__":
#         study = optuna.create_study(direction='minimize')
#         study.optimize(objective, n_trials=100)
#         print(f"Best trial: {study.best_trial.value}")
#         print(f"Best parameters: {study.best_trial.params}")
    
#     print("Printed output will be saved to output_file.txt")

#     # Restore the original stdout
#     sys.stdout = original_stdout

    if __name__ == "__main__":
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=1)
        print(f"Best trial: {study.best_trial.value}")
        print(f"Best parameters: {study.best_trial.params}")