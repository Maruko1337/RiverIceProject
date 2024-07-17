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

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=2, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    return parser.parse_args()

def setup_cuda(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    return args

def objective(trial):
    args = parse_arguments()
    args = setup_cuda(args)

    model_name = "GCN"
    area_of_interest = "lake"
    test_years = ["2017"]
    to_mask = False

    for test_year in test_years:
        hidden = 4
        print("Running experiment for year:", test_year)

        # Hyperparameters to tune
        lr = 0.8
        weight_decay = 5e-3
        dropout = 0.2
        n_features = 2
        n_class = 2
        n_layers = 4
        nNodes = 2131 if area_of_interest == "lake" else 1181

        # Model and optimizer
        model = GCN(nfeat=n_features, nhid=hidden, nclass=n_class, dropout=dropout, nNodes=nNodes)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        if args.cuda:
            model.cuda()

        def train(epoch, adj, features, labels):
            model.train()
            optimizer.zero_grad()
            output = model(features, adj)
            weights = calculate_weights(labels)
            loss_train = F.cross_entropy(output, labels, weight=weights)
            acc_train = accuracy(output, labels)
            f1_train = f1_score(output, labels)
            loss_train.backward()
            optimizer.step()
            print(f'Epoch: {epoch+1}, loss_train: {loss_train.item():.4f}, acc_train: {acc_train:.4f}, f1_train: {f1_train:.4f}')
            return loss_train.item(), acc_train, f1_train

        def train_model(path, train_files):
            train_loss_list, train_acc_list, train_f1_list = [], [], []
            for file_name in train_files:
                if not file_name.startswith(test_year):
                    file_path = os.path.join(path, file_name)
                    print("Training on:", file_path)
                    adj, features, labels, _ = load_data(file_path)
                    if args.cuda:
                        features, adj, labels = features.cuda(), adj.cuda(), labels.cuda()
                    output = model(features, adj)
                    weights = calculate_weights(labels)
                    train_loss = F.cross_entropy(output, labels, weight=weights)
                    train_acc = accuracy(output, labels)
                    train_f1 = f1_score(output, labels)
                    train_loss_list.append(train_loss.item())
                    train_acc_list.append(train_acc)
                    train_f1_list.append(train_f1)
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()
            return np.mean(train_loss_list), np.mean(train_acc_list), np.mean(train_f1_list)

        def cross_validation(path, n_splits=4):
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
            val_loss_list, val_acc_list, val_f1_list = [], [], []
            train_loss_list, train_acc_list, train_f1_list = [], [], []

            for train_idx, val_idx in kf.split(os.listdir(path)):
                train_files = [os.listdir(path)[i] for i in train_idx]
                val_files = [os.listdir(path)[i] for i in val_idx]
                train_loss, train_acc, train_f1 = train_model(path, train_files)
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
                    adj, features, labels, _ = load_data(file_path)
                    if args.cuda:
                        features, adj, labels = features.cuda(), adj.cuda(), labels.cuda()
                    output = model(features, adj)
                    weights = calculate_weights(labels)
                    val_loss = F.cross_entropy(output, labels, weight=weights)
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
                    adj, features, labels, _ = load_data(file_path)
                    if args.cuda:
                        features, adj, labels = features.cuda(), adj.cuda(), labels.cuda()
                    output = model(features, adj)
                    weights = calculate_weights(labels)
                    loss_test = F.cross_entropy(output, labels, weight=weights)
                    acc_test = accuracy(output, labels)
                    f1_test = f1_score(output, labels)
                    loss_list.append(loss_test)
                    acc_list.append(acc_test)
                    f1_list.append(f1_test)
                    if last:
                        visualize(file_name, adj, features, output, labels, area_of_interest, model_name)
            return np.mean(loss_list), np.mean(acc_list), np.mean(f1_list)

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
            loss_test, acc_test, f1_test = test(path)
            loss_test_list.append(loss_test)
            acc_test_list.append(acc_test)
            f1_test_list.append(f1_test)
            print(f"Epoch {epoch+1}: Test Loss={loss_test}, Test Accuracy={acc_test}, Test F1={f1_test}")

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

    return objective

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    print(f"Best trial: {study.best_trial.value}")
    print(f"Best parameters: {study.best_trial.params}")
