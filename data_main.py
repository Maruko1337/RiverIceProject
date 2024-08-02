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

from torch_geometric.data import Data

def convert_to_data_object(adj, features, labels, mask):
    # Convert adjacency matrix to edge index
    adj = adj.coalesce()  # Ensure it's in COO format
    edge_index = torch.stack([adj._indices()[0], adj._indices()[1]], dim=0)
    
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.bool)
    
    return Data(x=x, edge_index=edge_index, y=y, train_mask=mask)

# Create a list of Data objects
data_list = []
if AOI == "lake":
    path = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/lake_stlawrence/new_label/graphs"
else:
    path = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/7relabel_graphs"
for file_path in path:  # Replace with your method to load each graph
    adj, features, labels, mask = load_data(file_path)
    data = convert_to_data_object(adj, features, labels, mask)
    data_list.append(data)
    for file_name in train_files:
        if not file_name.startswith(test_year):
            file_path = os.path.join(path, file_name)
            print("Training on:", file_path)
            if TO_MASK:
                adj, features, labels, mask = load_data(file_path, mask = True)