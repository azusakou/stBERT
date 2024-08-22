import csv
import random
import os
import numpy as np
import torch

from sklearn.metrics import cluster

def eval_embedding(pred, embedding=None):
    sc = cluster.silhouette_score(embedding, pred, metric='euclidean')
    db = cluster.davies_bouldin_score(embedding, pred)
    return sc, db


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_dir(directory_path, new_folder_name):
    """Creates an expected directory if it does not exist"""
    directory_path = os.path.join(directory_path, new_folder_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path
