import torch 
import torch.nn.functional as F
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import pandas as pd 
import numpy as np 
import os 
import csv

import sys 
from timeit import default_timer
import operator
from functools import reduce
import torch.utils.checkpoint as checkpoint
import importlib.util

# from tqdm import tqdm


def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c


def scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def LR_schedule(learning_rate, steps, scheduler_step, scheduler_gamma):
        return learning_rate * np.power(scheduler_gamma, (steps // scheduler_step)) 

def batched_extract_embeddings(model, smiles_list, batch_size=128, device="cuda"):
    all_idx = []
    all_token_embeddings = []

    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        idx, token_emb, smiles_emb = model.extract_embeddings(batch)
        
        all_idx.append(idx.cpu())
        all_token_embeddings.append(token_emb.cpu())
        all_smiles_embeddings.append(smiles_emb.cpu())

    return (
        torch.cat(all_idx, dim=0),
        torch.cat(all_token_embeddings, dim=0),
        torch.cat(all_smiles_embeddings, dim=0),
    )


def load_smi_ted_explicit():
    # Base path to this script
    base_dir = os.path.dirname(__file__)

    # Path to the smi_ted_light folder
    light_path = os.path.abspath(os.path.join(base_dir, "smi_ted_light"))
    
    # Make sure Python can find fast_transformers inside smi_ted_light
    if light_path not in sys.path:
        sys.path.insert(0, light_path)

    # Path to load.py inside smi_ted_light
    load_path = os.path.join(light_path, "load.py")
    print("Trying to load:", load_path)

    if not os.path.isfile(load_path):
        raise FileNotFoundError(f"Cannot find load.py at {load_path}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("smi_infer_load", load_path)
    smi_infer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smi_infer)

    return smi_infer.Smi_ted, smi_infer.MolTranBertTokenizer