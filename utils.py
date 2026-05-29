import torch 
import torch.nn.functional as F
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
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


class SmiMultiLabelDataset(Dataset):
    def __init__(self, smi_path: str):
        # columns: idx, smiles, 48 labels
        df = pd.read_csv(smi_path, sep=r"\s+", header=None)
        if df.shape[1] != 50:
            raise ValueError(f"Expected 50 columns (id+smiles+48), got {df.shape[1]}")

        self.smiles = df.iloc[:, 1].astype(str).tolist()
        self.y = torch.tensor(df.iloc[:, 2:50].to_numpy(dtype="int64"), dtype=torch.float32)  # multi-label

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        return self.smiles[i], self.y[i]

def collate_smiles_labels(batch):
    smiles, y = zip(*batch)
    return list(smiles), torch.stack(y, dim=0)


def count_params(model):
    # Check if model is actually an instance, not a class
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected a nn.Module instance, but got {type(model)}. Make sure you're passing an instantiated model, not the class itself.")
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
    all_smiles_embeddings = []

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
    # print("Trying to load:", load_path)

    if not os.path.isfile(load_path):
        raise FileNotFoundError(f"Cannot find load.py at {load_path}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("smi_infer_load", load_path)
    smi_infer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smi_infer)

    return smi_infer.Smi_ted, smi_infer.MolTranBertTokenizer


class StringDataset(Dataset):
    """Dataset for handling lists of strings (like SMILES)"""
    def __init__(self, string_list, tokenizer=None, max_length=202):
        self.string_list = string_list
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.string_list)
    
    def __getitem__(self, idx):
        string = self.string_list[idx]
        if self.tokenizer is not None:
            # Tokenize the string if tokenizer is provided
            tokens = self.tokenizer.encode(string)
            # Pad or truncate to max_length
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            else:
                tokens.extend([0] * (self.max_length - len(tokens)))
            return torch.tensor(tokens)
        else:
            # Return the string as-is if no tokenizer
            return string


class FlattenedActivationDataset(Dataset):
    def __init__(self, activations):
        # Flatten activations from [M, 202, 768] to [M*202, 768]
        self.data = activations.view(-1, activations.shape[-1])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]  # Returns a single tensor [768]


class ActivationDataset(Dataset):
    def __init__(self, activations):
        # Keep activations as [M, 768]
        self.data = activations
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]  # Returns a single tensor [768]


class SMILESPropertyDataset(Dataset):
    def __init__(self, smiles_list, property_values):
        self.smiles = smiles_list
        self.properties = torch.tensor(property_values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        return self.smiles[idx], self.properties[idx]



def mutual_information_binary(x: torch.Tensor, y: torch.Tensor, log_base: float = 2.0):
    """
    x, y: 1D tensors of 0/1 values, same length, on GPU or CPU.
    Returns MI(x; y) in bits by default (log base 2).
    """
    assert x.shape == y.shape
    device = x.device

    x = x.to(torch.long)
    y = y.to(torch.long)
    n = x.numel()

    # Encode pairs (x,y) into 0..3: 2*x + y
    idx = 2 * x + y  # 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)

    # Joint counts and probs on GPU
    counts = torch.bincount(idx, minlength=4).to(device)  # shape [4]
    p_joint = counts.float() / n                          # shape [4]

    # Reshape to 2x2 joint table
    p_joint_2x2 = p_joint.view(2, 2)  # rows: x=0,1; cols: y=0,1

    # Marginals
    p_x = p_joint_2x2.sum(dim=1, keepdim=True)  # shape [2,1]
    p_y = p_joint_2x2.sum(dim=0, keepdim=True)  # shape [1,2]

    # Avoid log(0) / division by 0
    eps = 1e-12
    p_joint_safe = p_joint_2x2 + eps
    p_prod_safe = p_x @ p_y + eps

    # Choose log base
    log_fn = torch.log2 if log_base == 2.0 else torch.log
    if log_base not in (2.0, torch.e):
        # change of base: log_b(z) = ln(z) / ln(b)
        log_fn = lambda z: torch.log(z) / torch.log(torch.tensor(log_base, device=device))

    mi = (p_joint_safe * (log_fn(p_joint_safe) - log_fn(p_prod_safe))).sum()
    return mi
        