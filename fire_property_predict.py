# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
# from trainers import TrainerRegressor
#from utils import RMSELoss, get_optim_groups, get_optim_groups_freeze_encoder
#from utils import RMSELoss, get_optim_groups, get_optim_groups_freeze_encoder

# Data
import pandas as pd
import numpy as np

# Standard library
import args_nl as args
import os
import csv
from models import prediction_Model, prediction_Model_2, Decoder, Decoder2, nonlinear_regression_model2, Star_encoder
from utils import load_smi_ted_explicit, count_params

from Canonicalize_for_Optimize import canonicalize_smiles


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print(f'>> Device being used: {device}')
    else:
        print(f'>> Device being used: {device} ({torch.cuda.get_device_name(0)})')

    # load dataset
    #smiles_idx = 0
    print(f">> Load example psmiles, smiles idx = {config.smiles_idx}.")
    smiles_idx = config.smiles_idx
    #df_init = pd.read_csv(f"example_smiles_new.csv", encoding='utf-8-sig')
    df_init = pd.read_csv(f"example_smiles.txt", delimiter=",")
    smile = df_init['smiles'][smiles_idx-1]
    #smile = df_init['smiles_canonicalized'][config.smiles_idx]
    # embeddings = torch.load(f"example_smiles_emb.pt", weights_only=True)[config.smiles_idx, :]

    print(f">> Load pre-trained encoder.")
    base_dir = os.getcwd()
    model_dir = f'{base_dir}/checkpoint/Star_Lang_1'
    star_model_dir = os.path.join(model_dir, 'star', f'star_encoder_lr_{float(3e-5)}_gamma_{float(0.7)}_wd_{float(1e-5)}')
    star_encoder_filename = '%s/star_encoder.ckpt' % (star_model_dir)
    star_encoder = Star_encoder().to(device)
    star_encoder.load_state_dict(torch.load(star_encoder_filename, weights_only=True))
    star_encoder.eval()
    # print(f'>> Total parameters of star encoder: {count_params(star_encoder)}')


    # print(f">> Load pre-trained smi-ted model.")
    Smi_ted, MolTranBertTokenizer = load_smi_ted_explicit()
    tokenizer = MolTranBertTokenizer(f"{base_dir}/smi_ted_light/bert_vocab_curated.txt")
    Smi_ted_token = Smi_ted(tokenizer) 
    Smi_ted_token.load_checkpoint(f"{base_dir}/smi_ted_light/smi-ted-Light_40.pt")
    Smi_ted_token.eval()

    print(f'>> Loading pretrained smi-ted model for tig..')
    # model_tig = load_smi_ted(folder=config.model_path_tig, ckpt_filename=config.ckpt_filename_tig, n_output=config.n_output, eval=True).to(device)
    model_tig = nonlinear_regression_model2().to(device)
    model_tig.load_state_dict(torch.load(f"{base_dir}/checkpoint/optimal_model/Tig_best_model.ckpt", weights_only=True, map_location="cuda"))
    print(f'>> Loading pretrained smi-ted model for phrr..')
    # model_qpua_pk = load_smi_ted(folder=config.model_path_qpua_pk, ckpt_filename=config.ckpt_filename_qpua_pk, n_output=config.n_output, eval=True).to(device)
    model_qpua_pk = nonlinear_regression_model2().to(device)
    model_qpua_pk.load_state_dict(torch.load(f"{base_dir}/checkpoint/optimal_model/pHRR_best_model.ckpt", weights_only=True, map_location="cuda"))
    print(f'>> Loading pretrained smi-ted model for sea..')
    # model_sea = load_smi_ted(folder=config.model_path_sea, ckpt_filename=config.ckpt_filename_sea, n_output=config.n_output, eval=True).to(device)
    model_sea = nonlinear_regression_model2().to(device)
    model_sea.load_state_dict(torch.load(f"{base_dir}/checkpoint/optimal_model/sea_best_model.ckpt", weights_only=True, map_location="cuda")) 
    print(f'>> Loading pretrained smi-ted model for co..')
    # model_co = load_smi_ted(folder=config.model_path_co, ckpt_filename=config.ckpt_filename_co, n_output=config.n_output, eval=True).to(device)
    model_co = nonlinear_regression_model2().to(device)
    model_co.load_state_dict(torch.load(f"{base_dir}/checkpoint/optimal_model/co_best_model.ckpt", weights_only=True, map_location="cuda"))


    model_tig.eval()
    model_qpua_pk.eval()
    model_sea.eval()
    model_co.eval()


    print(f">> Begin property prediction.")
    with torch.no_grad():
        idx_t, token_embedding_t, _ = Smi_ted_token.extract_embeddings(smile)
        idx_t, token_embedding_t = idx_t.to(device), token_embedding_t.to(device)
        mask_439 = (idx_t == 439)
        positions = torch.nonzero(mask_439, as_tuple=False)       # size [numbers,2]
        selected_embeddings = token_embedding_t[positions[:, 0], positions[:, 1], :]   # select * token 
        star_embeddings = star_encoder(selected_embeddings)
        token_embedding_t[positions[:, 0], positions[:, 1], :] = star_embeddings

        # latent = vae.encoder(token_embedding_t.view(-1, 202*768))
        embedding = token_embedding_t.sum(1)#latent.detach().cpu()

        tig_pred = model_tig(embedding).item()
        qpua_pk_pred = model_qpua_pk(embedding).item()
        sea_pred = model_sea(embedding).item()
        co_pred = model_co(embedding).item()

        print(f">> For pSMILES {smile}, the predicted tig: {tig_pred:.6f}, phrr: {qpua_pk_pred:.6f}, sea: {sea_pred:.6f}, co: {co_pred:.6f}.")
    print(f">> Property prediction ends.")


if __name__ == '__main__':
    parser = args.get_parser()
    config = parser.parse_args()
    main(config)
