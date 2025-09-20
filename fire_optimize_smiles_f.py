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
from utilities3 import count_params
from utils import load_smi_ted_explicit

from Canonicalize_for_Optimize import canonicalize_smiles


def main(config):
    tig_range = [1, 600]
    qpua_pk_range = [19, 1761]
    sea_range = [24.8, 1300]
    co_range = [0.01, 0.075]
    c_tig = config.c_tig
    c_qpua = config.c_qpua
    c_sea = config.c_sea
    c_co = config.c_co

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print(f'>> Device being used: {device}')
    else:
        print(f'>> Device being used: {device} ({torch.cuda.get_device_name(0)})')

    # load dataset
    #smiles_idx = 0
    print(f">> Load example psmiles, smiles idx = {config.smiles_idx}.")
    smiles_idx = config.smiles_idx
    df_init = pd.read_csv(f"example_smiles_new.csv", encoding='utf-8-sig')
    # df_init = pd.read_csv(f"example_smiles.txt", delimiter=",")
    # smiles_init = df_init['smiles'][smiles_idx]
    smiles_init = df_init['smiles_canonicalized'][config.smiles_idx]
    # embeddings = torch.load(f"example_smiles_emb.pt", weights_only=True)[config.smiles_idx, :]

    print(f">> Load pre-trained encoder.")
    base_dir = f'/home/office5090/Desktop/cLLM'
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

    print(f">> Find the initialization embedding using the pre-trained encoder.")
    with torch.no_grad():
        idx_t, token_embedding_t, _ = Smi_ted_token.extract_embeddings(smiles_init)
        idx_t, token_embedding_t = idx_t.to(device), token_embedding_t.to(device)
        mask_439 = (idx_t == 439)
        positions = torch.nonzero(mask_439, as_tuple=False)       # size [numbers,2]
        selected_embeddings = token_embedding_t[positions[:, 0], positions[:, 1], :]   # select * token 
        star_embeddings = star_encoder(selected_embeddings)
        token_embedding_t[positions[:, 0], positions[:, 1], :] = star_embeddings

        # latent = vae.encoder(token_embedding_t.view(-1, 202*768))
        embeddings = token_embedding_t.sum(1)#latent.detach().cpu()

    
    # Load vocabulary
    with open("bert_vocab_curated.txt", "r") as f:
        vocab = [line.strip() for line in f]
    id_to_token = {i: token for i, token in enumerate(vocab)}

    

    print(f'>> Loading pretrained smi-ted model for tig..')
    # model_tig = load_smi_ted(folder=config.model_path_tig, ckpt_filename=config.ckpt_filename_tig, n_output=config.n_output, eval=True).to(device)
    model_tig = nonlinear_regression_model2().to(device)
    model_tig.load_state_dict(torch.load(f"{base_dir}/checkpoint/optimal_model_sep15/tig_model_lr1e-05_params21245953.ckpt", weights_only=True, map_location="cuda"))
    print(f'>> Loading pretrained smi-ted model for phrr..')
    # model_qpua_pk = load_smi_ted(folder=config.model_path_qpua_pk, ckpt_filename=config.ckpt_filename_qpua_pk, n_output=config.n_output, eval=True).to(device)
    model_qpua_pk = nonlinear_regression_model2().to(device)
    model_qpua_pk.load_state_dict(torch.load(f"{base_dir}/checkpoint/optimal_model_sep15/pHRR_model_lr1e-05_params21245953.ckpt", weights_only=True, map_location="cuda"))
    print(f'>> Loading pretrained smi-ted model for sea..')
    # model_sea = load_smi_ted(folder=config.model_path_sea, ckpt_filename=config.ckpt_filename_sea, n_output=config.n_output, eval=True).to(device)
    model_sea = nonlinear_regression_model2().to(device)
    model_sea.load_state_dict(torch.load(f"{base_dir}/checkpoint/optimal_model_sep15/sea_model_lr0.001_params21245953.ckpt", weights_only=True, map_location="cuda")) 
    print(f'>> Loading pretrained smi-ted model for co..')
    # model_co = load_smi_ted(folder=config.model_path_co, ckpt_filename=config.ckpt_filename_co, n_output=config.n_output, eval=True).to(device)
    model_co = nonlinear_regression_model2().to(device)
    model_co.load_state_dict(torch.load(f"{base_dir}/checkpoint/optimal_model_sep15/co_model_lr0.0001_params21245953.ckpt", weights_only=True, map_location="cuda"))

    print(f">> Load pre-trained polymer generation model.")
    lang_model = prediction_Model(n_embd= 768, mid_size = 768*4, n_vocab=2393).to(device)
    # lang_model = prediction_Model_2(n_embd = 768, mid_size1 = 768, mid_size2= 768 * 2, n_vocab = 2393).to(device)
    # print(f'Total parameters of lang model: {count_params(lang_model)}')
    # print(f">> Load decoder layer.")
    # decoder = Decoder(feature_size=202*768, mid_size=768*4, latent_size=768).to(device)
    decoder = Decoder2(feature_size=202*768, mid_size=768*4, mid_size_2=768*2, latent_size=768).to(device)
    # print(f'Total parameters of decoder layer: {count_params(decoder)}')

    checkpoint = torch.load(f"{base_dir}/checkpoint/decoder_predictor/model_params{count_params(lang_model) + count_params(decoder)}_lamb1_train.ckpt", weights_only=True)
    decoder.load_state_dict(checkpoint["decoder"])
    lang_model.load_state_dict(checkpoint["predictor"])

    model_tig.eval()
    model_qpua_pk.eval()
    model_sea.eval()
    model_co.eval()
    decoder.eval()
    lang_model.eval()

    tig_range = torch.tensor(tig_range).to(device)
    qpua_pk_range = torch.tensor(qpua_pk_range).to(device)
    sea_range = torch.tensor(sea_range).to(device)
    co_range = torch.tensor(co_range).to(device)


    save_embeddings = [embeddings.view([1, 768]).clone()]
    save_smiles = [(-1, smiles_init)]
    save_fire_props = []
    save_canonicalized_smiles = []

    print(f">> Begin optimization.")

    for ep in range(config.max_epochs):
        # Create a new tensor that requires gradients for each iteration
        embeddings = embeddings.clone().detach().to(device)
        embeddings.requires_grad_(True)
        
        tig = model_tig(embeddings)
        qpua_pk = model_qpua_pk(embeddings)
        sea = model_sea(embeddings)
        co = model_co(embeddings)
        # print(co.item())

        loss =  c_tig * torch.clamp(-tig / tig_range[1], min= tig_range[0] / tig_range[1]) + c_qpua * torch.clamp(qpua_pk / qpua_pk_range[1], min=qpua_pk_range[0] / qpua_pk_range[1]) + c_sea * torch.clamp(sea / sea_range[1], min= sea_range[0]/sea_range[1]) + c_co * torch.clamp(co / co_range[1], min=co_range[0]/co_range[1])
        save_fire_props.append([(torch.clamp(tig / tig_range[1], min= tig_range[0] / tig_range[1]) * tig_range[1]).item(),
                                (torch.clamp(qpua_pk / qpua_pk_range[1], min=qpua_pk_range[0] / qpua_pk_range[1]) *  qpua_pk_range[1]).item(),
                                (torch.clamp(sea / sea_range[1], min= sea_range[0]/sea_range[1]) * sea_range[1] ).item(),
                                (torch.clamp(co / co_range[1], min=co_range[0]/co_range[1]) * co_range[1]).item()])

        loss.backward()

        # Update embeddings using gradients
        with torch.no_grad():
            embeddings = embeddings - config.lr_start * embeddings.grad
            token_emb_rec = decoder(embeddings.view([1, 768]).detach().clone()).view(-1, 202, 768)
            logits = lang_model(token_emb_rec.view(-1, 768))
            log_probs = F.log_softmax(logits, dim=1) # [M*202, 2393]
            pred_idx = log_probs.view(-1, 202, 2393).argmax(-1)
            
            # Convert token IDs to tokens
            tokens = [id_to_token.get(int(idx), '<unk>') for idx in pred_idx.flatten().cpu().numpy()]
            smiles_tokens = [token for token in tokens if token not in ['<pad>', '<unk>', '<s>', '</s>', '<bos>', '<eos>']]
            smiles_string = ''.join(smiles_tokens)
            if not save_smiles or smiles_string != save_smiles[-1][-1]:
                save_smiles.append((ep, smiles_string))

                # Canonicalize the optimized smiles 
                canon_smiles = canonicalize_smiles(smiles_string)
                save_canonicalized_smiles.append((ep,canon_smiles))
        
        save_embeddings.append(embeddings.view([1, 768]).detach().clone())

    print(f">> End optimization.")

    print(f">> Begin saving the generated psmiles and properties.")
    # save_file = f'tig{config.c_tig}_phrr{config.c_qpua}_sea{config.c_sea}_co{config.c_co}'
    # base_paths = [
    # "optimized_embeddings",
    # "optimized_smiles",
    # "optimized_smiles_canonicalized",
    # "log_optimized_prop"
    # ]
    # [os.makedirs(f"{base_dir}/results_0920/{subdir}/{save_file}", exist_ok=True) for subdir in base_paths]
    

    # with open(f"{base_dir}/results_0920/optimized_embeddings/{save_file}/optimized_embeddings_{smiles_idx}_lr_{config.lr_start:.4f}.txt", "w") as f:
    #     for embd in save_embeddings:
    #         np.savetxt(f, embd.cpu().numpy(), fmt="%.6f")
    
    # with open(f"{base_dir}/results_0920/optimized_smiles/{save_file}/optimized_smiles_{smiles_idx}_lr_{config.lr_start:.4f}.txt", "w") as f:
    #     for ep, smiles in save_smiles:
    #         f.write(f"Epoch {ep}: {smiles}\n")
    
    # with open(f"{base_dir}/results_0920/optimized_smiles_canonicalized/{save_file}/optimized_smiles_canonicalized_{smiles_idx}_lr_{config.lr_start:.4f}.txt", "w") as f:
    #     for ep, smiles in save_canonicalized_smiles:
    #         f.write(f"Epoch {ep}: {smiles}\n")

    # with open(f"{base_dir}/results_0920/log_optimized_prop/{save_file}/log_optimized_props_{smiles_idx}_lr_{config.lr_start:.4f}.txt", "w") as f:
    #     f.write('tig50, qpua_pk50, sea, co\n')
    #     writer  = csv.writer(f)
    #     for row in save_fire_props:
    #         writer.writerow([f'{x:.6f}' for x in row])

    



if __name__ == '__main__':
    parser = args.get_parser()
    config = parser.parse_args()
    # print(config)
    main(config)
