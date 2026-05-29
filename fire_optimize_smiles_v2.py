# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# Data
import pandas as pd
import numpy as np

# Standard library
import args_nl as args
import os
import csv
from models import (prediction_Model, Decoder2, star_encoder,
                    AutoEncoderLayer3)
from utils1 import load_smi_ted_explicit
from Canonicalize_for_Optimize import canonicalize_smiles

def main(config):
    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'>> Device being used: {device}')
    
    # 2. Load Scalers and Define Ranges
    sc = torch.load('checkpoint/optimal_model/scalers.pt', map_location=device, weights_only=False)
    z_tilde_mean = sc['X_mean'].to(device)
    z_tilde_std = sc['X_std'].to(device)
    
    y_mean = sc['Y_mean'].to(device)
    y_std = sc['Y_std'].to(device)
    
    # Property stats for denormalization
    tig_mean, tig_std = y_mean[0, 0], y_std[0, 0]
    qpua_pk_mean, qpua_pk_std = y_mean[0, 1], y_std[0, 1]
    sea_mean, sea_std = y_mean[0, 2], y_std[0, 2]
    co_mean, co_std = y_mean[0, 3], y_std[0, 3]

    # Standardized Ranges for Clamping
    tig_range = (torch.tensor([11.0, 898.0], device=device) - tig_mean) / tig_std
    qpua_pk_range = (torch.tensor([22.0, 2748.0], device=device) - qpua_pk_mean) / qpua_pk_std
    sea_range = (torch.tensor([0.001, 0.149], device=device) - sea_mean) / sea_std
    co_range = (torch.tensor([0.01, 0.45], device=device) - co_mean) / co_std
    
    # 3. Prepare Input Conditions
    flux_tensor = torch.tensor([[config.flux]], device=device, dtype=torch.float32)
    thk_tensor = torch.tensor([[config.thickness]], device=device, dtype=torch.float32)

    # 4. Load Models
    # input_dim = 770 
    def build_regressor(input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1)
        ).to(device)

    model_tig = build_regressor(input_dim=770)
    # model_tig.load_state_dict(torch.load("checkpoint/optimal_model/Tig_best_model.ckpt", map_location=device))
    model_tig.load_state_dict(torch.load("checkpoint/predictor_model/tig_model_params821761_mean.ckpt", map_location=device))
    
    model_qpua_pk = build_regressor(input_dim=770)
    # model_qpua_pk.load_state_dict(torch.load("checkpoint/optimal_model/pHRR_best_model.ckpt", map_location=device))
    model_qpua_pk.load_state_dict(torch.load("checkpoint/predictor_model/pkhrr_model_params821761_mean.ckpt", map_location=device))
    
    model_sea = build_regressor(input_dim=770)
    # model_sea.load_state_dict(torch.load("checkpoint/optimal_model/sea_best_model.ckpt", map_location=device)) 
    model_sea.load_state_dict(torch.load("checkpoint/predictor_model/Ysmk_model_params821761_mean.ckpt", map_location=device))
    
    model_co = build_regressor(input_dim=768)
    # model_co.load_state_dict(torch.load("checkpoint/optimal_model/co_best_model.ckpt", map_location=device))
    model_co.load_state_dict(torch.load("checkpoint/predictor_model/Yco_model_params820737_mean.ckpt", map_location=device))

    ae_ckpt = torch.load("checkpoint/decoder_predictor_autoencoder/model_params974699520_lamb1_best_val_current.ckpt", map_location=device)
    decoder = Decoder2(feature_size=202*768, mid_size=768*4, mid_size_2=768*2, latent_size=768).to(device)
    lang_model = prediction_Model(n_embd=768, mid_size=768*4, n_vocab=2393).to(device)
    decoder.load_state_dict(ae_ckpt["decoder"])
    lang_model.load_state_dict(ae_ckpt["predictor"])

    Smi_ted, MolTranBertTokenizer = load_smi_ted_explicit()
    tokenizer = MolTranBertTokenizer("smi_ted_light/bert_vocab_curated.txt")
    Smi_ted_token = Smi_ted(tokenizer) 
    Smi_ted_token.load_checkpoint("smi_ted_light/smi-ted-Light_40.pt")

    A_encoder = star_encoder(dim=1).to(device)
    A_encoder.eval()

    autoencoder = AutoEncoderLayer3(feature_size=202*768, mid_size=768*4, mid_size_2=768*2, latent_size=768).to(device)
    autoencoder.encoder.load_state_dict(ae_ckpt["autoencoder_encoder"])
    autoencoder.encoder.eval()

    for m in [model_tig, model_qpua_pk, model_sea, model_co, decoder, lang_model, Smi_ted_token]:
        m.eval()

    # 5. Initialization
    smiles_idx = config.smiles_idx
    df_init = pd.read_csv("example_smiles.txt", delimiter=",")
    smiles_init = df_init['smiles'][smiles_idx-1]

    with torch.no_grad():
        idx_t, token_emb_t, _ = Smi_ted_token.extract_embeddings(smiles_init)
        idx_t, token_emb_t = idx_t.to(device), token_emb_t.to(device)
        mask_439 = (idx_t == 439)
        pos = torch.nonzero(mask_439, as_tuple=False)
        token_emb_t[pos[:, 0], pos[:, 1], :] = A_encoder(idx_t[pos[:, 0], pos[:, 1]].view([-1, 1]).float())
        embeddings = autoencoder.encoder(token_emb_t.view(-1, 202*768))

    with open("bert_vocab_curated.txt", "r") as f:
        vocab = [line.strip() for line in f]; id_to_token = {i: t for i, t in enumerate(vocab)}

    save_embeddings = [embeddings.clone().detach()]
    save_smiles = [(-1, smiles_init)]
    save_fire_props = []
    save_canonicalized_smiles = []   # <-- defined here

    # 6. Optimization Loop
    for ep in range(config.max_epochs):
        emb_norm = (embeddings - z_tilde_mean[:, :768]) / z_tilde_std[:, :768]
        emb_norm = emb_norm.clone().detach().requires_grad_(True)
        
        flux_norm = (flux_tensor - z_tilde_mean[:, 769]) / z_tilde_std[:, 769]
        thk_norm = (thk_tensor - z_tilde_mean[:, 768]) / z_tilde_std[:, 768]
        
        z_tilde = torch.cat([emb_norm, thk_norm, flux_norm], dim=-1)
        
        # 1. Predictions
        tig_raw = model_tig(z_tilde)
        pkhrr_raw = model_qpua_pk(z_tilde)
        sea_raw = model_sea(z_tilde)
        co_raw = model_co(emb_norm)

        # 2. Limit the ranges using clamp (Standardized Space)
        tig_pred = torch.clamp(tig_raw, min=tig_range[0], max=tig_range[1])
        pkhrr_pred = torch.clamp(pkhrr_raw, min=qpua_pk_range[0], max=qpua_pk_range[1])
        sea_pred = torch.clamp(sea_raw, min=sea_range[0], max=sea_range[1])
        co_pred = torch.clamp(co_raw, min=co_range[0], max=co_range[1])

        # 3. Loss Calculation (raw-space squared, rescaled by per-property y_max^2 so all four are dimensionless ~[0,1])
        tig_raw_val = tig_pred * tig_std + tig_mean
        pkhrr_raw_val = pkhrr_pred * qpua_pk_std + qpua_pk_mean
        sea_raw_val = sea_pred * sea_std + sea_mean
        co_raw_val = co_pred * co_std + co_mean

        loss_tig = config.c_tig * torch.pow(F.relu(900.0 - tig_raw_val), 2) / (900.0 ** 2)
        loss_pkhrr = config.c_qpua * torch.pow(pkhrr_raw_val, 2) / (2748.0 ** 2)
        loss_sea = config.c_sea * torch.pow(sea_raw_val, 2) / (0.149 ** 2)
        loss_co = config.c_co * torch.pow(co_raw_val, 2) / (0.45 ** 2)

        total_loss = loss_tig + loss_pkhrr + loss_sea + loss_co

        # Log denormalized properties
        save_fire_props.append([
            (tig_pred * tig_std + tig_mean).item(),
            (pkhrr_pred * qpua_pk_std + qpua_pk_mean).item(),
            (sea_pred * sea_std + sea_mean).item(),
            (co_pred * co_std + co_mean).item()
        ])

        total_loss.backward()

        with torch.no_grad():
            emb_norm = emb_norm - config.lr_start * emb_norm.grad
            embeddings = emb_norm * z_tilde_std[:, :768] + z_tilde_mean[:, :768]

            # Reconstruction Logic
            token_emb_rec = decoder(embeddings.clone()).view(-1, 202, 768)
            logits = lang_model(token_emb_rec.view(-1, 768))
            pred_idx = F.log_softmax(logits, dim=1).view(-1, 202, 2393).argmax(-1)
            
            tokens = [id_to_token.get(int(idx), '<unk>') for idx in pred_idx.flatten().cpu().numpy()]
            smiles_tokens = [t for t in tokens if t not in ['<pad>', '<unk>', '<s>', '</s>', '<bos>', '<eos>']]
            smiles_string = ''.join(smiles_tokens)
            
            if not save_smiles or smiles_string != save_smiles[-1][-1]:
                canon_smiles = canonicalize_smiles(smiles_string)
                save_smiles.append((ep, smiles_string))
                print(canon_smiles)
                if canon_smiles != "Canonicalized smiles can not be produced":
                    save_canonicalized_smiles.append((ep,canon_smiles))
        
        save_embeddings.append(embeddings.clone().detach())

    print(">> Optimization Finished.")

    # 7. Final Saving
    print(f">> Begin saving the generated psmiles and properties.")
    save_file = f'tig{config.c_tig}_phrr{config.c_qpua}_sea{config.c_sea}_co{config.c_co}_flux{config.flux}_thickness{config.thickness}'
    base_paths = [
    "optimized_embeddings",
    "optimized_smiles",
    "optimized_smiles_canonicalized",
    "log_optimized_prop"
    ]
    [os.makedirs(f"results_new2/{subdir}/{save_file}", exist_ok=True) for subdir in base_paths]


    with open(f"results_new2/optimized_embeddings/{save_file}/optimized_embeddings_{smiles_idx}_lr_{config.lr_start:.4f}.txt", "w") as f:
        for embd in save_embeddings:
            np.savetxt(f, embd.cpu().numpy(), fmt="%.6f")

    with open(f"results_new2/optimized_smiles/{save_file}/optimized_smiles_{smiles_idx}_lr_{config.lr_start:.4f}.txt", "w") as f:
        for ep, smiles in save_smiles:
            f.write(f"Epoch {ep}: {smiles}\n")

    with open(f"results_new2/optimized_smiles_canonicalized/{save_file}/optimized_smiles_canonicalized_{smiles_idx}_lr_{config.lr_start:.4f}.txt", "w") as f:
        for ep, smiles in save_canonicalized_smiles:
            f.write(f"Epoch {ep}: {smiles}\n")

    with open(f"results_new2/log_optimized_prop/{save_file}/log_optimized_props_{smiles_idx}_lr_{config.lr_start:.4f}.txt", "w") as f:
        f.write('tig, qpua_pk, sea, co\n')
        writer  = csv.writer(f)
        for row in save_fire_props:
            writer.writerow([f'{x:.6f}' for x in row])
    
    

if __name__ == '__main__':
    parser = args.get_parser()
    config = parser.parse_args()
    main(config)
