# Deep learning
import torch
import torch.nn as nn

# Data
import pandas as pd

# Standard library
import args_nl as args
import os
from models import star_encoder, AutoEncoderLayer3
from utils import load_smi_ted_explicit
from Canonicalize_for_Optimize import canonicalize_smiles


def build_regressor(input_dim, device):
    return nn.Sequential(
        nn.Linear(input_dim, 512), nn.GELU(),
        nn.Linear(512, 512), nn.GELU(),
        nn.Linear(512, 256), nn.GELU(),
        nn.Linear(256, 128), nn.GELU(),
        nn.Linear(128, 1)
    ).to(device)


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'>> Device being used: {device}')

    sc = torch.load('checkpoint/optimal_model/scalers.pt', map_location=device, weights_only=False)
    z_tilde_mean = sc['X_mean'].to(device)
    z_tilde_std = sc['X_std'].to(device)
    y_mean = sc['Y_mean'].to(device)
    y_std = sc['Y_std'].to(device)

    tig_mean, tig_std = y_mean[0, 0], y_std[0, 0]
    qpua_pk_mean, qpua_pk_std = y_mean[0, 1], y_std[0, 1]
    sea_mean, sea_std = y_mean[0, 2], y_std[0, 2]
    co_mean, co_std = y_mean[0, 3], y_std[0, 3]

    flux_tensor = torch.tensor([[config.flux]], device=device, dtype=torch.float32)
    thk_tensor = torch.tensor([[config.thickness]], device=device, dtype=torch.float32)

    print(f">> Loading property predictors.")
    model_tig = build_regressor(input_dim=770, device=device)
    model_tig.load_state_dict(torch.load("checkpoint/predictor_model/tig_model_params821761_mean.ckpt", map_location=device))

    model_qpua_pk = build_regressor(input_dim=770, device=device)
    model_qpua_pk.load_state_dict(torch.load("checkpoint/predictor_model/pkhrr_model_params821761_mean.ckpt", map_location=device))

    model_sea = build_regressor(input_dim=770, device=device)
    model_sea.load_state_dict(torch.load("checkpoint/predictor_model/Ysmk_model_params821761_mean.ckpt", map_location=device))

    model_co = build_regressor(input_dim=768, device=device)
    model_co.load_state_dict(torch.load("checkpoint/predictor_model/Yco_model_params820737_mean.ckpt", map_location=device))

    print(f">> Loading encoder (smi-ted + star_encoder + autoencoder).")
    ae_ckpt = torch.load("checkpoint/decoder_predictor_autoencoder/model_params974699520_lamb1_best_val_current.ckpt", map_location=device)

    Smi_ted, MolTranBertTokenizer = load_smi_ted_explicit()
    tokenizer = MolTranBertTokenizer("smi_ted_light/bert_vocab_curated.txt")
    Smi_ted_token = Smi_ted(tokenizer)
    Smi_ted_token.load_checkpoint("smi_ted_light/smi-ted-Light_40.pt")

    A_encoder = star_encoder(dim=1).to(device)

    autoencoder = AutoEncoderLayer3(feature_size=202*768, mid_size=768*4, mid_size_2=768*2, latent_size=768).to(device)
    autoencoder.encoder.load_state_dict(ae_ckpt["autoencoder_encoder"])

    for m in [model_tig, model_qpua_pk, model_sea, model_co, A_encoder, autoencoder.encoder, Smi_ted_token]:
        m.eval()

    smiles_idx = config.smiles_idx
    df_init = pd.read_csv("example_smiles.txt", delimiter=",")
    smile = df_init['smiles'][smiles_idx-1]
    print(f">> Predicting properties for pSMILES idx={smiles_idx}: {smile}")

    with torch.no_grad():
        idx_t, token_emb_t, _ = Smi_ted_token.extract_embeddings(smile)
        idx_t, token_emb_t = idx_t.to(device), token_emb_t.to(device)
        mask_439 = (idx_t == 439)
        pos = torch.nonzero(mask_439, as_tuple=False)
        token_emb_t[pos[:, 0], pos[:, 1], :] = A_encoder(idx_t[pos[:, 0], pos[:, 1]].view([-1, 1]).float())
        embeddings = autoencoder.encoder(token_emb_t.view(-1, 202*768))

        emb_norm = (embeddings - z_tilde_mean[:, :768]) / z_tilde_std[:, :768]
        flux_norm = (flux_tensor - z_tilde_mean[:, 769]) / z_tilde_std[:, 769]
        thk_norm = (thk_tensor - z_tilde_mean[:, 768]) / z_tilde_std[:, 768]
        z_tilde = torch.cat([emb_norm, thk_norm, flux_norm], dim=-1)

        tig_pred = (model_tig(z_tilde) * tig_std + tig_mean).item()
        pkhrr_pred = (model_qpua_pk(z_tilde) * qpua_pk_std + qpua_pk_mean).item()
        sea_pred = (model_sea(z_tilde) * sea_std + sea_mean).item()
        co_pred = (model_co(emb_norm) * co_std + co_mean).item()

    canon = canonicalize_smiles(smile)
    print(f">> Canonicalized pSMILES: {canon}")
    print(f">> Conditions: flux={config.flux}, thickness={config.thickness}")
    print(f">> Predicted Tig:  {tig_pred:.4f}")
    print(f">> Predicted pHRR: {pkhrr_pred:.4f}")
    print(f">> Predicted SEA:  {sea_pred:.6f}")
    print(f">> Predicted CO:   {co_pred:.6f}")


if __name__ == '__main__':
    parser = args.get_parser()
    config = parser.parse_args()
    main(config)
