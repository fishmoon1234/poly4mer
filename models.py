import torch 
import torch.nn.functional as F
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class Decoder(nn.Module):
    def __init__(self, feature_size, mid_size, latent_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, mid_size)
        self.ln_f = nn.LayerNorm(mid_size)
        self.rec = nn.Linear(mid_size, feature_size, bias=False)
    
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.ln_f(x)
        x = self.rec(x)
        return x # -> (N, L*D)
    

class Decoder2(nn.Module):
    def __init__(self, feature_size, mid_size, mid_size_2, latent_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, mid_size_2)
        self.ln_f = nn.LayerNorm(mid_size_2)
        self.fc2 = nn.Linear(mid_size_2, mid_size)
        self.ln_f_2 = nn.LayerNorm(mid_size)
        self.lat = nn.Linear(mid_size, feature_size, bias=False)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.ln_f(x)
        x = F.gelu(self.fc2(x))
        x = self.ln_f_2(x)
        x = self.lat(x)
        return x  
        

class AutoEncoderLayer2(nn.Module):
    def __init__(self, feature_size, mid_size, latent_size):
        super().__init__()
        self.encoder = self.Encoder(feature_size, mid_size, latent_size)
        self.decoder = self.Decoder(feature_size, mid_size, latent_size)
    
    class Encoder(nn.Module):

        def __init__(self, feature_size, mid_size, latent_size):
            super().__init__()
            self.is_cuda_available = torch.cuda.is_available()
            self.fc1 = nn.Linear(feature_size, mid_size)
            self.ln_f = nn.LayerNorm(mid_size)
            self.lat = nn.Linear(mid_size, latent_size, bias=False)
        
        def forward(self, x):
            x = F.gelu(self.fc1(x))
            x = self.ln_f(x)
            x = self.lat(x)
            return x # -> (N, D)
        
    class Decoder(nn.Module):

        def __init__(self, feature_size, mid_size, latent_size):
            super().__init__()
            self.is_cuda_available = torch.cuda.is_available()
            self.fc1 = nn.Linear(latent_size, mid_size)
            self.ln_f = nn.LayerNorm(mid_size)
            self.rec = nn.Linear(mid_size, feature_size, bias=False)
        
        def forward(self, x):
            x = F.gelu(self.fc1(x))
            x = self.ln_f(x)
            x = self.rec(x)
            return x # -> (N, L*D)
        

class AutoEncoderLayer3(nn.Module):
    def __init__(self, feature_size, mid_size,mid_size_2, latent_size):
        super().__init__()
        self.encoder = self.Encoder(feature_size, mid_size, mid_size_2 ,latent_size)
        self.decoder = self.Decoder(feature_size, mid_size, mid_size_2 ,latent_size)

    class Encoder(nn.Module):

        def __init__(self, feature_size, mid_size, mid_size_2,latent_size):
            super().__init__()
            self.is_cuda_available = torch.cuda.is_available()
            self.fc1 = nn.Linear(feature_size, mid_size)
            self.ln_f = nn.LayerNorm(mid_size)
            self.fc2 = nn.Linear(mid_size, mid_size_2)
            self.ln_f_2 = nn.LayerNorm(mid_size_2)
            self.lat = nn.Linear(mid_size_2, latent_size, bias=False)

        def forward(self, x):
            x = F.gelu(self.fc1(x))
            x = self.ln_f(x)
            x = F.gelu(self.fc2(x))
            x = self.ln_f_2(x)
            x = self.lat(x)
            return x  # -> (N, D)

    class Decoder(nn.Module):

        def __init__(self, feature_size, mid_size,mid_size_2,latent_size):
            super().__init__()
            self.is_cuda_available = torch.cuda.is_available()
            self.fc1 = nn.Linear(latent_size, mid_size_2)
            self.ln_f = nn.LayerNorm(mid_size_2)
            self.fc2 = nn.Linear(mid_size_2, mid_size)
            self.ln_f_2 = nn.LayerNorm(mid_size)
            self.lat = nn.Linear(mid_size, feature_size, bias=False)

        def forward(self, x):
            x = F.gelu(self.fc1(x))
            x = self.ln_f(x)
            x = F.gelu(self.fc2(x))
            x = self.ln_f_2(x)
            x = self.lat(x)
            return x  # -> (N, D)
        

class Star_encoder(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        return self.mlp(x)


class star_encoder(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 768)
        )

    def forward(self, x):
        return self.mlp(x)



class prediction_Model_1(nn.Module):   # this model use original lang model structure in smi_ted_light.load.langlayer 
    def __init__(self, n_embd = 768,n_vocab = 2393):
        super().__init__()
        self.embed = nn.Linear(n_embd, n_embd)
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, n_vocab, bias=False)

    def forward(self, tensor):
        tensor = self.embed(tensor)
        tensor = F.gelu(tensor)
        tensor = self.ln_f(tensor)
        tensor = self.head(tensor)
        return tensor


class prediction_Model(nn.Module):   
    def __init__(self, n_embd = 768, mid_size=768 * 4, n_vocab = 2393):
        super().__init__()
        #self.is_cuda_available = torch.cuda.is_available()
        self.embed = nn.Linear(n_embd, mid_size)
        self.ln_f = nn.LayerNorm(mid_size)
        self.head = nn.Linear(mid_size, n_vocab, bias=False)

    def forward(self, tensor):
        tensor = self.embed(tensor)
        tensor = F.gelu(tensor)
        tensor = self.ln_f(tensor)
        tensor = self.head(tensor)
        return tensor
    

class prediction_Model_2(nn.Module): 
    def __init__(self, n_embd = 768, mid_size1 = 768 * 4, mid_size2 = 768 * 8, n_vocab = 2393):
        super().__init__()
        #self.is_cuda_available = torch.cuda.is_available()
        self.embed = nn.Linear(n_embd, mid_size1)
        self.ln_f = nn.LayerNorm(mid_size1)
        self.embed_2 = nn.Linear(mid_size1, mid_size2)
        self.ln_f_2 = nn.LayerNorm(mid_size2)
        self.head = nn.Linear(mid_size2, n_vocab, bias=False)

    def forward(self, tensor):
        tensor = self.embed(tensor)
        tensor = F.gelu(tensor)
        tensor = self.ln_f(tensor)
        tensor = self.embed_2(tensor)
        tensor = F.gelu(tensor)
        tensor = self.ln_f_2(tensor)
        tensor = self.head(tensor)
        return tensor
    

class nonlinear_regression_model2(nn.Module):
    def __init__(self, n_embd = 768, mid = 768 * 4, property = 1, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, mid)
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Linear(mid, mid)
        self.dropout2 = nn.Dropout(dropout)
        self.relu2 = nn.GELU()
        self.fc3 = nn.Linear(mid, mid)
        self.dropout3 = nn.Dropout(dropout)
        self.relu3 = nn.GELU()
        self.final = nn.Linear(mid, property)

    def forward(self, smiles_emb):
        x_out = self.fc1(smiles_emb)
        x_out = self.dropout1(x_out)
        x_out = self.relu1(x_out)

        z = self.fc2(x_out)
        z = self.dropout2(z)
        z = self.relu2(z)
        z = self.fc3(z)
        z = self.dropout3(z)
        z = self.relu3(z)
        z = self.final(z)
        # z = F.sigmoid(z)
        return z