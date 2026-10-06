import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class PolymerBinaryClassifier(nn.Module):
    def __init__(self, input_dim=770, hidden_dim=256):
        super(PolymerBinaryClassifier, self).__init__()
        # Input is 768 (embedding) + 2 (thickness, flux)
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) # Output logit for Binary Cross Entropy
        )

    def forward(self, x):
        # Ensure thickness and flux are [Batch, 1]
        return self.net(x)


class LogisticRegression1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # input dim = 1, output dim = 1

    def forward(self, x):
        # x: (batch_size, 1)
        # return logits; apply sigmoid only if you need probabilities
        return self.linear(x)


class LogisticRegression(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # input dim = 1, output dim = 1

    def forward(self, x):
        # x: (batch_size, 1)
        # return logits; apply sigmoid only if you need probabilities
        return self.linear(x)


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


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 output_dim,
                 activation='relu',
                 dropout_rate=0.1,
                 use_batch_norm=False,
                 use_layer_norm=False,
                 bias=True,
                 final_activation=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm

        # Validate inputs
        if not isinstance(hidden_dims, (list, tuple)):
            raise ValueError("hidden_dims must be a list or tuple")
        if dropout_rate < 0 or dropout_rate > 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        if use_batch_norm and use_layer_norm:
            raise ValueError("use_batch_norm and use_layer_norm cannot be True at the same time")

        # Build layers
        self.layers = nn.ModuleList()

        # Input layer
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, hidden_dim, bias=bias))
            current_dim = hidden_dim

            # Add normalization
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            if use_layer_norm:
                self.layers.append(nn.LayerNorm(hidden_dim))

            # Add activation
            if activation == 'relu':
                self.layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                self.layers.append(nn.LeakyReLU())
            elif activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif activation == 'gelu':
                self.layers.append(nn.GELU())
            elif activation == 'sigmoid':
                self.layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Invalid activation function: {activation}")

            # Add dropout
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))

        # Output layer
        self.output_layer = nn.Linear(current_dim, output_dim, bias=bias)

        # Final activation
        self.final_activation = final_activation if final_activation is not None else nn.Identity()

    def forward(self, x):
        # Apply hidden layers
        for layer in self.layers:
            x = layer(x)

        # Apply output layer
        x = self.output_layer(x)

        # Apply final activation
        x = self.final_activation(x)
        return x


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


class TopKSAE(nn.Module):
    """
    Top-K Sparse Autoencoder for pSMILES embeddings (Eq. 1 of Section 2.1):

        c = TopK(W_enc z + b_enc)           c in R^D, exactly K nonzeros
        z_hat = W_dec c + b_dec             z_hat in R^d

    D = expansion_factor * d. Decoder columns are kept unit-norm.
    """

    def __init__(self, d=768, expansion_factor=8, k=80):
        super().__init__()
        assert isinstance(k, int) and k > 0
        D = expansion_factor * d
        self.d = d
        self.D = D
        self.expansion_factor = expansion_factor
        self.register_buffer("k", torch.tensor(k, dtype=torch.int))

        self.encoder = nn.Linear(d, D, bias=True)
        self.decoder = nn.Linear(D, d, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d))

        with torch.no_grad():
            W = torch.randn(d, D)
            W = W / W.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.decoder.weight.copy_(W)             # [d, D]
            self.encoder.weight.copy_(W.T.clone())   # [D, d]
            self.encoder.bias.zero_()

    def encode(self, z, return_topk=False):
        """z: [B, d] -> c: [B, D] with exactly k nonzeros per row."""
        pre = F.relu(self.encoder(z))
        k = int(self.k.item())
        topk = pre.topk(k, sorted=False, dim=-1)
        c = torch.zeros_like(pre).scatter_(-1, topk.indices, topk.values)
        if return_topk:
            return c, topk.values, topk.indices
        return c

    def decode(self, c):
        return self.decoder(c) + self.b_dec

    def forward(self, z, output_features=False):
        c = self.encode(z)
        z_hat = self.decode(c)
        if output_features:
            return z_hat, c
        return z_hat

    @torch.no_grad()
    def normalize_decoder_(self):
        """Project decoder columns back to unit norm (call after optimizer step)."""
        w = self.decoder.weight.data
        w.div_(w.norm(dim=0, keepdim=True).clamp(min=1e-8))


############ save here just in case old poly4mer model needs these models ############

class Star_encoder_old(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
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
