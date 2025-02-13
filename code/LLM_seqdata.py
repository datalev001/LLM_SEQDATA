#########################################
#1) the extended MLA/reinf-QL modifications in deepseek manner  using Q-learning
#   using target for reward search in RL
###########################################

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
import matplotlib.pyplot as plt

df_tokens = pd.read_csv('df_tokens.csv')

# -------------------------------
# Vocabulary and Dataset Preparation
# -------------------------------
# Build vocabulary from token sequences.
all_text = "".join(df_tokens["tokens"].values)
vocab = sorted(list(set(all_text)))
vocab_dict = {char: idx + 1 for idx, char in enumerate(vocab)}  # Reserve 0 for PAD.
print("Vocabulary:", vocab_dict)
vocab_size = len(vocab_dict)
max_len = df_tokens["tokens"].apply(len).max()

def tokenize(token_str):
    return [vocab_dict[char] for char in token_str]

class TokenDataset(Dataset):
    def __init__(self, df, max_len):
        self.df = df
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        token_str = self.df.iloc[idx]["tokens"]
        target = self.df.iloc[idx]["target"]
        token_ids = tokenize(token_str)
        pad_length = self.max_len - len(token_ids)
        token_ids = token_ids + [0] * pad_length
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor([target], dtype=torch.float)

dataset = TokenDataset(df_tokens, max_len)
total_samples = len(dataset)
train_size = int(0.8 * total_samples)
val_size = total_samples - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# -------------------------------
# Positional Encoding Module
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# -------------------------------
# Transformer Model with Extended MLA and Q-Learning Style RL
# -------------------------------
class MLA_RL_Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len, lambda_policy=0.1):
        """
        This model embeds token sequences, applies positional encoding, and passes the data
        through a Transformer encoder. It outputs:
         - A forecast (predicted target) via a linear layer.
         - A policy value used for a Q-learning style RL loss.
        """
        super(MLA_RL_Transformer, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_forecast = nn.Linear(embed_dim, 1)
        self.policy_net = nn.Linear(embed_dim, 1)
        self.lambda_policy = lambda_policy
        
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x) * np.sqrt(self.embed_dim)  # (batch, seq_len, embed_dim)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (seq_len, batch, embed_dim)
        encoded = self.transformer_encoder(x)  # (seq_len, batch, embed_dim)
        pooled = encoded.mean(dim=0)  # (batch, embed_dim)
        forecast = self.fc_forecast(pooled)  # (batch, 1)
        policy_value = self.policy_net(pooled)  # (batch, 1)
        return forecast, policy_value

# -------------------------------
#  Helper Functions to Compute Target from Tokens (for Computed MAPE)
# -------------------------------
# Create an inverse vocabulary for decoding tokens back to characters.
inv_vocab = {v: k for k, v in vocab_dict.items()}

def compute_target_from_tokens(token_ids):
    total = 0.0
    count = 0
    for i, token in enumerate(token_ids):
        if token != 0:  # ignore PAD tokens
            char = inv_vocab[token]
            # Use the same lookup as in data generation
            lookup = {'A': 6, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
            total += lookup[char] * (i + 1)
            count += 1
    return total / count if count > 0 else 0.0

def compute_batch_target(x_batch):
    # x_batch: tensor of shape (batch, seq_len)
    targets = []
    for sample in x_batch:
        token_ids = sample.cpu().tolist()
        token_ids = [t for t in token_ids if t != 0]  # remove PAD tokens
        targets.append(compute_target_from_tokens(token_ids))
    return torch.tensor(targets, dtype=torch.float, device=x_batch.device).unsqueeze(1)

# -------------------------------
# Training and Validation Functions
# -------------------------------
def train_model(model, dataloader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        forecast, policy_value = model(x)
        loss_mse = F.mse_loss(forecast, y)
        loss_mae = F.l1_loss(forecast, y)
        loss_forecast = 0.5 * loss_mse + 0.5 * loss_mae
        
        # Q-learning style policy loss:
        advantage = (y - forecast)
        baseline = 0.5
        r_t = policy_value / baseline
        epsilon = 0.1
        r_t_clipped = torch.clamp(r_t, 1 - epsilon, 1 + epsilon)
        policy_loss = -torch.min(r_t * advantage, r_t_clipped * advantage).mean()
        
        loss = loss_forecast + model.lambda_policy * policy_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

def validate_model_provided(model, dataloader, device):
    # Validation using the provided target from the dataset.
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            forecast, _ = model(x)
            all_preds.append(forecast.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mape = np.mean(np.abs((all_targets - all_preds) / (all_targets + 1e-6))) * 100
    return mape

def validate_model_computed(model, dataloader, device):
    # Validation using a computed target from tokens.
    model.eval()
    all_preds = []
    all_computed = []
    with torch.no_grad():
        for x, _ in dataloader:  # Ignore the provided target.
            x = x.to(device)
            forecast, _ = model(x)
            computed_target = compute_batch_target(x)
            all_preds.append(forecast.cpu().numpy())
            all_computed.append(computed_target.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_computed = np.concatenate(all_computed, axis=0)
    mape = np.mean(np.abs((all_computed - all_preds) / (all_computed + 1e-6))) * 100
    return mape

def query_model(model, query_str, max_len):
    token_ids = tokenize(query_str)
    pad_length = max_len - len(token_ids)
    token_ids = token_ids + [0] * pad_length
    x = torch.tensor([token_ids], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        forecast, _ = model(x)
    return forecast.item()

# -------------------------------
# Main Training Loop with Early Stopping
# -------------------------------
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLA_RL_Transformer(vocab_size=vocab_size, embed_dim=16, num_heads=2, num_layers=2, max_len=max_len, lambda_policy=0.1)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 50
    mape_history = []
    
    print("Training MLA-RL Transformer Model...")
    for epoch in range(n_epochs):
        train_loss = train_model(model, train_loader, optimizer, device, grad_clip=1.0)
        provided_mape = validate_model_provided(model, val_loader, device)
        computed_mape = validate_model_computed(model, val_loader, device)
        mape_history.append(provided_mape)
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {train_loss:.4f}, Provided MAPE: {provided_mape:.2f}%, Computed MAPE: {computed_mape:.2f}%")
        
        # Early stopping: if after 10 epochs no improvement of at least 1% relative to the best of the last 10 epochs.
        if epoch >= 10:
            recent_best = min(mape_history[-10:])
            if provided_mape > recent_best * 0.995:
                print("Early stopping triggered: No improvement of at least 1% in the last 10 epochs.")
                break
    
    # Interactive query example:
    sample_query = "ACB"  # Example token sequence
    predicted_target = query_model(model, sample_query, max_len)
    print(f"Query: {sample_query} -> Predicted Target: {predicted_target:.2f}")

if __name__ == "__main__":
    main()





#########################################
# 2) the extended MLA/GRPO modifications in deepseek manner using GRPO
#     using target for reward search in RL
###########################################
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

df_tokens = pd.read_csv('df_tokens.csv')

# -------------------------------
# Vocabulary and Dataset Preparation
# -------------------------------
all_text = "".join(df_tokens["tokens"].values)
vocab = sorted(list(set(all_text)))
vocab_dict = {char: idx + 1 for idx, char in enumerate(vocab)}  # Reserve 0 for PAD.
vocab_size = len(vocab_dict)
max_len = df_tokens["tokens"].apply(len).max()

def tokenize(token_str):
    return [vocab_dict[char] for char in token_str]

class TokenDataset(Dataset):
    def __init__(self, df, max_len):
        self.df = df
        self.max_len = max_len
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        token_str = self.df.iloc[idx]["tokens"]
        target = self.df.iloc[idx]["target"]
        token_ids = tokenize(token_str)
        pad_length = self.max_len - len(token_ids)
        token_ids = token_ids + [0] * pad_length
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor([target], dtype=torch.float)

dataset = TokenDataset(df_tokens, max_len)
total_samples = len(dataset)
train_size = int(0.8 * total_samples)
val_size = total_samples - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# -------------------------------
# Positional Encoding Module
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# -------------------------------
# Transformer Model with GRPO-inspired Loss (MLA-style)
# -------------------------------
class GRPOTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len, lambda_policy=0.1):
        super(GRPOTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_forecast = nn.Linear(embed_dim, 1)
        self.policy_net = nn.Linear(embed_dim, 1)
        self.lambda_policy = lambda_policy
    
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x) * np.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x).transpose(0, 1)  # (seq_len, batch, embed_dim)
        encoded = self.transformer_encoder(x).mean(dim=0)  # (batch, embed_dim)
        forecast = self.fc_forecast(encoded)  # (batch, 1)
        policy_value = self.policy_net(encoded)  # (batch, 1)
        return forecast, policy_value

# -------------------------------
#  Helper Functions to Compute Target from Tokens (for "Computed MAPE")
# -------------------------------
# Define a global lookup table (same as in data generation)
lookup = {'A': 6, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
# Create inverse vocabulary for decoding tokens back to characters
inv_vocab = {v: k for k, v in vocab_dict.items()}

def compute_target_from_tokens(token_ids):
    total = 0.0
    count = 0
    for i, token in enumerate(token_ids):
        if token != 0:  # ignore PAD tokens
            char = inv_vocab[token]
            total += lookup[char] * (i + 1)
            count += 1
    return total / count if count > 0 else 0.0

def compute_batch_target(x_batch):
    # x_batch: tensor of shape (batch, seq_len)
    targets = []
    for sample in x_batch:
        token_ids = sample.cpu().tolist()
        token_ids = [t for t in token_ids if t != 0]  # remove PAD tokens
        targets.append(compute_target_from_tokens(token_ids))
    return torch.tensor(targets, dtype=torch.float, device=x_batch.device).unsqueeze(1)

# -------------------------------
# Training and Validation Functions with Early Stopping
# -------------------------------
def train_model(model, dataloader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        forecast, policy_value = model(x)
        loss_mse = F.mse_loss(forecast, y)
        loss_mae = F.l1_loss(forecast, y)
        loss_forecast = 0.5 * loss_mse + 0.5 * loss_mae
        
        # Group-based GRPO policy loss:
        group_advantage = y - forecast
        baseline = group_advantage.mean(dim=0, keepdim=True)
        group_relative_advantage = group_advantage - baseline
        
        r_t = policy_value / baseline
        r_t_clipped = torch.clamp(r_t, 0.9, 1.1)
        policy_loss = -torch.min(r_t * group_relative_advantage, r_t_clipped * group_relative_advantage).mean()
        
        loss = loss_forecast + model.lambda_policy * policy_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

def validate_model_provided(model, dataloader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            forecast, _ = model(x)
            all_preds.append(forecast.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    mape = np.mean(np.abs((all_targets - all_preds) / (all_targets + 1e-6))) * 100
    return mape

def validate_model_computed(model, dataloader, device):
    model.eval()
    all_preds, all_computed = [], []
    with torch.no_grad():
        for x, _ in dataloader:  # Ignore the provided target here.
            x = x.to(device)
            forecast, _ = model(x)
            computed_target = compute_batch_target(x)
            all_preds.append(forecast.cpu().numpy())
            all_computed.append(computed_target.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_computed = np.concatenate(all_computed)
    mape = np.mean(np.abs((all_computed - all_preds) / (all_computed + 1e-6))) * 100
    return mape

def query_model(model, query_str, max_len):
    token_ids = tokenize(query_str)
    pad_length = max_len - len(token_ids)
    token_ids = token_ids + [0] * pad_length
    x = torch.tensor([token_ids], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        forecast, _ = model(x)
    return forecast.item()

# -------------------------------
#  Main Training Loop with Early Stopping
# -------------------------------
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRPOTransformer(vocab_size=vocab_size, embed_dim=16, num_heads=2, num_layers=2, max_len=max_len, lambda_policy=0.1)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 50
    mape_history = []
    best_mape = float("inf")
    epochs_since_improvement = 0
    early_stop_patience = 10  # Stop if no 1% improvement in the last 10 epochs
    
    print("Training GRPO Transformer Model...")
    for epoch in range(n_epochs):
        train_loss = train_model(model, train_loader, optimizer, device, grad_clip=1.0)
        provided_mape = validate_model_provided(model, val_loader, device)
        computed_mape = validate_model_computed(model, val_loader, device)
        mape_history.append(provided_mape)
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {train_loss:.4f}, Provided MAPE: {provided_mape:.2f}%, Computed MAPE: {computed_mape:.2f}%")
        
        if provided_mape < best_mape * 0.995:
            best_mape = provided_mape
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        
        if epoch >= 10 and epochs_since_improvement >= early_stop_patience:
            print("Early stopping triggered: No improvement of at least 1% in the last 10 epochs.")
            break
    
    sample_query = "ACB"  # Example token sequence
    predicted_target = query_model(model, sample_query, max_len)
    print(f"Query: {sample_query} -> Predicted Target: {predicted_target:.2f}")

if __name__ == "__main__":
    main()






#########################################
# 3) traditional Transformer model using multi‐head attention 
#    (without the extended MLA/GRPO modifications)
###########################################
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
import matplotlib.pyplot as plt

df_tokens = pd.read_csv('df_tokens.csv')

# -------------------------------
#  Vocabulary and Dataset Preparation
# -------------------------------
# Build vocabulary from token sequences.
all_text = "".join(df_tokens["tokens"].values)
vocab = sorted(list(set(all_text)))
vocab_dict = {char: idx + 1 for idx, char in enumerate(vocab)}  # Reserve 0 for PAD.
print("Vocabulary:", vocab_dict)
vocab_size = len(vocab_dict)
max_len = df_tokens["tokens"].apply(len).max()

# Create inverse vocabulary for decoding
inv_vocab = {v: k for k, v in vocab_dict.items()}

def tokenize(token_str):
    return [vocab_dict[char] for char in token_str]

class TokenDataset(Dataset):
    def __init__(self, df, max_len):
        self.df = df
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        token_str = self.df.iloc[idx]["tokens"]
        target = self.df.iloc[idx]["target"]
        token_ids = tokenize(token_str)
        pad_length = self.max_len - len(token_ids)
        token_ids = token_ids + [0] * pad_length
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor([target], dtype=torch.float)

dataset = TokenDataset(df_tokens, max_len)
total_samples = len(dataset)
train_size = int(0.8 * total_samples)
val_size = total_samples - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# -------------------------------
#  Positional Encoding Module
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# -------------------------------
# Traditional Transformer Model (Multi-Head Attention)
# -------------------------------
class TraditionalTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len):
        """
        This model uses a traditional transformer encoder to predict a target value from a token sequence.
        It consists of:
         - an embedding layer,
         - positional encoding,
         - a transformer encoder with multi-head attention,
         - mean pooling, and
         - a linear layer to output the forecast.
        """
        super(TraditionalTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_forecast = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x) * np.sqrt(self.embed_dim)  # (batch, seq_len, embed_dim)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (seq_len, batch, embed_dim)
        encoded = self.transformer_encoder(x)  # (seq_len, batch, embed_dim)
        pooled = encoded.mean(dim=0)  # (batch, embed_dim)
        forecast = self.fc_forecast(pooled)  # (batch, 1)
        return forecast

# -------------------------------
#  Helper Functions to Compute Target from Tokens
# -------------------------------
# Global lookup dictionary (same as used in data generation)
lookup = {'A': 6, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

def compute_target_from_tokens(token_ids):
    total = 0.0
    count = 0
    for i, token in enumerate(token_ids):
        if token != 0:  # ignore PAD tokens
            char = inv_vocab[token]
            total += lookup[char] * (i + 1)
            count += 1
    return total / count if count > 0 else 0.0

def compute_batch_target(x_batch):
    # x_batch: tensor of shape (batch, seq_len)
    targets = []
    for sample in x_batch:
        token_ids = sample.cpu().tolist()
        token_ids = [t for t in token_ids if t != 0]  # remove PAD tokens
        targets.append(compute_target_from_tokens(token_ids))
    return torch.tensor(targets, dtype=torch.float, device=x_batch.device).unsqueeze(1)

# -------------------------------
# Training and Validation Functions with Early Stopping
# -------------------------------
def train_model(model, dataloader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        forecast = model(x)
        loss_mse = F.mse_loss(forecast, y)
        loss_mae = F.l1_loss(forecast, y)
        loss = 0.5 * loss_mse + 0.5 * loss_mae
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

def validate_model_provided(model, dataloader, device):
    """
    Validation using the provided target from the dataset.
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            forecast = model(x)
            all_preds.append(forecast.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mape = np.mean(np.abs((all_targets - all_preds) / (all_targets + 1e-6))) * 100
    return mape

def validate_model_computed(model, dataloader, device):
    """
    Validation using a computed target from tokens.
    """
    model.eval()
    all_preds = []
    all_computed = []
    with torch.no_grad():
        for x, _ in dataloader:  # ignore the provided target here
            x = x.to(device)
            forecast = model(x)
            computed_target = compute_batch_target(x)
            all_preds.append(forecast.cpu().numpy())
            all_computed.append(computed_target.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_computed = np.concatenate(all_computed, axis=0)
    mape = np.mean(np.abs((all_computed - all_preds) / (all_computed + 1e-6))) * 100
    return mape

def query_model(model, query_str, max_len):
    token_ids = tokenize(query_str)
    pad_length = max_len - len(token_ids)
    token_ids = token_ids + [0] * pad_length
    x = torch.tensor([token_ids], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        forecast = model(x)
    return forecast.item()

# -------------------------------
# Main Training Loop with Early Stopping
# -------------------------------
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraditionalTransformer(vocab_size=vocab_size, embed_dim=16, num_heads=2, num_layers=2, max_len=max_len)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 50
    mape_history = []
    
    print("Training Traditional Transformer Model...")
    for epoch in range(n_epochs):
        train_loss = train_model(model, train_loader, optimizer, device, grad_clip=1.0)
        mape_provided = validate_model_provided(model, val_loader, device)
        mape_computed = validate_model_computed(model, val_loader, device)
        mape_history.append(mape_provided)
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {train_loss:.4f}, Provided MAPE: {mape_provided:.2f}%, Computed MAPE: {mape_computed:.2f}%")
        
        # Early stopping: after 10 epochs, if no improvement of at least 1% relative to the best MAPE of the last 10 epochs, stop.
        if epoch >= 10:
            recent_best = min(mape_history[-10:])
            if mape_provided > recent_best * 0.99:
                print("Early stopping triggered: No improvement of at least 1% in the last 10 epochs.")
                break
    
    sample_query = "ACB"  # Example token sequence
    predicted_target = query_model(model, sample_query, max_len)
    print(f"Query: {sample_query} -> Predicted Target: {predicted_target:.2f}")

if __name__ == "__main__":
    main()




########################################################
4)  A) using MLA and GRPO strategy in deepseek manner 
    B) not using the column target in GRPO stage and GRPO learn reward itself
    C) using 'target' only for validation purposes 
    D) using an early stop approach
#######################################################
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

df_tokens = pd.read_csv('df_tokens.csv')

# -------------------------------
# Vocabulary and Dataset Preparation
# -------------------------------
all_text = "".join(df_tokens["tokens"].values)
vocab = sorted(list(set(all_text)))
vocab_dict = {char: idx + 1 for idx, char in enumerate(vocab)}  # Reserve 0 for PAD.
# Create inverse vocabulary for decoding
inv_vocab = {v: k for k, v in vocab_dict.items()}
vocab_size = len(vocab_dict)
max_len = df_tokens["tokens"].apply(len).max()

def tokenize(token_str):
    return [vocab_dict[char] for char in token_str]

class TokenDataset(Dataset):
    def __init__(self, df, max_len):
        self.df = df
        self.max_len = max_len
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        token_str = self.df.iloc[idx]["tokens"]
        target = self.df.iloc[idx]["target"]  # Provided target (for validation only)
        token_ids = tokenize(token_str)
        pad_length = self.max_len - len(token_ids)
        token_ids = token_ids + [0] * pad_length
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor([target], dtype=torch.float)

dataset = TokenDataset(df_tokens, max_len)
total_samples = len(dataset)
train_size = int(0.8 * total_samples)
val_size = total_samples - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# -------------------------------
# Positional Encoding Module
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# -------------------------------
# Transformer Model with Extended MLA and GRPO-inspired Loss (Training Using Only Tokens)
# -------------------------------
class GRPOTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len, lambda_policy=0.1):
        super(GRPOTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_forecast = nn.Linear(embed_dim, 1)
        self.policy_net = nn.Linear(embed_dim, 1)
        self.lambda_policy = lambda_policy
        
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x) * np.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x).transpose(0, 1)
        encoded = self.transformer_encoder(x).mean(dim=0)
        forecast = self.fc_forecast(encoded)
        policy_value = self.policy_net(encoded)
        return forecast, policy_value

# -------------------------------
# Helper Functions to Compute Target from Tokens
# -------------------------------
# Global lookup dictionary (same as used in data generation)
lookup = {'A': 6, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

def compute_target_from_tokens(token_ids):
    total = 0.0
    count = 0
    for i, token in enumerate(token_ids):
        if token != 0:  # ignore PAD
            char = inv_vocab[token]
            total += lookup[char] * (i + 1)
            count += 1
    return total / count if count > 0 else 0.0

def compute_batch_target(x_batch):
    # x_batch: tensor of shape (batch, seq_len)
    targets = []
    for sample in x_batch:
        token_ids = sample.cpu().tolist()
        # Remove PAD tokens (value 0)
        token_ids = [t for t in token_ids if t != 0]
        targets.append(compute_target_from_tokens(token_ids))
    return torch.tensor(targets, dtype=torch.float, device=x_batch.device).unsqueeze(1)

# -------------------------------
# Training and Validation Functions with Early Stopping
# -------------------------------
def train_model(model, dataloader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for x, _ in dataloader:  # Ignore provided target during training
        x = x.to(device)
        computed_target = compute_batch_target(x)
        optimizer.zero_grad()
        forecast, policy_value = model(x)
        
        loss_mse = F.mse_loss(forecast, computed_target)
        loss_mae = F.l1_loss(forecast, computed_target)
        loss_forecast = 0.5 * loss_mse + 0.5 * loss_mae
        
        group_advantage = computed_target - forecast
        baseline = group_advantage.mean(dim=0, keepdim=True)
        group_relative_advantage = group_advantage - baseline
        
        r_t = policy_value / baseline
        r_t_clipped = torch.clamp(r_t, 0.9, 1.1)
        policy_loss = -torch.min(r_t * group_relative_advantage, r_t_clipped * group_relative_advantage).mean()
        
        loss = loss_forecast + model.lambda_policy * policy_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

def validate_model_using_provided(model, dataloader, device):
    # Validation using the provided target from the dataset
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            forecast, _ = model(x)
            all_preds.append(forecast.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    mape = np.mean(np.abs((all_targets - all_preds) / (all_targets + 1e-6))) * 100
    return mape

def validate_model_using_computed(model, dataloader, device):
    # Validation using computed target from tokens
    model.eval()
    all_preds, all_computed = [], []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            computed_target = compute_batch_target(x)
            forecast, _ = model(x)
            all_preds.append(forecast.cpu().numpy())
            all_computed.append(computed_target.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_computed = np.concatenate(all_computed)
    mape = np.mean(np.abs((all_computed - all_preds) / (all_computed + 1e-6))) * 100
    return mape

# -------------------------------
# Interactive Query Function
# -------------------------------
def query_model(model, query_str, max_len):
    token_ids = tokenize(query_str)
    pad_length = max_len - len(token_ids)
    token_ids = token_ids + [0] * pad_length
    x = torch.tensor([token_ids], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        forecast, _ = model(x)
    return forecast.item()

# -------------------------------
# Main Training Loop with Early Stopping
# -------------------------------
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRPOTransformer(vocab_size=vocab_size, embed_dim=16, num_heads=2, num_layers=2, max_len=max_len, lambda_policy=0.1)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 50
    mape_history = []
    best_mape = float("inf")
    epochs_since_improvement = 0
    early_stop_patience = 10  # if no improvement in 10 epochs, stop
    
    print("Training GRPO Transformer Model (using tokens only)...")
    for epoch in range(n_epochs):
        train_loss = train_model(model, train_loader, optimizer, device, grad_clip=1.0)
        mape_provided = validate_model_using_provided(model, val_loader, device)
        mape_computed = validate_model_using_computed(model, val_loader, device)
        mape_history.append(mape_provided)
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {train_loss:.4f}, Provided MAPE: {mape_provided:.2f}%, Computed MAPE: {mape_computed:.2f}%")
        
        # Early stopping based on provided-target MAPE
        if epoch >= 10:
            recent_best = min(mape_history[-10:])
            if mape_provided > recent_best * 1.01:
                epochs_since_improvement += 1
            else:
                epochs_since_improvement = 0
            if epochs_since_improvement >= early_stop_patience:
                print("Early stopping triggered: No improvement of at least 1% in the last 10 epochs.")
                break
    
    sample_query = "ACB"  # Example token sequence
    predicted_target = query_model(model, sample_query, max_len)
    print(f"Query: {sample_query} -> Predicted Target: {predicted_target:.2f}")

if __name__ == "__main__":
    main()




##############################################################
5)
A) Using MLA and GRPO strategy in DeepSeeker manner
B) add MCTS-inspired method in RL:
C) Not Using the column target in GRPO stage and GRPO learning its reward itself:    
D) Using 'target' only for validation purposes:
E) Using an early stop approach:    
#############################################################
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

df_tokens = pd.read_csv('df_tokens.csv')

# -------------------------------
# Vocabulary and Dataset Preparation
# -------------------------------
all_text = "".join(df_tokens["tokens"].values)
vocab = sorted(list(set(all_text)))
vocab_dict = {char: idx + 1 for idx, char in enumerate(vocab)}  # Reserve 0 for PAD.
# Create inverse vocabulary for decoding
inv_vocab = {v: k for k, v in vocab_dict.items()}
vocab_size = len(vocab_dict)
max_len = df_tokens["tokens"].apply(len).max()

def tokenize(token_str):
    return [vocab_dict[char] for char in token_str]

class TokenDataset(Dataset):
    def __init__(self, df, max_len):
        self.df = df
        self.max_len = max_len
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        token_str = self.df.iloc[idx]["tokens"]
        target = self.df.iloc[idx]["target"]  # Provided target (for validation only)
        token_ids = tokenize(token_str)
        pad_length = self.max_len - len(token_ids)
        token_ids = token_ids + [0] * pad_length
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor([target], dtype=torch.float)

dataset = TokenDataset(df_tokens, max_len)
total_samples = len(dataset)
train_size = int(0.8 * total_samples)
val_size = total_samples - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# -------------------------------
# Positional Encoding Module
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# -------------------------------
# Transformer Model with Extended MLA and GRPO-inspired Loss
# -------------------------------
# (Note: The model still outputs a forecast and a policy value.)
class GRPOTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len, lambda_policy=0.1):
        super(GRPOTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_forecast = nn.Linear(embed_dim, 1)
        self.policy_net = nn.Linear(embed_dim, 1)
        self.lambda_policy = lambda_policy
        
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x) * np.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x).transpose(0, 1)  # Transformer expects (seq_len, batch, embed_dim)
        encoded = self.transformer_encoder(x).mean(dim=0)  # Mean over sequence length => (batch, embed_dim)
        forecast = self.fc_forecast(encoded)
        policy_value = self.policy_net(encoded)
        return forecast, policy_value

# -------------------------------
# Helper Functions to Compute Target from Tokens
# -------------------------------
# Global lookup dictionary (same as used in data generation)
lookup = {'A': 6, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

def compute_target_from_tokens(token_ids):
    total = 0.0
    count = 0
    for i, token in enumerate(token_ids):
        if token != 0:  # ignore PAD tokens
            char = inv_vocab[token]
            total += lookup[char] * (i + 1)
            count += 1
    return total / count if count > 0 else 0.0

def compute_batch_target(x_batch):
    # x_batch: tensor of shape (batch, seq_len)
    targets = []
    for sample in x_batch:
        token_ids = sample.cpu().tolist()
        token_ids = [t for t in token_ids if t != 0]  # Remove PAD tokens
        targets.append(compute_target_from_tokens(token_ids))
    return torch.tensor(targets, dtype=torch.float, device=x_batch.device).unsqueeze(1)

# -------------------------------
# MCTS-Inspired Candidate Generation and Training Function
# -------------------------------
def generate_candidate_paths(token_ids, num_candidates, max_len):
    """
    Given a token sequence (list of ints), generate candidate paths by shuffling non-PAD tokens.
    The candidate is padded to max_len.
    """
    # Remove PAD tokens (0)
    token_list = [int(t) for t in token_ids if int(t) != 0]
    candidates = []
    # Always include the original order as one candidate
    candidates.append(token_list + [0]*(max_len - len(token_list)))
    # Generate additional candidates by shuffling
    for _ in range(num_candidates - 1):
        candidate = token_list.copy()
        np.random.shuffle(candidate)
        candidate = candidate + [0]*(max_len - len(candidate))
        candidates.append(candidate)
    return candidates

def train_model_mcts(model, dataloader, optimizer, device, grad_clip=1.0, num_candidates=3):
    """
    For each sample in a batch, generate num_candidates candidate sequences (paths) by shuffling.
    For each candidate, compute the forecast and policy output and forecast loss (combination of MSE and MAE
    computed against the computed target from that candidate). Then select the candidate with the minimum
    forecast loss and use it to compute the overall loss (forecast loss + GRPO-style policy loss).
    """
    model.train()
    total_loss = 0.0
    for x, _ in dataloader:  # Provided target is ignored during training
        x = x.to(device)  # x shape: (batch, seq_len)
        batch_size = x.size(0)
        batch_forecasts = []
        batch_policy_values = []
        batch_targets = []
        forecast_loss_sum = 0.0
        
        # Process each sample individually (MCTS-style candidate exploration)
        for i in range(batch_size):
            base_seq = x[i].tolist()  # Base token sequence for sample i
            candidate_paths = generate_candidate_paths(base_seq, num_candidates, max_len)
            candidate_tensor = torch.tensor(candidate_paths, dtype=torch.long, device=device)  # (num_candidates, seq_len)
            
            # Forward pass for all candidate paths
            forecasts, policy_values = model(candidate_tensor)  # Each: (num_candidates, 1)
            
            # Compute computed target for each candidate path
            candidate_targets = []
            for candidate in candidate_paths:
                candidate_target = compute_target_from_tokens(candidate)
                candidate_targets.append(candidate_target)
            candidate_targets = torch.tensor(candidate_targets, dtype=torch.float, device=device).unsqueeze(1)
            
            # Compute forecast loss for each candidate: 0.5 * MSE + 0.5 * MAE
            mse_losses = F.mse_loss(forecasts, candidate_targets, reduction='none')  # (num_candidates, 1)
            mae_losses = F.l1_loss(forecasts, candidate_targets, reduction='none')   # (num_candidates, 1)
            candidate_losses = 0.5 * mse_losses + 0.5 * mae_losses  # (num_candidates, 1)
            
            # Select the candidate with the minimal forecast loss
            min_loss, min_index = torch.min(candidate_losses, dim=0)
            selected_forecast = forecasts[min_index]       # (1,)
            selected_policy_value = policy_values[min_index] # (1,)
            selected_target = candidate_targets[min_index]   # (1,)
            forecast_loss_sum += min_loss  # Accumulate the selected forecast loss
            
            batch_forecasts.append(selected_forecast)
            batch_policy_values.append(selected_policy_value)
            batch_targets.append(selected_target)
        
        # Stack selected candidate outputs into batch tensors
        selected_forecasts_batch = torch.stack(batch_forecasts)  # (batch, 1)
        selected_policy_values_batch = torch.stack(batch_policy_values)  # (batch, 1)
        selected_targets_batch = torch.stack(batch_targets)  # (batch, 1)
        
        # Compute GRPO-style policy loss based on group relative advantage
        group_advantage = selected_targets_batch - selected_forecasts_batch  # (batch, 1)
        baseline = group_advantage.mean(dim=0, keepdim=True)
        group_relative_advantage = group_advantage - baseline
        # Compute ratio r_t; add a small epsilon to baseline to avoid division by zero
        r_t = selected_policy_values_batch / (baseline + 1e-6)
        r_t_clipped = torch.clamp(r_t, 0.9, 1.1)
        policy_loss = -torch.min(r_t * group_relative_advantage, r_t_clipped * group_relative_advantage).mean()
        
        # Total loss: average forecast loss over batch + weighted policy loss
        loss = (forecast_loss_sum / batch_size) + model.lambda_policy * policy_loss
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * batch_size
    return total_loss / len(dataloader.dataset)

# -------------------------------
# Validation Functions
# -------------------------------
def validate_model_using_provided(model, dataloader, device):
    # Validation using the provided target from the dataset
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            forecast, _ = model(x)
            all_preds.append(forecast.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    mape = np.mean(np.abs((all_targets - all_preds) / (all_targets + 1e-6))) * 100
    return mape

def validate_model_using_computed(model, dataloader, device):
    # Validation using computed target from tokens
    model.eval()
    all_preds, all_computed = [], []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            computed_target = compute_batch_target(x)
            forecast, _ = model(x)
            all_preds.append(forecast.cpu().numpy())
            all_computed.append(computed_target.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_computed = np.concatenate(all_computed)
    mape = np.mean(np.abs((all_computed - all_preds) / (all_computed + 1e-6))) * 100
    return mape

# -------------------------------
# Interactive Query Function
# -------------------------------
def query_model(model, query_str, max_len):
    token_ids = tokenize(query_str)
    pad_length = max_len - len(token_ids)
    token_ids = token_ids + [0] * pad_length
    x = torch.tensor([token_ids], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        forecast, _ = model(x)
    return forecast.item()

# -------------------------------
# Main Training Loop with Early Stopping (Using MCTS in RL Stage)
# -------------------------------
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRPOTransformer(vocab_size=vocab_size, embed_dim=16, num_heads=2, num_layers=2, max_len=max_len, lambda_policy=0.1)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 50
    mape_history = []
    best_mape = float("inf")
    epochs_since_improvement = 0
    early_stop_patience = 10  # Stop if no improvement in 10 epochs
    
    print("Training GRPO Transformer Model with MCTS-inspired RL (using tokens only)...")
    for epoch in range(n_epochs):
        train_loss = train_model_mcts(model, train_loader, optimizer, device, grad_clip=1.0, num_candidates=3)
        mape_provided = validate_model_using_provided(model, val_loader, device)
        mape_computed = validate_model_using_computed(model, val_loader, device)
        mape_history.append(mape_provided)
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {train_loss:.4f}, Provided MAPE: {mape_provided:.2f}%, Computed MAPE: {mape_computed:.2f}%")
        
        # Early stopping based on provided-target MAPE improvement over recent epochs
        if epoch >= 10:
            recent_best = min(mape_history[-10:])
            if mape_provided > recent_best * 1.01:
                epochs_since_improvement += 1
            else:
                epochs_since_improvement = 0
            if epochs_since_improvement >= early_stop_patience:
                print("Early stopping triggered: No improvement of at least 1% in the last 10 epochs.")
                break
    
    sample_query = "ACB"  # Example token sequence
    predicted_target = query_model(model, sample_query, max_len)
    print(f"Query: {sample_query} -> Predicted Target: {predicted_target:.2f}")

if __name__ == "__main__":
    main()

