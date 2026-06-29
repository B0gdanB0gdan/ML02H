import torch
import torch.nn as nn


class MatrixFactorizationModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 32):
        super().__init__()

        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor):
        p_u = self.user_emb(user_idx)
        q_i = self.item_emb(item_idx)
        b_u = self.user_bias(user_idx).squeeze(1)
        b_i = self.item_bias(item_idx).squeeze(1)

        dot = (p_u * q_i).sum(dim=-1)
        score = dot + b_u + b_i 
        return torch.sigmoid(score)
    

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
 
    for batch in loader:
        
        users, items, ratings = batch
        users, items, ratings = users.to(device), items.to(device), ratings.to(device)
        preds = model(users, items)
 
        loss = criterion(preds, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(ratings)
 
    return total_loss / len(loader.dataset)