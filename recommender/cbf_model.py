import torch
import torch.nn as nn


class ContentFilteringModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1), 
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        return self.net(x).squeeze(1)
    

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
 
    for x, ratings in loader:
        x, ratings = x.to(device), ratings.to(device)
        preds = model(x)
        
        loss = criterion(preds, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(ratings)
 
    return total_loss / len(loader.dataset)