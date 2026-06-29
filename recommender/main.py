from cbf_model import ContentFilteringModel, train_epoch as cbf_train_epoch
from cf import MatrixFactorizationModel, train_epoch as cf_train_epoch
import torch
import torch.nn as nn
from data import get_data_loaders


DEVICE = 'cpu'
BATCH_SIZE = 64
DATA_DIR = 'recommender/data'
EPOCHS = 5


def train_cbf_model(data):
    meta = data["meta"]
    cbf_train_loader, cbf_test_loader = data["cbf"]
    model = ContentFilteringModel(
        input_dim=2*meta["n_features"]
    )
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    for _ in range(EPOCHS):
        loss = cbf_train_epoch(model, cbf_train_loader, optimizer, criterion, DEVICE)
        print(loss)

def train_cf_model(data): 
    meta = data["meta"]
    cf_train_loader, cf_test_loader = data["cf"]
    model = MatrixFactorizationModel(
        meta["n_users"], 
        meta["n_items"], 
        embedding_dim=32
    )
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    for _ in range(EPOCHS):
        loss = cf_train_epoch(model, cf_train_loader, optimizer, criterion, DEVICE)
        print(loss)

    

def main():
    data = get_data_loaders(DATA_DIR, BATCH_SIZE)
    print("CBF model training:")
    train_cbf_model(data)
    print("CF model training:")
    train_cf_model(data)
    

if __name__ == "__main__":
    main()
 