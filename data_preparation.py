import json
import logging
from collections import Counter
from datetime import datetime
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split


# set log format
handlers_list = [logging.StreamHandler()]

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

logger = logging.getLogger(__name__)


"""
Create your data loader for training/testing local & global model.
Keep the value of the return variable for normal operation.
"""

def collate_fn(batch):
    sequences, pKa_values = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences, torch.tensor(pKa_values)

class SMILESDataset(Dataset):
    def __init__(self, smiles_list, pKa_list):
        self.smiles_list = smiles_list
        self.pKa_list = pKa_list
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(list(set(''.join(smiles_list))))
        
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        encoded = self.label_encoder.transform(list(smiles))
        pKa = self.pKa_list[idx]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(pKa, dtype=torch.float32)

def load_partition(dataset, validation_split, batch_size):
    """
    The variables train_loader, val_loader, and test_loader must be returned fixedly.
    """

    file_path = './carboxylic_data.csv'
    data = pd.read_csv(file_path)

    filtered_data = data.dropna(subset=['pKa'])
    smiles_list = filtered_data['SMILES'].tolist()
    pKa_list = filtered_data['pKa'].tolist()

    dataset = SMILESDataset(smiles_list, pKa_list)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # 데이터 분할
    train_data, test_data = train_test_split(filtered_data, test_size=0.2, random_state=42)
    train_smiles = train_data['SMILES'].tolist()
    train_pKa = train_data['pKa'].tolist()
    test_smiles = test_data['SMILES'].tolist()
    test_pKa = test_data['pKa'].tolist()

    train_dataset = SMILESDataset(train_smiles, train_pKa)
    test_dataset = SMILESDataset(test_smiles, test_pKa)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)



    # DataLoader for client training, validation, and test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader

def gl_model_torch_validation(batch_size):
    """
    Setting up a dataset to evaluate a global model on the server
    """

    file_path = './carboxylic_data.csv'
    data = pd.read_csv(file_path)

    filtered_data = data.dropna(subset=['pKa'])
    smiles_list = filtered_data['SMILES'].tolist()
    pKa_list = filtered_data['pKa'].tolist()

    dataset = SMILESDataset(smiles_list, pKa_list)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # 데이터 분할
    train_data, test_data = train_test_split(filtered_data, test_size=0.2, random_state=42)
    train_smiles = train_data['SMILES'].tolist()
    train_pKa = train_data['pKa'].tolist()
    test_smiles = test_data['SMILES'].tolist()
    test_pKa = test_data['pKa'].tolist()

    train_dataset = SMILESDataset(train_smiles, train_pKa)
    test_dataset = SMILESDataset(test_smiles, test_pKa)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # DataLoader for global model validation
    gl_val_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return gl_val_loader