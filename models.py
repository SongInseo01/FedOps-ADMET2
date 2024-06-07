from sklearn.metrics import f1_score
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

class SMILESModel(nn.Module):
    def __init__(self, output_size, n_layers=1):
        super(SMILESModel, self).__init__()
        input_dim = 25
        hidden_dim = 128
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

def train_torch():
    def custom_train_torch(model, train_loader, epochs, cfg):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])

        print("Starting training...")
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, targets in tqdm(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

        return model

    return custom_train_torch

def test_torch():
    def custom_test_torch(model, test_loader, cfg):
        criterion = nn.MSELoss()
        model.eval()
        test_losses = []
        all_targets = []
        all_predictions = []

        print("Starting evaluation...")
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader):
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                test_losses.append(loss.item())
                all_targets.extend(targets.numpy())
                all_predictions.extend(outputs.numpy())

        average_loss = sum(test_losses) / len(test_losses)
        accuracy = None  # Accuracy metric can be defined based on the specific task, if required

        print(f'Mean Test Loss: {average_loss:.4f}')


        model.to("cpu")  # move model back to CPU
        return average_loss, accuracy

    return custom_test_torch