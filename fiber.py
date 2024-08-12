import torch
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm
import os

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Enable cuDNN autotuner
torch.backends.cudnn.benchmark = True

# Standardization function
def standardize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std, mean, std

class PulseDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.data = self.preload_data()

    def preload_data(self):
        data = []
        for file in tqdm(self.file_list, desc="Preloading data"):
            npz_data = np.load(file)
            initial_pulse_real, _, _ = standardize_data(npz_data['initial_pulse_real'])
            initial_pulse_imag, _, _ = standardize_data(npz_data['initial_pulse_imag'])
            output_pulse_real, _, _ = standardize_data(npz_data['output_pulse_real'])
            output_pulse_imag, _, _ = standardize_data(npz_data['output_pulse_imag'])
            fiber_params, _, _ = standardize_data(npz_data['fiber_params'])

            initial_pulse = np.stack((initial_pulse_real, initial_pulse_imag), axis=0)
            output_pulse = np.stack((output_pulse_real, output_pulse_imag), axis=0)

            data.append((initial_pulse, fiber_params, output_pulse))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        initial_pulse, fiber_params, output_pulse = self.data[idx]
        return (torch.from_numpy(initial_pulse).float(),
                torch.from_numpy(fiber_params).float(),
                torch.from_numpy(output_pulse).float())

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class BranchNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BranchNet, self).__init__()
        self.conv1 = nn.Conv1d(2, hidden_size, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_size, hidden_size),
            ResidualBlock(hidden_size, hidden_size),
            ResidualBlock(hidden_size, hidden_size)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x is expected to be of shape [batch, 2, 1024]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)
        x = self.global_avg_pool(x).squeeze(2)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size])).to(device)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = torch.softmax(attention, dim=-1)
        
        x = torch.matmul(attention, V)
        return x

class TrunkNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TrunkNet, self).__init__()
        self.attention = SelfAttention(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.attention(x.unsqueeze(1)).squeeze(1)
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

class DeepONet(nn.Module):
    def __init__(self, branch_input_size, trunk_input_size, hidden_size, output_size):
        super(DeepONet, self).__init__()
        self.branch_net = BranchNet(branch_input_size, hidden_size)
        self.trunk_net = TrunkNet(trunk_input_size, hidden_size)
        
        # Multi-layer FNN after multiplication
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, output_size * 2)
        )

    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        combined = branch_output * trunk_output
        output = self.mlp(combined)
        return output.view(branch_input.size(0), 2, -1)

def train(model, train_loader, optimizer, criterion, device, scheduler):
    model.train()
    total_loss = 0
    for initial_pulse, fiber_params, output_pulse in train_loader:
        initial_pulse = initial_pulse.to(device, non_blocking=True)
        fiber_params = fiber_params.to(device, non_blocking=True)
        output_pulse = output_pulse.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        outputs = model(initial_pulse, fiber_params)
        loss = criterion(outputs, output_pulse)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for initial_pulse, fiber_params, output_pulse in test_loader:
            initial_pulse = initial_pulse.to(device, non_blocking=True)
            fiber_params = fiber_params.to(device, non_blocking=True)
            output_pulse = output_pulse.to(device, non_blocking=True)

            outputs = model(initial_pulse, fiber_params)
            loss = criterion(outputs, output_pulse)
            total_loss += loss.item()
    return total_loss / len(test_loader)

def main():
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    file_list = glob.glob('dataset/*.npz')
    dataset = PulseDataset(file_list)

    train_size = int(0.95 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    batch_size = 256
    num_workers = min(os.cpu_count(), 8)  # Limit to 8 workers max
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    branch_input_size = 1024
    trunk_input_size = 5
    hidden_size = 512
    output_size = 1024

    model = DeepONet(branch_input_size, trunk_input_size, hidden_size, output_size).to(device)
    model = nn.DataParallel(model)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    criterion = nn.MSELoss()

    num_epochs = 10000
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device, scheduler)
        train_losses.append(train_loss)
        
        if epoch % 5 == 0:  # Validate every 5 epochs
            val_loss = validate(model, test_loader, criterion, device)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)  # Update the learning rate based on validation loss
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.module.state_dict(), 'best_model.pth')
                print(f'New best model saved with validation loss: {best_val_loss:.6f}')
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}')

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(range(0, len(val_losses)*5, 5), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.close()

    print("Training completed. Best model saved as 'best_model.pth'")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
