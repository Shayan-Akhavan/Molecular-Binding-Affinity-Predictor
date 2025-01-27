import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

class MolecularGraphDataset(Dataset):
    def __init__(self, smiles_list, targets):
        self.smiles_list = smiles_list
        self.targets = targets
        
    def mol_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        
        # Get node features (atom features)
        node_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetNumRadicalElectrons(),
                int(atom.GetIsAromatic()),
                atom.GetHybridization()
            ]
            node_features.append(features)
        
        # Get edge indices (bonds)
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])  # Add reverse edge for undirected graph
            
        return torch.tensor(node_features, dtype=torch.float), \
               torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        target = self.targets[idx]
        
        x, edge_index = self.mol_to_graph(smiles)
        target = torch.tensor([[target]], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=target)

class RangeAwareLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super(RangeAwareLoss, self).__init__()
        self.alpha = alpha  # Weight for MSE loss
        self.beta = beta   # Weight for range penalty
        
    def forward(self, pred, target):
        # Basic MSE loss
        mse_loss = F.mse_loss(pred, target)
        
        # Range penalty to discourage regression to mean
        mean_pred = torch.mean(pred)
        pred_variance = torch.var(pred)
        target_variance = torch.var(target)
        range_penalty = F.mse_loss(pred_variance, target_variance)
        
        # Combined loss
        total_loss = self.alpha * mse_loss + self.beta * range_penalty
        return total_loss

class RangeAwareGNN(torch.nn.Module):
    def __init__(self, node_features, hidden_channels):
        super(RangeAwareGNN, self).__init__()
        
        # First GNN layer
        self.conv1 = GATConv(node_features, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        # Second GNN layer
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)  # Make sure this matches hidden_channels
        
        # Third GNN layer
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        # Output layers
        self.linear1 = nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, 1)
        
    def forward(self, x, edge_index, batch):
        # First layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Third layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Output layers
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        
        return x

# When initializing the model, make sure to use consistent dimensions:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RangeAwareGNN(node_features=6, hidden_channels=64).to(device)

def fetch_chembl_data(target_id, min_confidence=8):
    """Fetch binding data from ChEMBL API for a specific target"""
    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity"
    params = {
        "target_chembl_id": target_id,
        "type": "IC50",
        "relation": "=",
        "confidence_score__gte": min_confidence,
        "format": "json"
    }
    
    response = requests.get(base_url, params=params)
    data = response.json()
    
    activities = []
    for activity in data["activities"]:
        if activity.get("canonical_smiles") and activity.get("value"):
            activities.append({
                "smiles": activity["canonical_smiles"],
                "activity": float(activity["value"]),
                "units": activity["standard_units"]
            })
    
    return pd.DataFrame(activities)

def preprocess_data(df):
    """Preprocess the activity data"""
    # Convert to log scale (typically used for IC50 values)
    df['pActivity'] = -np.log10(df['activity'] * 1e-9)  # Convert to M if in nM
    
    # Remove invalid values
    df = df.dropna()
    df = df[df['pActivity'].between(0, 15)]  # Remove unrealistic values
    
    return df

def create_stratified_loader(dataset, batch_size, activity_bins=10):
    """Create a data loader that ensures representation across the activity range"""
    activities = [data.y.item() for data in dataset]
    
    # Create bins based on activity values
    bins = np.linspace(min(activities), max(activities), activity_bins + 1)
    bin_indices = np.digitize(activities, bins) - 1
    
    # Create weighted sampler to balance across bins
    bin_counts = np.bincount(bin_indices)
    weights = 1.0 / bin_counts[bin_indices]
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

def train_range_aware_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate_range_aware_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            
            predictions.extend(out.cpu().numpy().flatten())
            actuals.extend(data.y.cpu().numpy().flatten())
    
    # Calculate metrics
    mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
    r2 = np.corrcoef(predictions, actuals)[0, 1]**2
    
    # Calculate range-specific metrics
    pred_range = max(predictions) - min(predictions)
    actual_range = max(actuals) - min(actuals)
    range_coverage = min(pred_range / actual_range, actual_range / pred_range)
    
    return {
        'loss': total_loss / len(loader),
        'mae': mae,
        'r2': r2,
        'range_coverage': range_coverage,
        'predictions': predictions,
        'actuals': actuals
    }

def analyze_predictions(smiles_list, predictions, actuals, n=5):
    """Analyze top performing and worst performing predictions"""
    errors = np.abs(np.array(predictions) - np.array(actuals))
    
    # Get indices for best and worst predictions
    best_indices = np.argsort(errors)[:n]
    worst_indices = np.argsort(errors)[-n:][::-1]
    
    print("\nTop {} Best Predictions:".format(n))
    for idx in best_indices:
        print(f"SMILES: {smiles_list[idx]}")
        print(f"Predicted: {predictions[idx]:.2f}, Actual: {actuals[idx]:.2f}, Error: {errors[idx]:.2f}")
        print(f"Structure features: {analyze_structure(smiles_list[idx])}\n")
    
    print("\nTop {} Worst Predictions:".format(n))
    for idx in worst_indices:
        print(f"SMILES: {smiles_list[idx]}")
        print(f"Predicted: {predictions[idx]:.2f}, Actual: {actuals[idx]:.2f}, Error: {errors[idx]:.2f}")
        print(f"Structure features: {analyze_structure(smiles_list[idx])}\n")

def analyze_structure(smiles):
    """Analyze key structural features of a molecule"""
    mol = Chem.MolFromSmiles(smiles)
    features = []
    
    # Check for common functional groups
    if '[N+](=O)[O-]' in smiles:
        features.append('nitro group')
    if 'C#N' in smiles:
        features.append('cyano group')
    if 'O)c' in smiles or 'c(O' in smiles:
        features.append('hydroxyl group')
    if 'OC' in smiles:
        features.append('methoxy group')
    if 'NN' in smiles:
        features.append('hydrazine link')
        
    # Count rings
    ring_count = Chem.rdMolDescriptors.CalcNumRings(mol)
    features.append(f'{ring_count} rings')
    
    return ', '.join(features)

def visualize_results(predictions, actuals):
    """Create visualization of predictions vs actuals"""
    
    # Set the style for better visualization
    #plt.style.use('seaborn')
    plt.style.use('seaborn-v0_8')

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    ax1.scatter(actuals, predictions, alpha=0.5, color='blue')
    ax1.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--', label='Perfect Prediction')
    ax1.set_xlabel('Actual pActivity')
    ax1.set_ylabel('Predicted pActivity')
    ax1.set_title('Predicted vs Actual Binding Affinities')
    ax1.legend()

    # Distribution of errors
    errors = np.array(predictions) - np.array(actuals)
    sns.histplot(errors, bins=30, ax=ax2, color='blue', alpha=0.6)
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Prediction Errors')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    # Also save the plots
    fig.savefig('binding_affinity_results.png')
    plt.close()

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Fetch and preprocess data
    print("Fetching data from ChEMBL...")
    target_id = "CHEMBL203"  # EGFR
    df = fetch_chembl_data(target_id)
    df = preprocess_data(df)
    
    print(f"Dataset size: {len(df)} compounds")
    print("\nTarget protein: EGFR (CHEMBL203)")
    print(f"Activity range: {df['pActivity'].min():.2f} - {df['pActivity'].max():.2f} pActivity")
    
    # Create dataset
    dataset = MolecularGraphDataset(df['smiles'].values, df['pActivity'].values)
    
    # Split dataset
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    print(f"\nTraining set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = create_stratified_loader(train_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model and training components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = RangeAwareGNN(node_features=6, hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Training loop
    num_epochs = 100
    best_r2 = 0
    training_history = []
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        train_loss = train_range_aware_model(model, train_loader, optimizer, criterion, device)
        test_metrics = evaluate_range_aware_model(model, test_loader, criterion, device)
        
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_metrics['loss'],
            'test_r2': test_metrics['r2'],
            'test_mae': test_metrics['mae']
        })
        
        if test_metrics['r2'] > best_r2:
            best_r2 = test_metrics['r2']
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pt')
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Test Loss: {test_metrics["loss"]:.4f}')
            print(f'  Test R²: {test_metrics["r2"]:.4f}')
            print(f'  Test MAE: {test_metrics["mae"]:.4f}')
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('best_model.pt'))
    final_metrics = evaluate_range_aware_model(model, test_loader, criterion, device)
    
    print("\nFinal Model Performance:")
    print(f"Best epoch: {best_epoch}")
    print(f"Test R²: {final_metrics['r2']:.4f}")
    print(f"Test MAE: {final_metrics['mae']:.4f}")
    
    # Analyze predictions
    visualize_results(final_metrics['predictions'], final_metrics['actuals'])
    analyze_predictions(df['smiles'].values[-len(test_dataset):], 
                          final_metrics['predictions'], 
                          final_metrics['actuals'])