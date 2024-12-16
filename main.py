import torch
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from models import GiG
from datasets.dataloader import DataLoader
import pickle

# Step 1: Configuration
config = {
    "node_level_module": "GIN",
    "num_node_features": 7,
    "node_layers": [32, 16],
    "node_level_hidden_layers_number": len([32, 16]),  # Number of hidden layers
    "pooling": "mean",
    "population_level_module": "LGL",
    "population_layers": [16, 8],
    "temp": 0.5,
    "theta": 0.1,
    "gnn_type": "GraphConv",
    "gnn_layers": [8],
    "gnn_aggr": "mean",
    "classifier_layers": [8],
    "output_dim": 2405,
}

# Load the train dataset
with open("./Graph Outputs/train_pg_subgraph.pkl", "rb") as f:
    train_pg_subgraph = pickle.load(f)

    # Load the test dataset
with open("./Graph Outputs/val_pg_subgraph.pkl", "rb") as f:
    val_pg_subgraph = pickle.load(f)

with open("./Graph Outputs/test_pg_subgraph.pkl", "rb") as f:
    test_pg_subgraph = pickle.load(f)
print("pickles loaded")

# Create a mapping from gene IDs to class indices
unique_genes = set()
for graph in train_pg_subgraph + val_pg_subgraph + test_pg_subgraph:
    if hasattr(graph, 'true_gene_ids'):
        unique_genes.update(graph.true_gene_ids)

gene_to_class = {gene: i for i, gene in enumerate(unique_genes)}


# Update ground truth `y` in each graph
def map_labels(dataset):
    for graph in dataset:
        if hasattr(graph, 'true_gene_ids') and graph.true_gene_ids:
            graph.y = torch.tensor([gene_to_class[gene] for gene in graph.true_gene_ids], dtype=torch.long)


map_labels(train_pg_subgraph)
map_labels(val_pg_subgraph)
map_labels(test_pg_subgraph)
config['output_dim'] = len(gene_to_class)


batch_size = 32
train_loader = DataLoader(train_pg_subgraph, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_pg_subgraph, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_pg_subgraph, batch_size=batch_size, shuffle=False)
print("data loaded")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda: ", torch.cuda.is_available())
model = GiG(config).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        predictions, _, _, _, _ = model(batch)  # Forward pass
        """
        preds = predictions.argmax(dim=1).cpu().numpy()
        labels = batch.y.cpu().numpy()
        print(f"Predictions: {preds}")
        print(f"True Labels: {labels}")
    """
        loss = criterion(predictions, batch.y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions, _, _, _, _ = model(batch)
            preds = predictions.argmax(dim=1).cpu().numpy()
            labels = batch.y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    return accuracy_score(all_labels, all_preds)


epochs = 3
for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_accuracy = evaluate(model, val_loader, device)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# test the model
test_accuracy = evaluate(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.4f}")
