import torch
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from models import GiG
from datasets.dataloader import DataLoader
import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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


def visualize_learned_graph(adj, predictions, labels, misc_indices, epoch, output_path="./population_graphs"):
    adj_np = adj.detach().cpu().numpy()

    misclassified = (predictions != labels).nonzero()[0]

    # Define node colors
    node_colors = []
    for i, label in enumerate(labels):
        if i in misclassified:
            node_colors.append("red")  # Red for misclassified nodes
        else:
            node_colors.append("green")

    # Create graph from filtered adjacency matrix
    G = nx.from_numpy_array(adj_np)
    # Add node labels as attributes (optional)
    if labels is not None:
        for i, label in enumerate(labels):
            G.nodes[i]["label"] = label

    with open(os.path.join(output_dir, f"learned_graph_epoch_{epoch}.pkl"), "wb") as f:
        pickle.dump(G, f)
    # Visualize graph
    plt.figure(figsize=(10, 8))
    nx.draw(
        G,
        node_color=node_colors[:len(G.nodes())],  # Ensure the color matches the node count
        with_labels=True,
        edge_color="gray",
        cmap=plt.cm.Blues
    )
    plt.title(f"Learned Population Graph with Misclassified Nodes Highlighted Epoch {epoch}")

    # Save the graph
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, f"learned_graph_with_misclassified_nodes_epoch_{epoch}.png"))
    plt.show()


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
output_dir = "./population_graphs"
os.makedirs(output_dir, exist_ok=True)


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    adj = None
    batch_idx = 0
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)

        batch.y.to(device)
        optimizer.zero_grad()
        predictions, _, _, _, adj = model(batch)  # Forward pass

        loss = criterion(predictions, batch.y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        total_loss += loss.item()
    print(total_loss)
    # Save the adjacency matrix
    adj_np = adj.detach().cpu().numpy()
    save_path = os.path.join(output_dir, f"epoch_{epoch}_batch_{batch_idx}.npy")
    np.save(save_path, adj_np)
    # Optional: Visualize the graph
    G = nx.from_numpy_array(adj_np)
    plt.figure(figsize=(10, 8))
    nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray")
    plt.title(f"Population Graph - Epoch {epoch}, Batch {batch_idx}")
    plt.savefig(os.path.join(output_dir, f"graph_epoch_{epoch}_batch_{batch_idx}.png"))
    plt.close()
    return total_loss / len(train_loader)


def evaluate(model, loader, device, epoch):
    model.eval()
    all_preds = []
    all_labels = []
    misclassified_indices = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions, _, _, _, adj = model(batch)
            predicted_classes = predictions.argmax(dim=1).cpu().numpy()
            labels = batch.y.cpu().numpy()
            # Compare predictions and labels
            misclassified = (predicted_classes != labels).nonzero()[0]
            misclassified_indices.extend(misclassified.tolist())  # Collect misclassified indices


            all_preds.extend(predicted_classes)
            all_labels.extend(labels)

            # Calculate overall accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        visualize_learned_graph(adj, predicted_classes, labels, misclassified_indices, epoch)
        return accuracy, misclassified_indices


epochs = 30
for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

    val_accuracy, misclassified_indices = evaluate(model, val_loader, device, epoch)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# test the model
test_accuracy = evaluate(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.4f}")
