import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch_geometric.data import DataLoader
from models import GiG
from torch_geometric.data import Data
import pickle
import os
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features):
        features = F.normalize(features, p=2, dim=1)  # Normalize embeddings
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Mask out self-similarity
        mask = torch.eye(similarity_matrix.size(0), device=features.device).bool()
        similarity_matrix.masked_fill_(mask, -1)  # Mask self-similarity

        # 🔹 Select hardest negatives (max similarity excluding diagonal)
        hardest_negatives = similarity_matrix.max(dim=1).values.unsqueeze(1)  # Shape: [batch_size, 1]

        # 🔹 Compute contrastive loss using hardest negatives
        labels = torch.arange(features.size(0), device=features.device)
        logits = torch.cat([similarity_matrix, hardest_negatives], dim=1)  # Add hardest negatives
        contrastive_loss = F.cross_entropy(logits, labels)

        return contrastive_loss



def optimized_collate_fn(batch):
    batch_size = len(batch)
    cumsum_nodes = 0

    adjusted_edge_indices = []
    for data in batch:
        if data.edge_index.max() >= data.num_nodes:
            print(f"ERROR in single graph: edge_index max {data.edge_index.max()} exceeds num_nodes {data.num_nodes}")
            continue  # Skip problematic graphs
        edge_index = data.edge_index + cumsum_nodes
        adjusted_edge_indices.append(edge_index)
        cumsum_nodes += data.num_nodes

    x = torch.cat([data.x for data in batch], dim=0)  # Keep node-level x
    y = torch.cat([data.y for data in batch], dim=0)  # Ensure y is also batched correctly

    edge_index = torch.cat(adjusted_edge_indices, dim=1)

    max_valid_index = x.shape[0] - 1
    edge_index = torch.clamp(edge_index, min=0, max=max_valid_index)
    # edge_attr = torch.cat([data.edge_attr for data in batch], dim=0) if batch[0].edge_attr is not None else None

    edge_attrs = [data.edge_attr for data in batch if data.edge_attr is not None]
    if edge_attrs:
        edge_attr = torch.cat(edge_attrs, dim=0)

        # Fix NaNs or Infs in `edge_attr`
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=1.0, neginf=0.0)

        # Ensure valid argmax operation
        assert edge_attr.shape[1] > 1, " edge_attr has too few features for argmax!"
    else:
        edge_attr = None

    # Fix `edge_index` to prevent out-of-bounds errors
    assert edge_index.max() < x.shape[
        0], f" edge_index contains out-of-bounds values! Max index: {edge_index.max()} Num nodes: {x.shape[0]}"

    #  Prevent negative indices (if any)
    edge_index = torch.clamp(edge_index, min=0, max=x.shape[0] - 1)
    batch_tensor = torch.cat([torch.full((data.num_nodes,), i, dtype=torch.long) for i, data in enumerate(batch)])

    original_ids = torch.cat([
        torch.tensor(data.original_ids, dtype=torch.long) if isinstance(data.original_ids, list) else data.original_ids
        for data in batch if data.original_ids is not None
    ])

    return Data(
        x=x,
        y=y,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch_tensor,
        original_ids=original_ids,
        batch_size=batch_size  # Explicitly store batch size
    )


config = {
    "num_node_features": 256,
    "input_dim": 256,  # Dimension of node embeddings
    "output_dim": 2405,  # Multi-class classification
    "node_level_module": "GAT",  # "GraphConv", "GIN"
    "projection_layers": [256, 256],
    "node_layers": [64],  # Hidden layers for node-level processing
    "pooling": "add",  # "add" or "mean"
    "population_level_module": "LGLKL",  # "LGL", "LGLKL"
    "mu": 0.5,  # Initial mean for KL loss
    "sigma": 0.5,  # Initial standard deviation for KL loss
    "population_layers": [128],  # Layers for population-level module
    "temp": 1.0,  # Learnable temperature
    "theta": 1.0,  # Learnable threshold
    "gnn_type": "GraphConv",  # "GraphConv", "GAT", "SAGEConv", etc.
    "gnn_layers": [128],  # GNN layers

    "gnn_aggr": "mean",
    "classifier_layers": [128, 2405],  # 2405 classes
    "batch_size": 128,
    "epochs": 20,
    "lr": 2e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "wandb_project": "true_gene-classification",
    "num_embeddings": 105220  # Total unique node IDs
}


class GeneClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Node Embeddings for `original_node_id`
        self.global_node_embedding = nn.Embedding(
            num_embeddings=config["num_embeddings"], embedding_dim=config["input_dim"]
        )
        self.embedding_projection = nn.Sequential(
            nn.Linear(config["input_dim"], 128),
            nn.LeakyReLU(),
            nn.Linear(128, config["input_dim"])
        )
        self.model = GiG(config)

    def forward(self, data):
        # Replace `data.x` with embedded node IDs
        if isinstance(data.original_ids, list):
            max_length = max(len(ids) for ids in data.original_ids)
            padded_ids = [ids + [0] * (max_length - len(ids)) for ids in data.original_ids]  # Pad with 0s
            data.original_ids = torch.tensor(padded_ids, dtype=torch.long, device=data.x.device)

            # Ensure values are within embedding range
        data.original_ids = torch.clamp(data.original_ids, min=0, max=self.global_node_embedding.num_embeddings - 1)

        # print("Original IDs shape:", data.original_ids.shape)
        # print("Min ID:", data.original_ids.min().item(), "Max ID:", data.original_ids.max().item())
        # print("Embedding Num:", self.global_node_embedding.num_embeddings)
        # Apply embedding layer (Shape: [batch_size, max_length, embedding_dim])
        embedded_x = self.global_node_embedding(data.original_ids)
        #  Flatten original_ids: it should match the number of nodes
        total_nodes_in_batch = data.batch.shape[0]  # Total nodes across all graphs

        #  Ensure `original_ids` is within valid embedding range
        data.original_ids = torch.clamp(data.original_ids, min=0, max=self.global_node_embedding.num_embeddings - 1)

        #  Compute per-graph embeddings (batch_size, embedding_dim)
        graph_embeddings = self.embedding_projection(self.global_node_embedding(data.original_ids))

        #  Expand per-graph embeddings to match total nodes in batch
        data.x = graph_embeddings[data.batch]  # Expands to (total_nodes_in_batch, embedding_dim)
        data.x = data.x.mean(dim=1)  # Shape: [batch_size, embedding_dim]

        # print("Batch size:", data.batch.max().item() + 1)  # Should print batch size
        # print("Total nodes in batch:", total_nodes_in_batch)
        # print("x shape after expansion:", data.x.shape)  # Should match edge_index

        # Aggregate embeddings across `max_length`

        data.x = torch.nan_to_num(data.x, nan=0.0)
        '''
        print("NaN count in data.x after fix:", torch.isnan(data.x).sum().item())
        print("=== Forward Pass of gene classifier ===")
        print("Data.x shape:", data.x.shape, "Data.x dtype:", data.x.dtype)
        print("Data.edge_index shape:", data.edge_index.shape)
        print("Data.edge_attr shape:", None if data.edge_attr is None else data.edge_attr.shape)
        print("Data.batch shape:", None if data.batch is None else data.batch.shape)
        print("Data.y shape:", None if data.y is None else data.y.shape)
        '''
        return self.model(data)


def train(model, train_loader, val_loader, config):
    wandb.init(project=config["wandb_project"], config=config)

    device = config["device"]
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()  # Classification Loss
    contrastive_loss_fn = ContrastiveLoss(temperature=0.1)  # Stronger Contrastive Loss

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"last_models/run_{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    for epoch in range(config["epochs"]):
        model.train()
        total_loss, total_cl_loss, total_class_loss = 0.0, 0.0, 0.0
        correct_train, total_train = 0, 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # 🔹 Forward Pass: Get features earlier in the network
            outputs, graph_features, node_features, _, _, _ = model(batch)

            # Compute contrastive loss (on node features)
            cl_loss = contrastive_loss_fn(node_features)

            # Compute classification loss
            class_loss = criterion(outputs, batch.y.long())

            # Weighted loss
            loss = 0.4 * class_loss + 0.6 * cl_loss  # 🔥 Contrastive loss is more dominant

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            total_loss += loss.item()
            total_class_loss += class_loss.item()
            total_cl_loss += cl_loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == batch.y).sum().item()
            total_train += batch.y.size(0)

        avg_train_loss = total_loss / len(train_loader)
        avg_class_loss = total_class_loss / len(train_loader)
        avg_cl_loss = total_cl_loss / len(train_loader)
        train_acc = correct_train / total_train

        val_loss, val_acc, val_cl_loss = evaluate(model, val_loader, criterion, contrastive_loss_fn, device)

        wandb.log({"epoch": epoch + 1,
                   "train_loss": avg_train_loss,
                   "train_acc": train_acc,
                   "val_loss": val_loss,
                   "val_acc": val_acc,
                   "contrastive_loss": avg_cl_loss,
                   "val_contrastive_loss": val_cl_loss})  # Log validation contrastive loss

        print(f"Epoch [{epoch + 1}/{config['epochs']}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"Contrastive Loss: {avg_cl_loss:.4f}, Val Contrastive Loss: {val_cl_loss:.4f}")

    final_model_path = f"{run_dir}/final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"🏁 Final model saved as: {final_model_path}")
    wandb.finish()

def evaluate(model, val_loader, criterion, contrastive_loss_fn, device):
    model.eval()
    total_loss, total_cl_loss, total_class_loss = 0.0, 0.0, 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # 🔹 Forward Pass
            outputs, graph_features, node_features, _, _, _ = model(batch)

            # Compute classification loss
            class_loss = criterion(outputs, batch.y.long())

            # Compute contrastive loss (on node features)
            cl_loss = contrastive_loss_fn(node_features)

            # Total loss = classification loss + contrastive loss
            loss = 0.4 * class_loss + 0.6 * cl_loss

            total_loss += loss.item()
            total_class_loss += class_loss.item()
            total_cl_loss += cl_loss.item()

            # Compute accuracy
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.y.size(0)

    avg_loss = total_loss / len(val_loader)
    avg_class_loss = total_class_loss / len(val_loader)
    avg_cl_loss = total_cl_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy, avg_cl_loss  # Return contrastive loss too


if __name__ == "__main__":
    # Load train, validation, and test datasets

    with open("./DataLast/corrected_datasets/train_shuffled_y.pkl", "rb") as f:
        train_dataset = pickle.load(f)

    with open("./DataLast/corrected_datasets/val_shuffled_y.pkl", "rb") as f:
        val_dataset = pickle.load(f)

    with open("./DataLast/corrected_datasets/test_shuffled_y.pkl", "rb") as f:
        test_dataset = pickle.load(f)

    # %%
    train_dataset = train_dataset[0:20000]
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              collate_fn=optimized_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                            collate_fn=optimized_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False,
                             collate_fn=optimized_collate_fn)
    # %%
    # Initialize Model
    model = GeneClassifier(config)

    # Train
    train(model, train_loader, val_loader, config)

    # Final Evaluation on Test Set
    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), config["device"])
    print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
