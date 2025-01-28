import pickle
import torch
import wandb
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.loader.dataloader import DataLoader as GeoDataLoader
from models import LGL
from models import GiG

from pytorch_lightning.loggers import WandbLogger


def create_id_to_class_mapping(dataset_files):
    id_to_class = {}
    current_class = 0
    for dataset_file in dataset_files:
        with open(dataset_file, 'rb') as f:
            graphs = pickle.load(f)
            for graph in graphs:
                for gene_id in graph.true_gene_ids:
                    if gene_id not in id_to_class:
                        id_to_class[gene_id] = current_class
                        current_class += 1
    return id_to_class


# Custom Dataset
class PGSubgraphDataset(Dataset):
    def __init__(self, pkl_file, id_to_class):
        with open(pkl_file, 'rb') as f:
            self.graphs = pickle.load(f)

        self.id_to_class = id_to_class
        num_classes = len(self.id_to_class)
        # Assign one-hot encoded y to each graph
        for graph in self.graphs:
            y = torch.zeros(num_classes)  # One-hot vector of size [len(id_to_class)]
            for gene_id in graph.true_gene_ids:
                if gene_id in self.id_to_class:
                    y[self.id_to_class[gene_id]] = 1
            graph.y = y

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


# Lightning Data Module
class PGSubgraphDataModule(LightningDataModule):
    def __init__(self, train_file, val_file, test_file, id_to_class, batch_size=32):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.id_to_class = id_to_class

    def setup(self, stage=None):
        self.train_dataset = PGSubgraphDataset(self.train_file, self.id_to_class)
        self.val_dataset = PGSubgraphDataset(self.val_file, self.id_to_class)
        self.test_dataset = PGSubgraphDataset(self.test_file, self.id_to_class)

    def train_dataloader(self):
        return GeoDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return GeoDataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return GeoDataLoader(self.test_dataset, batch_size=self.batch_size)


# GiG Model
class GiGModel(LightningModule):
    def __init__(self, config, lambda_rank=0.8):
        super().__init__()
        self.save_hyperparameters()

        # Initialize GiG model from models.py
        self.model = GiG(config)
        self.lambda_rank = lambda_rank

    def forward(self, data):
        x, feature_matrix, edge_index, edge_weight, adj_matrix = self.model(data)
        return x

    def training_step(self, batch, batch_idx):
        out = self(batch)
        batch.y = batch.y.view(out.shape[0], -1)

        bce_loss = F.binary_cross_entropy_with_logits(out, batch.y.float())  # Use BCEWithLogitsLoss
        # Compute mean rank loss
        mean_rank_loss = self.calculate_mean_rank_loss(out, batch.y)

        # Combine losses
        total_loss = bce_loss + self.lambda_rank * mean_rank_loss

        # Log losses
        self.log('train_loss', bce_loss, batch_size=batch.x.shape[0], prog_bar=True)
        self.log('mean_rank_loss', mean_rank_loss, batch_size=batch.x.shape[0], prog_bar=True)
        self.log('total_loss', total_loss, batch_size=batch.x.shape[0], prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        batch.y = batch.y.view(out.shape[0], -1)  # Reshape to [32, 2405]
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(out, batch.y.float())
        self.log('val_loss', bce_loss, batch_size=batch.x.shape[0], prog_bar=True)

        # Compute mean rank loss for logging
        mean_rank = self.calculate_mean_rank_loss(out, batch.y)
        self.log('val_mean_rank', mean_rank, batch_size=batch.x.shape[0], prog_bar=True)

        return {"val_loss": bce_loss, "val_mean_rank": mean_rank}

    def test_step(self, batch, batch_idx):
        out = self(batch)
        batch.y = batch.y.view(out.shape[0], -1)
        bce_loss = F.binary_cross_entropy_with_logits(out, batch.y.float())
        self.log('test_loss', bce_loss, batch_size=batch.x.shape[0], prog_bar=True)

        # Compute mean rank for evaluation
        mean_rank = self.calculate_mean_rank_loss(out, batch.y)
        self.log('test_mean_rank', mean_rank, batch_size=batch.x.shape[0], prog_bar=True)

        return {"test_loss": bce_loss, "test_mean_rank": mean_rank}

    def calculate_mean_rank_loss(self, predictions, labels):
        """
        Calculates a differentiable approximation of the mean rank loss.
        """
        batch_size, num_classes = predictions.size()
        loss = 0.0

        for i in range(batch_size):
            scores = predictions[i]
            true_label_idx = labels[i].nonzero(as_tuple=True)[0].item()

            # Use a softmax approximation for ranking
            soft_ranks = torch.argsort(torch.argsort(-scores)) + 1
            rank_loss = soft_ranks[true_label_idx]
            loss += rank_loss

        return loss / batch_size  # Average rank loss across the batch

    def calculate_mean_rank(self, predictions, labels):
        """
        Computes the mean rank of the true labels in the prediction scores.

        Args:
            predictions (torch.Tensor): Shape [batch_size, num_classes], predicted scores.
            labels (torch.Tensor): Shape [batch_size, num_classes], one-hot true labels.

        Returns:
            float: Mean rank of the true labels.
        """
        batch_size, num_classes = predictions.size()
        ranks = []

        for i in range(batch_size):
            # Get the scores and true label index
            scores = predictions[i]
            true_label_idx = labels[i].nonzero(as_tuple=True)[0].item()

            # Rank the scores in descending order and find the rank of the true label
            sorted_indices = torch.argsort(scores, descending=True)
            true_label_rank = (sorted_indices == true_label_idx).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(true_label_rank)

        mean_rank = sum(ranks) / len(ranks)
        return mean_rank

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.config['lr'])


# Define Sweep Configuration
sweep_config = {
    'method': 'random',  # Random search
    'metric': {'name': 'val_mean_rank', 'goal': 'minimize'},
    'parameters': {
        'lr': {'values': [1e-3, 1e-4, 1e-5]},
        'batch_size': {'values': [16, 32, 64]},
        'gnn_layers': {'values': [[8], [16], [32, 16]]},
        'gnn_type': {'values': ['GraphConv', 'GAT', 'EdgeConv', 'SAGEConv']},
        'classifier_layers': {'values': [[8], [16, 8]]},
        'population_level_module': {'values': ['LGL', 'LGLKL']},
        'pooling': {'values': ['mean', 'add']},
        'gnn_aggr': {'values': ['mean', 'max']}
    }
}


def train(config=None):
    try:
        with wandb.init(config=config):
            config = wandb.config
            dataset_files = ["./Graph Outputs/train_pg_subgraph.pkl", "./Graph Outputs/val_pg_subgraph.pkl",
                             "./Graph Outputs/test_pg_subgraph.pkl"]
            id_to_class = create_id_to_class_mapping(dataset_files)

            data_module = PGSubgraphDataModule(
                train_file="./Graph Outputs/train_pg_subgraph.pkl",
                val_file="./Graph Outputs/val_pg_subgraph.pkl",
                test_file="./Graph Outputs/test_pg_subgraph.pkl",
                id_to_class=id_to_class,
                batch_size=config.batch_size
            )

            model_config = {
                "node_level_module": "GIN",
                "num_node_features": 7,
                "node_level_hidden_layers_number": len([32, 16]),
                "node_layers": [32, 16],
                "pooling": config.pooling,
                "population_level_module": "LGL",
                "population_layers": [16, 8],
                "temp": 0.5,
                "theta": 0.1,
                "gnn_type": "GraphConv",
                "gnn_layers": config.gnn_layers,
                "gnn_aggr": config.gnn_aggr,
                "classifier_layers": config.classifier_layers,
                "output_dim": 2405,
                "lr": config.lr
            }

            model = GiGModel(model_config)
            wandb.finish()
            wandb_logger = WandbLogger(project="gig_rare_diseases")

            trainer = Trainer(
                max_epochs=15,
                callbacks=[
                    ModelCheckpoint(monitor="val_mean_rank", save_top_k=1, mode="min"),
                    EarlyStopping(monitor="val_mean_rank", patience=5, mode="min")
                ],
                accelerator="gpu",
                devices=1,
                logger=wandb_logger
            )

            trainer.fit(model, data_module)
            trainer.test(model, data_module)

    except Exception as e:
        print(f"Error occurred: {e}")
        wandb.finish(exit_code=1)  # Mark the run as failed


# Initialize and Run the Sweep

sweep_id = wandb.sweep(sweep_config, project="gig_rare_diseases")
wandb.agent(sweep_id, function=train)
wandb.finish()
