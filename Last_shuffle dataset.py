import pickle
import os
import random


def load_pickle(file_path):
    """Load a pickle file and return the data."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, file_path):
    """Save data to a pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def correct_dataset(train_path, val_path, test_path, output_dir):
    """Ensure all true gene IDs are in the training dataset."""

    # Load datasets
    train_graphs = load_pickle(train_path)
    val_graphs = load_pickle(val_path)
    test_graphs = load_pickle(test_path)

    # Extract true gene IDs from training set
    train_gene_ids = set()
    for graph in train_graphs:
        train_gene_ids.update(graph.true_gene_ids)


    missing_val_graphs = []
    remaining_val_graphs = []
    for graph in val_graphs:
        if any(gene_id not in train_gene_ids for gene_id in graph.true_gene_ids):
            missing_val_graphs.append(graph)
        else:
            remaining_val_graphs.append(graph)


    missing_test_graphs = []
    remaining_test_graphs = []
    for graph in test_graphs:
        if any(gene_id not in train_gene_ids for gene_id in graph.true_gene_ids):
            missing_test_graphs.append(graph)
        else:
            remaining_test_graphs.append(graph)

    # Move missing true gene graphs from validation and test to training
    if missing_val_graphs:
        print(f"Moving {len(missing_val_graphs)} graphs from validation to training set.")
        train_graphs.extend(missing_val_graphs)
        val_graphs = remaining_val_graphs

    if missing_test_graphs:
        print(f"Moving {len(missing_test_graphs)} graphs from test to training set.")
        train_graphs.extend(missing_test_graphs)
        test_graphs = remaining_test_graphs

    val_replacements = random.sample(train_graphs, len(missing_val_graphs))
    test_replacements = random.sample(train_graphs, len(missing_test_graphs))

    val_graphs.extend(val_replacements)
    test_graphs.extend(test_replacements)
    train_graphs = [graph for graph in train_graphs if
                    graph not in val_replacements and graph not in test_replacements]

    os.makedirs(output_dir, exist_ok=True)
    save_pickle(train_graphs, os.path.join(output_dir, 'train_corrected.pkl'))
    save_pickle(val_graphs, os.path.join(output_dir, 'val_corrected.pkl'))
    save_pickle(test_graphs, os.path.join(output_dir, 'test_corrected.pkl'))

    print("Datasets corrected and saved.")



train_file = './Graph Outputs/train_pg_subgraph.pkl'
val_file = './Graph Outputs/val_pg_subgraph.pkl'
test_file = './Graph Outputs/test_pg_subgraph.pkl'
output_directory = './DataLast/corrected_datasets'


correct_dataset(train_file, val_file, test_file, output_directory)
