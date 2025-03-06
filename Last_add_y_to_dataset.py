import pickle


def create_id_to_class_mapping(dataset_files):
    unique_node_ids = set()
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



dataset_files = ["./DataLast/train_corrected.pkl", "./DataLast/val_corrected.pkl",
                 "./DataLast/test_corrected.pkl"]


def assign_y_to_datasets(dataset_files):
    """
    Assigns class labels (`y` attribute) to graphs in the dataset based on gene_id mapping.

    Returns:
        List of processed datasets with y labels.
    """
    id_to_class = create_id_to_class_mapping(dataset_files)  # Get gene_id → class mapping
    processed_datasets = []

    for dataset_file in dataset_files:
        with open(dataset_file, 'rb') as f:
            graphs = pickle.load(f)

        for graph in graphs:
            if hasattr(graph, 'true_gene_ids') and graph.true_gene_ids:
                # Assign valid gene_id's class
                graph.y = id_to_class[graph.true_gene_ids[0]]
            else:
                graph.y = -1

        processed_datasets.append(graphs)

    return processed_datasets


dataset_files = ["./Graph Outputs/train_pg_subgraph.pkl", "./Graph Outputs/val_pg_subgraph.pkl",
                 "./Graph Outputs/test_pg_subgraph.pkl"]
processed_datasets = assign_y_to_datasets(dataset_files)


for i, dataset in enumerate(processed_datasets):
    with open(f"processed_dataset_{i}.pkl", "wb") as f:
        pickle.dump(dataset, f)