import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Tuple, Optional, Set, Dict, List, Union, Optional, Dict, Any

import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import networkx as nx

class KnowledgeGraph:

    def __init__(self):
        """Initialize the Knowledge Graph"""
        self.G = nx.Graph()
        self.node_types = defaultdict(int)
        self.edge_types = defaultdict(int)
        
        # Define fixed color mapping - base colors that should always be used
        self.BASE_COLORS = {
            'gene/protein': '#FF4B4B',       # Red
            'anatomy': '#4286f4',            # Blue
            'disease': '#2ecc71',            # Green
            'effect/phenotype': '#9b59b6',   # Purple
            'pathway': '#e67e22',            # Orange
            'biological_process': '#1abc9c',  # Turquoise
            'cellular_component': '#3498db',  # Light Blue
            'molecular_function': '#f1c40f',  # Yellow
            'cell': '#95a5a6',               # Gray
        }
        
        # Initialize type_colors with base colors
        self.type_colors = self.BASE_COLORS.copy()
        
        # Add mappings for faster lookups
        self.type_to_nodes = defaultdict(set)
        self.relation_to_edges = defaultdict(set)
        self.name_to_nodes = defaultdict(set)
        self._all_seen_types = set()

    def _update_type_colors(self, new_types: set):
        """
        Update color mappings for node types, ensuring distinct colors.
        
        Args:
            new_types (set): Set of new node types to potentially add colors for
        """
        # Add new types to the set of all seen types
        self._all_seen_types.update(new_types)
        
        # Find types that don't have colors yet
        types_needing_colors = new_types - set(self.type_colors.keys())
        if types_needing_colors:
            # First try to use base colors if available
            for node_type in types_needing_colors:
                if node_type in self.BASE_COLORS:
                    self.type_colors[node_type] = self.BASE_COLORS[node_type]
            
            # For remaining types, assign from additional colors
            remaining_types = types_needing_colors - set(self.BASE_COLORS.keys())
            if remaining_types:
                additional_colors = [
                    '#d35400', '#c0392b', '#8e44ad', '#2980b9', '#27ae60',
                    '#16a085', '#7f8c8d', '#34495e', '#2c3e50', '#e74c3c'
                ]
                
                for i, node_type in enumerate(sorted(remaining_types)):
                    color_idx = i % len(additional_colors)
                    self.type_colors[node_type] = additional_colors[color_idx]


    
    def visualize_subgraph(self, 
                        nodes_of_interest: List[str], 
                        true_gene_ids: Optional[List[str]] = None,
                        all_paths: bool = False,
                        show_labels: bool = True, 
                        patient_id: Optional[str] = None,
                        show_types: bool = True,
                        figsize: Tuple[int, int] = (15, 10),
                        existing_subgraph: Optional[nx.Graph] = None) -> nx.Graph:
        """
        Visualizes a subgraph around nodes of interest with optional true genes and patient node.
        
        Args:
            nodes_of_interest: List of node IDs to highlight with red circles.
            true_gene_ids: Optional list of true gene IDs to highlight with yellow circles.
            all_paths: Whether to include all paths between nodes.
            show_labels: Whether to show node labels.
            patient_id: Optional patient ID to add as a node.
            show_types: Whether to color nodes by type.
            figsize: Figure size as (width, height).
            existing_subgraph: Optional existing subgraph to visualize instead of creating one.
        """
        try:
            plt.figure(figsize=figsize)
            
            # Use the existing subgraph if provided, otherwise create a new one
            if existing_subgraph is not None:
                subgraph = existing_subgraph
            else:
                subgraph = self.create_subgraph(
                    nodes_of_interest=nodes_of_interest,
                    true_gene_ids=true_gene_ids,
                    all_paths=all_paths,
                    patient_id=patient_id
                )
            
            # Get unique node types and update colors if needed
            node_types_dict = nx.get_node_attributes(subgraph, 'type')
            unique_node_types = set(node_types_dict.values())
            self._update_type_colors(unique_node_types)
            
            # Generate layout
            pos = nx.spring_layout(subgraph, k=1.5, iterations=50)
        
            # Draw edges with different colors for patient edges
            regular_edges = [(u, v) for u, v, d in subgraph.edges(data=True) 
                            if not d.get('is_patient_edge', False)]
            patient_edges = [(u, v) for u, v, d in subgraph.edges(data=True) 
                            if d.get('is_patient_edge', False)]
            
            # Draw regular edges
            nx.draw_networkx_edges(subgraph, pos, 
                                edgelist=regular_edges,
                                alpha=0.3, 
                                edge_color='gray', 
                                width=1)
            
            # Draw patient edges with different style
            if patient_edges:
                nx.draw_networkx_edges(subgraph, pos, 
                                    edgelist=patient_edges,
                                    alpha=0.5, 
                                    edge_color='blue', 
                                    width=1.5,
                                    style='dashed')
        
            # Convert sets for efficient lookup
            nodes_of_interest_set = set(nodes_of_interest)
            true_genes_set = set(true_gene_ids) if true_gene_ids else set()
            
            if show_types:
                # Draw all nodes with their type colors
                for node_type in unique_node_types:
                    # Filter out patient node if it exists
                    nodes = [n for n, t in node_types_dict.items() 
                            if t == node_type and not (patient_id and n == patient_id)]
                    if nodes:
                        color = self.type_colors.get(node_type, '#666666')
                        nx.draw_networkx_nodes(subgraph, pos, 
                                            nodelist=nodes,
                                            node_color=color,
                                            node_size=500, 
                                            label=node_type)
                
                # Draw patient node if it exists
                if patient_id is not None and patient_id in subgraph:
                    nx.draw_networkx_nodes(subgraph, pos,
                                        nodelist=[patient_id],
                                        node_color='lightblue',
                                        node_size=700,
                                        label='Patient')
                
                # Add red circles around nodes of interest
                if nodes_of_interest_set:
                    nx.draw_networkx_nodes(subgraph, pos,
                                        nodelist=list(nodes_of_interest_set),
                                        node_color='none',
                                        node_size=700,
                                        edgecolors='red',
                                        linewidths=2)
                
                # Add yellow circles around true genes
                if true_genes_set:
                    nx.draw_networkx_nodes(subgraph, pos,
                                        nodelist=list(true_genes_set),
                                        node_color='none',
                                        node_size=700,
                                        edgecolors='yellow',
                                        linewidths=2)
            
                # Create legend with node types and highlighted nodes
                legend_elements = []
                
                # Add node type colors to legend
                for node_type in unique_node_types:
                    if node_type != 'patient':  # Handle patient node separately
                        color = self.type_colors.get(node_type, '#666666')
                        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                        markerfacecolor=color,
                                                        markersize=10,
                                                        label=node_type))
            
                # Add patient node to legend if present
                if patient_id and patient_id in subgraph:
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                    markerfacecolor='lightblue',
                                                    markersize=10,
                                                    label='Patient'))
                
                # Add highlighted nodes to legend
                if nodes_of_interest_set:
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                    markerfacecolor='none',
                                                    markeredgecolor='red',
                                                    markersize=10,
                                                    label='Nodes of Interest'))
                if true_genes_set:
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                    markerfacecolor='none',
                                                    markeredgecolor='yellow',
                                                    markersize=10,
                                                    label='True Genes'))
                
                # Add edge type to legend
                legend_elements.append(plt.Line2D([0], [0], color='gray', alpha=0.3,
                                                label='Regular Edge'))
                if patient_edges:
                    legend_elements.append(plt.Line2D([0], [0], color='blue', alpha=0.5,
                                                    linestyle='--',
                                                    label='Patient Edge'))
                
                plt.legend(handles=legend_elements,
                        bbox_to_anchor=(1.02, 1), 
                        loc='upper left',
                        title="Node and Edge Types",
                        fontsize='small',
                        title_fontsize='small')
        
            if show_labels:
                labels = {n: subgraph.nodes[n].get('name', n) for n in subgraph.nodes()}
                nx.draw_networkx_labels(subgraph, pos, labels, 
                                        font_size=8,
                                        bbox=dict(facecolor='white', 
                                                alpha=0.7,
                                                edgecolor='none',
                                                pad=0.5))
            
            plt.title(f"Subgraph with {len(nodes_of_interest)} highlighted nodes"
                    f"{' and ' + str(len(true_genes_set)) + ' true genes' if true_genes_set else ''}\n"
                    f"{len(subgraph.nodes())} total nodes, {len(subgraph.edges())} edges")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            return subgraph
            
        except Exception as e:
            print(f"Error plotting subgraph: {str(e)}")
            if 'subgraph' in locals():
                return subgraph
            raise

    
    
    def visualize_subgraph9(self, 
                        nodes_of_interest: List[str], 
                        true_gene_ids: Optional[List[str]] = None,
                        all_paths: bool = False,
                        show_labels: bool = True, 
                        patient_id: Optional[str] = None,
                        show_types: bool = True,
                        figsize: Tuple[int, int] = (15, 10)) -> nx.Graph:
        """
        Creates and visualizes a subgraph around nodes of interest with optional true genes and patient node.
        
        Args:
            nodes_of_interest: List of node IDs to highlight with red circles
            true_gene_ids: Optional list of true gene IDs to highlight with yellow circles
            all_paths: Whether to include all paths between nodes
            show_labels: Whether to show node labels
            patient_id: Optional patient ID to add as a node
            show_types: Whether to color nodes by type
            figsize: Figure size as (width, height)
        """
        try:
            plt.figure(figsize=figsize)
            
            # Create subgraph with all specified nodes
            subgraph = self.create_subgraph(
                nodes_of_interest=nodes_of_interest,
                true_gene_ids=true_gene_ids,
                all_paths=all_paths,
                patient_id=patient_id
            )
            
            # Get unique node types and update colors if needed
            node_types_dict = nx.get_node_attributes(subgraph, 'type')
            unique_node_types = set(node_types_dict.values())
            self._update_type_colors(unique_node_types)
            
            # Generate layout
            pos = nx.spring_layout(subgraph, k=1.5, iterations=50)
        
            # Draw edges with different colors for patient edges
            regular_edges = [(u, v) for u, v, d in subgraph.edges(data=True) 
                            if not d.get('is_patient_edge', False)]
            patient_edges = [(u, v) for u, v, d in subgraph.edges(data=True) 
                            if d.get('is_patient_edge', False)]
            
            # Draw regular edges
            nx.draw_networkx_edges(subgraph, pos, 
                                edgelist=regular_edges,
                                alpha=0.3, 
                                edge_color='gray', 
                                width=1)
            
            # Draw patient edges with different style
            if patient_edges:
                nx.draw_networkx_edges(subgraph, pos, 
                                    edgelist=patient_edges,
                                    alpha=0.5, 
                                    edge_color='blue', 
                                    width=1.5,
                                    style='dashed')
        
            # Convert sets for efficient lookup
            nodes_of_interest_set = set(nodes_of_interest)
            true_genes_set = set(true_gene_ids) if true_gene_ids else set()
            
            if show_types:
                # Draw all nodes with their type colors
                for node_type in unique_node_types:
                    # Filter out patient node if it exists
                    nodes = [n for n, t in node_types_dict.items() 
                            if t == node_type and not (patient_id and n == patient_id)]
                    if nodes:
                        color = self.type_colors.get(node_type, '#666666')
                        nx.draw_networkx_nodes(subgraph, pos, 
                                            nodelist=nodes,
                                            node_color=color,
                                            node_size=500, 
                                            label=node_type)
            
                # Draw patient node if it exists
                if patient_id is not None and patient_id in subgraph:
                    nx.draw_networkx_nodes(subgraph, pos,
                                        nodelist=[patient_id],
                                        node_color='lightblue',
                                        node_size=700,
                                        label='Patient')
                
                # Add red circles around nodes of interest
                if nodes_of_interest_set:
                    nx.draw_networkx_nodes(subgraph, pos,
                                        nodelist=list(nodes_of_interest_set),
                                        node_color='none',
                                        node_size=700,
                                        edgecolors='red',
                                        linewidths=2)
                
                # Add yellow circles around true genes
                if true_genes_set:
                    nx.draw_networkx_nodes(subgraph, pos,
                                        nodelist=list(true_genes_set),
                                        node_color='none',
                                        node_size=700,
                                        edgecolors='yellow',
                                        linewidths=2)


                # Create legend with both node types and highlighted nodes
                legend_elements = []
                
                # Add node type colors to legend
                for node_type in unique_node_types:
                    if node_type != 'patient':  # Handle patient node separately
                        color = self.type_colors.get(node_type, '#666666')
                        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                        markerfacecolor=color,
                                                        markersize=10,
                                                        label=node_type))
            
                # Add patient node to legend if present
                if patient_id and patient_id in subgraph:
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                    markerfacecolor='lightblue',
                                                    markersize=10,
                                                    label='Patient'))
                
                # Add highlighted nodes to legend
                if nodes_of_interest_set:
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                    markerfacecolor='none',
                                                    markeredgecolor='red',
                                                    markersize=10,
                                                    label='Nodes of Interest'))
                if true_genes_set:
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                    markerfacecolor='none',
                                                    markeredgecolor='yellow',
                                                    markersize=10,
                                                    label='True Genes'))
                
                # Add edge type to legend
                legend_elements.append(plt.Line2D([0], [0], color='gray', alpha=0.3,
                                                label='Regular Edge'))
                if patient_edges:
                    legend_elements.append(plt.Line2D([0], [0], color='blue', alpha=0.5,
                                                    linestyle='--',
                                                    label='Patient Edge'))
                
                plt.legend(handles=legend_elements,
                        bbox_to_anchor=(1.02, 1), 
                        loc='upper left',
                        title="Node and Edge Types",
                        fontsize='small',
                        title_fontsize='small')
        
            if show_labels:
                labels = {n: subgraph.nodes[n].get('name', n) for n in subgraph.nodes()}
                nx.draw_networkx_labels(subgraph, pos, labels, 
                                    font_size=8,
                                    bbox=dict(facecolor='white', 
                                            alpha=0.7,
                                            edgecolor='none',
                                            pad=0.5))
            
            plt.title(f"Subgraph with {len(nodes_of_interest)} highlighted nodes"
                    f"{' and ' + str(len(true_genes_set)) + ' true genes' if true_genes_set else ''}\n"
                    f"{len(subgraph.nodes())} total nodes, {len(subgraph.edges())} edges")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            return subgraph
            
        except Exception as e:
            print(f"Error plotting subgraph: {str(e)}")
            print(f"Node types in subgraph: {unique_node_types}")
            print(f"Available colors: {list(self.type_colors.keys())}")
            if 'subgraph' in locals():
                return subgraph
            raise

    
    
    def create_from_df(self, df: pd.DataFrame, nodes_df: pd.DataFrame = None, is_chunk: bool = False) -> None:
        """
        Creates knowledge graph from DataFrame with support for chunked loading and additional nodes.
        
        Args:
            df (pd.DataFrame): DataFrame containing relationship data
            nodes_df (pd.DataFrame): Optional DataFrame containing additional nodes data with columns:
                                [node_idx, node_id, node_type, node_name, node_source]
            is_chunk (bool): Whether this DataFrame is part of a chunked load
        """
        required_columns = {
            'x_idx', 'y_idx', 'x_type', 'y_type', 'x_name', 'y_name',
            'x_source', 'y_source', 'relation', 'display_relation'
        }
        
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Reset mappings if not a chunk
        if not is_chunk:
            self.node_types.clear()
            self.edge_types.clear()
            self.type_colors.clear()
            self.type_to_nodes.clear()
            self.relation_to_edges.clear()
            self.name_to_nodes.clear()
            self._all_seen_types.clear()
        
        # Collect all unique types from this chunk
        unique_types = set(df['x_type'].unique()) | set(df['y_type'].unique())
        
        # If nodes_df is provided, add its types to unique_types
        if nodes_df is not None:
            unique_types.update(nodes_df['node_type'].unique())
        
        # Update color mappings
        self._update_type_colors(unique_types)

        # First add all nodes from nodes_df if provided
        if nodes_df is not None:
            for _, row in nodes_df.iterrows():
                node_attrs = {
                    'type': row['node_type'],
                    'name': row['node_name'],
                    'source': row['node_source']
                }
                
                # Add node with attributes
                self.G.add_node(str(row['node_idx']), **node_attrs)
                
                # Update mappings
                self._update_mappings(str(row['node_idx']), node_attrs)
                
                # Track node types
                self.node_types[node_attrs['type']] += 1
        
        # Process relationships from df
        for _, row in df.iterrows():
            # Extract node identifiers
            x_idx = str(row['x_idx'])
            y_idx = str(row['y_idx'])
            
            # Create attribute dictionaries for x and y nodes
            x_attrs = {
                'type': row['x_type'],
                'name': row['x_name'],
                'source': row['x_source']
            }
            
            y_attrs = {
                'type': row['y_type'],
                'name': row['y_name'],
                'source': row['y_source']
            }
            
            # Add nodes with attributes if they don't already exist
            if not self.G.has_node(x_idx):
                self.G.add_node(x_idx, **x_attrs)
                self._update_mappings(x_idx, x_attrs)
                self.node_types[x_attrs['type']] += 1
                
            if not self.G.has_node(y_idx):
                self.G.add_node(y_idx, **y_attrs)
                self._update_mappings(y_idx, y_attrs)
                self.node_types[y_attrs['type']] += 1
            
            # Create edge attributes
            edge_attrs = {
                'relation': row['relation'],
                'display_relation': row['display_relation']
            }
            
            # Add edge with attributes
            self.G.add_edge(x_idx, y_idx, **edge_attrs)
            
            # Track edge types and update mappings
            self.edge_types[edge_attrs['relation']] += 1
            self.relation_to_edges[edge_attrs['relation']].add((x_idx, y_idx))

    def create_from_csv(self, 
                    filepath: str, 
                    nodes_filepath: Optional[str] = None,
                    chunksize: int = 1000, 
                    show_progress: bool = True) -> None:
        """
        Creates knowledge graph from CSV file and optionally loads nodes from a pickle file.
        
        Args:
            filepath (str): Path to the relationships CSV file
            nodes_filepath (str, optional): Path to the nodes pickle file
            chunksize (int): Number of rows to process in each chunk
            show_progress (bool): Whether to show progress information
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Load nodes from pickle if provided
        nodes_df = None
        if nodes_filepath:
            nodes_path = Path(nodes_filepath)
            if not nodes_path.exists():
                raise FileNotFoundError(f"Nodes file not found: {nodes_filepath}")
            
            if show_progress:
                print(f"Loading nodes from {nodes_filepath}")
            try:
                with open(nodes_filepath, 'rb') as file:
                    nodes_df = pickle.load(file)
                if show_progress:
                    print(f"Loaded {len(nodes_df):,} nodes")
                    print(f"Node columns: {nodes_df.columns.tolist()}")
            except Exception as e:
                print(f"Error loading nodes pickle file: {str(e)}")
                raise
        
        try:
            total_chunks = 0
            if show_progress:
                print(f"Loading relationships from {filepath}")
            
            # Process relationships file in chunks
            for chunk_num, chunk in enumerate(pd.read_csv(filepath, chunksize=chunksize)):
                # Pass nodes_df only on first chunk
                current_nodes_df = nodes_df if chunk_num == 0 else None
                self.create_from_df(chunk, nodes_df=current_nodes_df, is_chunk=(chunk_num > 0))
                total_chunks += 1
                
                if show_progress and chunk_num % 10 == 0:
                    print(f"Processed {chunk_num + 1} chunks ({(chunk_num + 1) * chunksize:,} rows)")
                    print(f"Current graph size: {self.G.number_of_nodes():,} nodes, "
                        f"{self.G.number_of_edges():,} edges")
                    print(f"Number of node types: {len(self._all_seen_types)}")
                    
                # Optional: break after first chunk for testing
                # if chunk_num == 0:
                #     break
            
            if show_progress:
                print(f"\nFinished loading {total_chunks * chunksize:,} relationship rows")
                if nodes_df is not None:
                    print(f"Total nodes from pickle file: {len(nodes_df):,}")
                print(f"Total unique node types: {len(self._all_seen_types)}")
                self.print_summary(detailed=True)
                
        except Exception as e:
            print(f"Error loading relationships CSV: {str(e)}")
            raise

   
    def _update_mappings(self, node_id: str, attrs: dict):
        """Update internal mappings when adding nodes"""
        if 'type' in attrs:
            self.type_to_nodes[attrs['type']].add(node_id)
        if 'name' in attrs:
            self.name_to_nodes[attrs['name']].add(node_id)

    def get_nodes_by_type(self, node_type: str) -> Set[str]:
        """Get all nodes of a specific type"""
        return self.type_to_nodes[node_type].copy()

    def get_edges_by_relation(self, relation: str) -> Set[Tuple[str, str]]:
        """Get all edges of a specific relation type"""
        return self.relation_to_edges[relation].copy()

    def get_nodes_by_name(self, name: str) -> Set[str]:
        """Get all nodes with a specific name"""
        return self.name_to_nodes[name].copy()

    def get_node_info(self, node_id: str) -> Dict:
        """Get all information about a specific node."""
        return dict(self.G.nodes[node_id])

    def get_edge_info(self, node1: str, node2: str) -> Dict:
        """Get all information about a specific edge."""
        return dict(self.G.edges[node1, node2])

    def get_node_neighbors_by_relation(self, node_id: str, relation: str) -> Set[str]:
        """Get neighbors of a node connected by a specific relation type"""
        neighbors = set()
        for neighbor in self.G.neighbors(node_id):
            if self.G[node_id][neighbor].get('relation') == relation:
                neighbors.add(neighbor)
        return neighbors

    def create_type_subgraph(self, node_types: List[str], include_relations: Optional[List[str]] = None) -> nx.Graph:
        """Create a subgraph containing only specified node types and optionally filtered relations"""
        nodes = set()
        for ntype in node_types:
            nodes.update(self.get_nodes_by_type(ntype))
            
        subgraph = self.G.subgraph(nodes).copy()
        
        if include_relations:
            edges_to_remove = [
                (u, v) for u, v, attrs in subgraph.edges(data=True)
                if attrs.get('relation') not in include_relations
            ]
            subgraph.remove_edges_from(edges_to_remove)
            
        return subgraph

    def analyze_type_connections(self) -> pd.DataFrame:
        """Analyze connections between different node types"""
        type_connections = defaultdict(int)
        
        for u, v, attrs in self.G.edges(data=True):
            u_type = self.G.nodes[u]['type']
            v_type = self.G.nodes[v]['type']
            relation = attrs.get('relation', 'unknown')
            
            key = (u_type, v_type, relation)
            type_connections[key] += 1
            
        connections = []
        for (src_type, dst_type, relation), count in type_connections.items():
            connections.append({
                'source_type': src_type,
                'target_type': dst_type,
                'relation': relation,
                'count': count
            })
            
        return pd.DataFrame(connections)
    def get_hop_subgraph(self, node_id: str, hops: int = 1) -> nx.Graph:
        """
        Generate a subgraph containing nodes within a specified number of hops from the given node.

        Args:
            node_id (str): The starting node ID.
            hops (int): The number of hops to include (1, 2, or 3).

        Returns:
            nx.Graph: A subgraph containing nodes within the specified hops.
        """
        if node_id not in self.G:
            raise ValueError(f"Node ID {node_id} does not exist in the graph.")

        if hops < 1:
            raise ValueError("Hops must be a positive integer.")

        # Get nodes within the specified number of hops
        nodes_within_hops = nx.single_source_shortest_path_length(self.G, node_id, cutoff=hops).keys()

        # Create a subgraph
        subgraph = self.G.subgraph(nodes_within_hops).copy()

                
        subgraph.graph['original_nodes'] = [str(node_id)]
    
        return subgraph
    
    def combine_subgraphs_with_edges(self, subgraphs: List[nx.Graph]) -> nx.Graph:
        """
        Combine multiple subgraphs and add edges between overlapping nodes 
        if those edges exist in the original graph. Aggregate 'original_nodes'
        metadata from all subgraphs.

        Args:
            subgraphs (List[nx.Graph]): List of subgraphs to combine.

        Returns:
            nx.Graph: The combined graph with additional edges and aggregated 'original_nodes'.
        """
        if not subgraphs:
            raise ValueError("No subgraphs provided for combination.")
        
        # Initialize an empty graph for the combined result
        combined_graph = nx.Graph()
        combined_original_nodes = []

        # Merge all subgraphs into the combined graph
        for subgraph in subgraphs:
            combined_graph = nx.compose(combined_graph, subgraph)
            
            # Collect 'original_nodes' from each subgraph
            if 'original_nodes' in subgraph.graph:
                original_nodes = subgraph.graph['original_nodes']
                if isinstance(original_nodes, list):
                    combined_original_nodes.extend(original_nodes)
                else:
                    combined_original_nodes.append(original_nodes)

        # Add missing edges between overlapping nodes if they exist in the original graph
        for node1 in combined_graph.nodes:
            for node2 in combined_graph.nodes:
                if node1 != node2 and self.G.has_edge(node1, node2):
                    if not combined_graph.has_edge(node1, node2):
                        combined_graph.add_edge(node1, node2, **self.G[node1][node2])

        # Add combined original nodes to the combined graph metadata
        combined_graph.graph['original_nodes'] = list(set(combined_original_nodes))

        return combined_graph

    
    def create_subgraph(self, 
                    nodes_of_interest: List[str], 
                    true_gene_ids: Optional[List[str]] = None, 
                    all_paths: bool = False,
                    patient_id: Optional[str] = None) -> nx.Graph:
        """
        Creates subgraph using either minimal or all paths, optionally including true gene IDs
        and a patient node connected to nodes of interest of various types.
        """
        # Convert patient_id to string if provided
        if patient_id is not None:
            patient_id = str(patient_id)
        
        # If true gene IDs provided, add them to nodes of interest
        if true_gene_ids:
            nodes_to_include = set(nodes_of_interest + true_gene_ids)
        else:
            nodes_to_include = set(nodes_of_interest)
        
        # Create paths and initial subgraph
        if all_paths:
            for n1, n2 in combinations(nodes_of_interest, 2):
                try:
                    paths = nx.all_shortest_paths(self.G, n1, n2)
                    for path in paths:
                        nodes_to_include.update(path)
                except nx.NetworkXNoPath:
                    continue
        else:
            for n2 in nodes_of_interest[1:]:
                try:
                    path = nx.shortest_path(self.G, nodes_of_interest[0], n2)
                    nodes_to_include.update(path)
                except nx.NetworkXNoPath:
                    continue
        
        # Create the initial subgraph
        subgraph = self.G.subgraph(nodes_to_include).copy()
        
        # Add true gene IDs as graph attribute if provided
        if true_gene_ids:
            subgraph.graph['true_gene_ids'] = true_gene_ids
            subgraph.graph['original_nodes'] = nodes_of_interest # nodes that was in originl list during subgraph creation 
        
        # Add patient node if patient_id is provided
        if patient_id is not None:
            # Create patient node with attributes
            patient_node_attrs = {
                'type': 'patient',
                'name': f'Patient_{patient_id}',
                'source': 'patient_data',
                'is_patient': True
            }
        
            # Update type mappings for patient node type if needed
            if 'patient' not in self.node_type_to_idx:
                unique_node_types = sorted(self._all_seen_types | {'patient'})
                self.node_type_to_idx = {ntype: idx for idx, ntype in enumerate(unique_node_types)}
                self.idx_to_node_type = {idx: ntype for ntype, idx in self.node_type_to_idx.items()}
                self.node_types['patient'] = 0
                self._all_seen_types.add('patient')
            
            # Add the patient node to the subgraph
            subgraph.add_node(patient_id, **patient_node_attrs)
            
            # Define relation types based on node types
            relation_types = {
                'effect/phenotype': ('has_phenotype', 'Has_Phenotype'),
                'disease': ('has_disease', 'Has_Disease'),
                'gene/protein': ('has_gene_mutation', 'Has_Gene_Mutation'),
                'anatomy': ('has_anatomical_feature', 'Has_Anatomical_Feature'),
                'biological_process': ('involved_in_process', 'Involved_In_Process'),
                'pathway': ('involved_in_pathway', 'Involved_In_Pathway'),
                'cell': ('has_cell_involvement', 'Has_Cell_Involvement'),
                'molecular_function': ('has_molecular_feature', 'Has_Molecular_Feature')
            }
            
            # Pre-register all possible patient edge types, including fallback type
            all_relations = {relation for _, (relation, _) in relation_types.items()}
            all_relations.add('has_association')  # Add fallback type
            
            for relation in all_relations:
                if relation not in self.edge_type_to_idx:
                    # Update edge type mappings
                    unique_edge_types = sorted(set(self.edge_types.keys()) | {relation})
                    self.edge_type_to_idx = {etype: idx for idx, etype in enumerate(unique_edge_types)}
                    self.idx_to_edge_type = {idx: etype for etype, idx in self.edge_type_to_idx.items()}
                    # Initialize count in edge_types
                    self.edge_types[relation] = 0
            
            # Connect patient node to all original nodes of interest
            for node in nodes_of_interest:
                if node in subgraph:
                    node_type = subgraph.nodes[node].get('type', 'unknown')
                    relation, display = relation_types.get(
                        node_type, 
                        ('has_association', 'Has_Association')  # Fallback type
                    )
                    
                    edge_attrs = {
                        'relation': relation,
                        'display_relation': display,
                        'is_patient_edge': True,
                        'connected_node_type': node_type
                    }
                    subgraph.add_edge(patient_id, node, **edge_attrs)
                    
                    # Update edge type count
                    self.edge_types[relation] += 1
            
            # Add patient node info to graph attributes
            subgraph.graph['patient_id'] = patient_id
            subgraph.graph['patient_connections'] = {
                node: subgraph.nodes[node].get('type', 'unknown')
                for node in nodes_of_interest
                if node in subgraph
            }
            
            # Update node type count
            self.node_types['patient'] += 1
        
        return subgraph

    def random_walk_sample(self, num_nodes: int, start_node: Optional[str] = None) -> Set[str]:
        """Sample nodes using random walk."""
        if start_node is None:
            start_node = random.choice(list(self.G.nodes()))
            
        sampled_nodes = {start_node}
        current_node = start_node
        
        while len(sampled_nodes) < num_nodes:
            neighbors = list(self.G.neighbors(current_node))
            if not neighbors:
                current_node = random.choice(list(sampled_nodes))
                continue
                
            current_node = random.choice(neighbors)
            sampled_nodes.add(current_node)
            
        return sampled_nodes

    def snowball_sample(self, num_nodes: int, start_node: Optional[str] = None) -> Set[str]:
        """Sample nodes using snowball sampling."""
        if start_node is None:
            start_node = random.choice(list(self.G.nodes()))
            
        sampled_nodes = {start_node}
        frontier = {start_node}
        
        while len(sampled_nodes) < num_nodes and frontier:
            current_node = random.choice(list(frontier))
            frontier.remove(current_node)
            
            neighbors = set(self.G.neighbors(current_node)) - sampled_nodes
            frontier.update(neighbors)
            sampled_nodes.update(neighbors)
            
            if len(sampled_nodes) >= num_nodes:
                return set(list(sampled_nodes)[:num_nodes])
                
        return sampled_nodes

    def edge_sample(self, num_edges: int) -> Set[str]:
        """Sample by randomly selecting edges and including their nodes."""
        edges = random.sample(list(self.G.edges()), min(num_edges, self.G.number_of_edges()))
        nodes = set()
        for edge in edges:
            nodes.update(edge)
        return nodes
    def plot_random_subgraph(self, 
                        num_nodes: int = 20,
                        method: str = 'random_walk',
                        show_labels: bool = True,
                        show_types: bool = True,
                        title: Optional[str] = None,
                        figsize: Tuple[int, int] = (12, 8)) -> nx.Graph:
        """Sample and visualize a random subgraph."""
        try:
            # Set the style
            plt.style.use('default')
            
            # Sample nodes based on method
            if method == 'random_walk':
                sampled_nodes = self.random_walk_sample(num_nodes)
            elif method == 'snowball':
                sampled_nodes = self.snowball_sample(num_nodes)
            elif method == 'edge':
                sampled_nodes = self.edge_sample(num_nodes)
            else:
                raise ValueError("Method must be 'random_walk', 'snowball', or 'edge'")
            
            # Create subgraph
            subgraph = self.G.subgraph(sampled_nodes).copy()
            
            # Get unique node types from the subgraph
            node_types = set(nx.get_node_attributes(subgraph, 'type').values())
            
            # Generate color map if needed
            if not self.type_colors or not all(t in self.type_colors for t in node_types):
                # Create a new color palette for all types
                colors = sns.color_palette("husl", n_colors=len(node_types))
                self.type_colors = dict(zip(sorted(node_types), colors))
            
            # Create figure
            plt.figure(figsize=figsize)
            
            # Generate layout
            try:
                pos = nx.spring_layout(subgraph, k=1.5, iterations=50)
            except Exception:
                pos = nx.kamada_kawai_layout(subgraph)
            
            # Draw edges
            nx.draw_networkx_edges(subgraph, pos, alpha=0.3, edge_color='gray', width=1)
            
            if show_types:
                # Get node types
                node_type_dict = nx.get_node_attributes(subgraph, 'type')
                if not node_type_dict:  # If no types found
                    nx.draw_networkx_nodes(subgraph, pos, 
                                        node_color='lightblue',
                                        node_size=500)
                else:
                    # Draw nodes by type
                    for node_type in sorted(set(node_type_dict.values())):
                        nodes = [node for node, ntype in node_type_dict.items() 
                                if ntype == node_type]
                        if nodes:
                            nx.draw_networkx_nodes(subgraph, pos, 
                                                nodelist=nodes,
                                                node_color=[self.type_colors[node_type]],
                                                node_size=500, 
                                                label=node_type)
                    
                    # Add legend with better formatting
                    plt.legend(bbox_to_anchor=(1.02, 1), 
                            loc='upper left', 
                            fontsize='small',
                            title="Node Types",
                            title_fontsize='small')
            else:
                nx.draw_networkx_nodes(subgraph, pos, 
                                    node_color='lightblue',
                                    node_size=500)
            
            if show_labels:
                # Use names for labels if available, otherwise use node IDs
                labels = nx.get_node_attributes(subgraph, 'name')
                if not labels:
                    labels = {n: str(n) for n in subgraph.nodes()}
                
                nx.draw_networkx_labels(subgraph, pos, 
                                    labels, 
                                    font_size=8,
                                    bbox=dict(facecolor='white', 
                                            alpha=0.7,
                                            edgecolor='none',
                                            pad=0.5))
            
            if title is None:
                title = f"Random Subgraph ({method} sampling)\n{subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges"
            plt.title(title)
            
            plt.axis('off')
            plt.margins(x=0.2)
            plt.tight_layout()
            
            # Show the plot
            plt.show()
            
            return subgraph
            
        except Exception as e:
            print(f"Error plotting graph: {str(e)}")
            # Still return the subgraph even if visualization fails
            if 'subgraph' in locals():
                return subgraph
            raise

    def plot_sampling_comparison(self, num_nodes: int = 15, show_labels: bool = True, 
                               show_types: bool = True, figsize: Tuple[int, int] = (20, 6)):
        """Compare different sampling methods side by side."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        plt.suptitle("Comparison of Sampling Methods")
        
        methods = ['random_walk', 'snowball', 'edge']
        for ax, method in zip(axes, methods):
            plt.sca(ax)
            self.plot_random_subgraph(
                num_nodes=num_nodes,
                method=method,
                show_labels=show_labels,
                show_types=show_types,
                title=f"{method} sampling"
            )
        
        plt.tight_layout()
        plt.show()

    def analyze(self) -> Dict:
        """Performs comprehensive graph analysis."""
        stats = {
            'basic': {
                'num_nodes': self.G.number_of_nodes(),
                'num_edges': self.G.number_of_edges(),
                'density': nx.density(self.G),
                'avg_degree': sum(dict(self.G.degree()).values()) / self.G.number_of_nodes(),
                'num_components': nx.number_connected_components(self.G)
            },
            'node_type_distribution': dict(self.node_types),
            'edge_type_distribution': dict(self.edge_types),
            'degree_stats': {
                'min_degree': min(dict(self.G.degree()).values()),
                'max_degree': max(dict(self.G.degree()).values()),
                'median_degree': np.median(list(dict(self.G.degree()).values()))
            }
        }
        
        # Add component size distribution
        component_sizes = [len(c) for c in nx.connected_components(self.G)]
        stats['components'] = {
            'num_components': len(component_sizes),
            'largest_component_size': max(component_sizes),
            'smallest_component_size': min(component_sizes),
            'avg_component_size': np.mean(component_sizes)
        }
        
        return stats

    def print_summary(self, detailed: bool = False):
        """Prints summary of the knowledge graph."""
        stats = self.analyze()
        
        print("\n=== Knowledge Graph Summary ===")
        print(f"Nodes: {stats['basic']['num_nodes']:,}")
        print(f"Edges: {stats['basic']['num_edges']:,}")
        print(f"Density: {stats['basic']['density']:.6f}")
        print(f"Average Degree: {stats['basic']['avg_degree']:.2f}")
        print(f"Connected Components: {stats['components']['num_components']:,}")
        
        if detailed:
            print("\nNode Types Distribution:")
            for ntype, count in sorted(stats['node_type_distribution'].items()):
                print(f"  {ntype}: {count:,}")
            
            print("\nEdge Types Distribution:")
            for etype, count in sorted(stats['edge_type_distribution'].items()):
                print(f"  {etype}: {count:,}")
            
            print("\nComponent Statistics:")
            print(f"  Largest Component: {stats['components']['largest_component_size']:,} nodes")
            print(f"  Average Component: {stats['components']['avg_component_size']:.2f} nodes")
            
            print("\nDegree Statistics:")
            print(f"  Min Degree: {stats['degree_stats']['min_degree']}")
            print(f"  Max Degree: {stats['degree_stats']['max_degree']}")
            print(f"  Median Degree: {stats['degree_stats']['median_degree']}")
            
            print("\nType Connections Summary:")
            type_connections = self.analyze_type_connections()
            for _, row in type_connections.iterrows():
                print(f"  {row['source_type']} -> {row['target_type']} "
                      f"({row['relation']}): {row['count']:,}")

    
    def save_graph(self, filepath: str) -> None:
        """Save the knowledge graph with all its attributes to disk."""
        save_data = {
            'graph_data': {
                'nodes': dict(self.G.nodes(data=True)),
                'edges': [(u, v, d) for u, v, d in self.G.edges(data=True)]
            },
            'mappings': {
                'node_types': dict(self.node_types),
                'edge_types': dict(self.edge_types),
                'type_colors': dict(self.type_colors),
                'base_colors': dict(self.BASE_COLORS),  # Save base colors too
                'type_to_nodes': {k: list(v) for k, v in self.type_to_nodes.items()},
                'relation_to_edges': {k: list(v) for k, v in self.relation_to_edges.items()},
                'name_to_nodes': {k: list(v) for k, v in self.name_to_nodes.items()},
                'all_seen_types': list(self._all_seen_types)
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

    def load_graph(self, filepath: str) -> None:
        """Load a knowledge graph from disk."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Clear and recreate graph
        self.G = nx.Graph()
        
        # Restore graph structure
        for node, attrs in save_data['graph_data']['nodes'].items():
            self.G.add_node(node, **attrs)
        
        for u, v, attrs in save_data['graph_data']['edges']:
            self.G.add_edge(u, v, **attrs)
        
        # Restore base colors and type colors
        if 'base_colors' in save_data['mappings']:
            self.BASE_COLORS = dict(save_data['mappings']['base_colors'])
        
        # Restore type colors with base colors as fallback
        self.type_colors = dict(save_data['mappings']['type_colors'])
        
        # Restore other mappings
        self.node_types = defaultdict(int, save_data['mappings']['node_types'])
        self.edge_types = defaultdict(int, save_data['mappings']['edge_types'])
        
        # Restore set-based mappings
        self.type_to_nodes = defaultdict(set)
        for k, v in save_data['mappings']['type_to_nodes'].items():
            self.type_to_nodes[k] = set(v)
        
        self.relation_to_edges = defaultdict(set)
        for k, v in save_data['mappings']['relation_to_edges'].items():
            self.relation_to_edges[k] = set(tuple(edge) for edge in v)
        
        self.name_to_nodes = defaultdict(set)
        for k, v in save_data['mappings']['name_to_nodes'].items():
            self.name_to_nodes[k] = set(v)
        
        # Restore seen types
        self._all_seen_types = set(save_data['mappings']['all_seen_types'])
        
        # Ensure all node types have colors
        self._update_type_colors(self._all_seen_types)
        # Ensure type mappings are created
        if not hasattr(self, 'node_type_to_idx'):
            self._create_type_to_index_mappings()

    def _create_type_to_index_mappings(self):
        """Create mappings between node/edge types and their indices"""
        # Create node type mapping
        unique_node_types = sorted(self._all_seen_types)
        self.node_type_to_idx = {ntype: idx for idx, ntype in enumerate(unique_node_types)}
        self.idx_to_node_type = {idx: ntype for ntype, idx in self.node_type_to_idx.items()}
        
        # Create edge type mapping
        unique_edge_types = sorted(self.edge_types.keys())
        self.edge_type_to_idx = {etype: idx for idx, etype in enumerate(unique_edge_types)}
        self.idx_to_edge_type = {idx: etype for etype, idx in self.edge_type_to_idx.items()}


    def create_pyg_data_from_subgraph(self, subgraph: nx.Graph, node_embeddings: Optional[torch.Tensor] = None) -> Data:
        """
        Convert subgraph to PyG Data with one-hot encoded types and preserved graph attributes.
        Ensures all IDs are converted to integers during creation.
        
        Args:
            subgraph: NetworkX subgraph to convert
            node_embeddings: Optional tensor of node embeddings where index corresponds to original node id
            
        Returns:
            PyG Data object with one-hot encodings and preserved graph attributes
        """
        # Ensure type mappings are created
        if not hasattr(self, 'node_type_to_idx'):
            self._create_type_to_index_mappings()
        
        def safe_int_convert(value):
            """Safely convert string or int to int"""
            try:
                return int(value)
            except (ValueError, TypeError):
                # If conversion fails, hash the string to get a consistent integer
                return hash(str(value)) % (2**31)
        
        # Create node mappings and features
        node_mapping = {}
        node_types = []
        original_ids = []
        node_names = []
        raw_node_types = []
        
        # Pre-convert all node IDs to integers
        node_id_mapping = {node: safe_int_convert(node) for node in subgraph.nodes()}
        
        for idx, (node, attrs) in enumerate(subgraph.nodes(data=True)):
            node_mapping[node] = idx
            node_type_idx = self.node_type_to_idx[attrs['type']]
            node_types.append(node_type_idx)
            original_ids.append(node_id_mapping[node])
            node_names.append(attrs.get('name', str(node)))
            raw_node_types.append(attrs['type'])
        
        # Convert to tensor once
        node_types_tensor = torch.tensor(node_types, dtype=torch.long)
        
        # Create one-hot node features efficiently
        num_node_types = len(self.node_type_to_idx)
        x_onehot = torch.zeros((len(node_mapping), num_node_types), dtype=torch.float)
        x_onehot.scatter_(1, node_types_tensor.unsqueeze(1), 1)
        
        # Combine with embeddings if provided
        if node_embeddings is not None:
            x_emb = node_embeddings[torch.tensor(original_ids, dtype=torch.long)]
            x = torch.cat([x_onehot, x_emb], dim=1)
        else:
            x = x_onehot
        
        # Process edges more efficiently
        edge_index_list = []
        edge_types = []
        
        # Pre-allocate if we know the graph is undirected
        expected_edges = len(subgraph.edges()) * (2 if not subgraph.is_directed() else 1)
        edge_index_list = torch.zeros((expected_edges, 2), dtype=torch.long)
        edge_types = torch.zeros(expected_edges, dtype=torch.long)
        
        # Fill edge data
        idx = 0
        for u, v, attrs in subgraph.edges(data=True):
            src_idx = node_mapping[u]
            dst_idx = node_mapping[v]
            edge_type_idx = self.edge_type_to_idx[attrs['relation']]
            
            edge_index_list[idx] = torch.tensor([src_idx, dst_idx])
            edge_types[idx] = edge_type_idx
            idx += 1
            
            # Add reverse edge if graph is undirected
            if not subgraph.is_directed():
                edge_index_list[idx] = torch.tensor([dst_idx, src_idx])
                edge_types[idx] = edge_type_idx
                idx += 1
        
        # Create edge features efficiently
        num_edge_types = len(self.edge_type_to_idx)
        edge_attr = torch.zeros((idx, num_edge_types), dtype=torch.float)
        edge_attr.scatter_(1, edge_types[:idx].unsqueeze(1), 1)
        
        # Convert true gene IDs to integers
        true_gene_ids = [safe_int_convert(gene_id) for gene_id in subgraph.graph.get('true_gene_ids', [])]
        original_nodes = [safe_int_convert(node) for node in subgraph.graph.get('original_nodes', [])]
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index_list[:idx].t().contiguous(),
            edge_attr=edge_attr,
            
            # Store metadata
            node_mapping=node_mapping,
            node_names=node_names,
            original_ids=original_ids,  # Now guaranteed to be integers
            node_type_mapping=self.node_type_to_idx,
            edge_type_mapping=self.edge_type_to_idx,
            node_types=raw_node_types,
            
            # Preserve graph attributes with integer IDs
            true_gene_ids=true_gene_ids,  # Now guaranteed to be integers
            original_nodes=original_nodes,  # Now guaranteed to be integers
            
            # Store feature dimensions
            num_node_types=num_node_types,
            embedding_dim=node_embeddings.shape[1] if node_embeddings is not None else 0
        )
        
        return data