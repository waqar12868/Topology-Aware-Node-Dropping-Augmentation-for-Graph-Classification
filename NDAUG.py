import torch
import argparse
import networkx as nx
import torch.nn.functional as F
parser = argparse.ArgumentParser()
from torch_geometric.utils import to_networkx
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import degree

def identify_motif_nodes(data, cycle_lengths):
    G = pyg_to_networkx(data)
    motif_nodes = torch.zeros(data.num_nodes, dtype=torch.bool)

    for length in cycle_lengths:
        cycles = nx.cycle_basis(G)
        for cycle in cycles:
            if len(cycle) == length:
                for node in cycle:
                    motif_nodes[node] = True

    return motif_nodes

def pyg_to_networkx(data):
    G = to_networkx(data, to_undirected=True)
    return G

def Augmentation_NDAUG(data, drop_prob=0.20):
    # Define cycle lengths to preserve
    cycle_lengths = [4, 5, 6]  # Include 4, 5, and 6-node cycles

    motif_nodes = identify_motif_nodes(data, cycle_lengths)
    num_nodes = data.num_nodes
    deg = degree(data.edge_index[0], num_nodes=num_nodes)
    # Assign degree as node feature
    data.x = deg.view(-1, 1).float()
    # Calculate drop probability for each node, considering motif preservation
    drop_probs = torch.rand(len(deg))
    nodes_to_drop = (deg <= 1) & (drop_probs < drop_prob) & ~motif_nodes

    # Convert boolean tensor to tensor of indices
    nodes_to_drop_tensor = nodes_to_drop.nonzero(as_tuple=True)[0]

    # Drop nodes in node features if they exist
    if data.x is not None:
        data.x = data.x[~nodes_to_drop, :]

    # Drop edges that are connected to the nodes being dropped
    edge_mask_0 = ~torch.isin(data.edge_index[0], nodes_to_drop_tensor)
    edge_mask_1 = ~torch.isin(data.edge_index[1], nodes_to_drop_tensor)
    edge_mask = edge_mask_0 & edge_mask_1
    data.edge_index = data.edge_index[:, edge_mask]

    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr[edge_mask]

    # Reindex the nodes
    remaining_nodes = torch.arange(num_nodes)[~nodes_to_drop]
    node_idx_map = {old.item(): new for new, old in enumerate(remaining_nodes)}

    # Update edge indices with the new node indices
    new_edge_index = torch.tensor([[node_idx_map[i.item()] for i in data.edge_index[0]],
                                   [node_idx_map[i.item()] for i in data.edge_index[1]]], dtype=torch.long)
    data.edge_index = new_edge_index

    # Update the number of nodes in the data object
    data.num_nodes = len(remaining_nodes)

    main_graph_mask = ~torch.isin(torch.arange(data.num_nodes), isolated_nodes)
    main_graph_indices = main_graph_mask.nonzero(as_tuple=True)[0]

    # Initialize GAT-based similarity model
    in_channels = data.num_node_features
    out_channels = 64  # Example output size, adjust as needed
    gat_similarity = GATBasedSimilarity(in_channels, out_channels)

    # Apply GAT and get the transformed node features
    gat_similarity.eval()
    with torch.no_grad():
        gat_output = gat_similarity(data.x, data.edge_index)

    # Compute similarities based on GAT output (e.g., using cosine similarity)
    similarity_scores = compute_similarity(gat_output)

    new_edges_list = []
    for isolated_node in isolated_nodes:
        if main_graph_indices.size(0) > 0:  # Ensure there are nodes in the main graph to connect to
            scores = similarity_scores[isolated_node.item(), main_graph_indices]
            best_match = scores.argmax()
            best_match_main_graph = main_graph_indices[best_match]
            new_edges_list.append([isolated_node.item(), best_match_main_graph.item()])
            new_edges_list.append([best_match_main_graph.item(), isolated_node.item()])  # adding reverse direction for undirected graph

    if new_edges_list:
        data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)

    return data
