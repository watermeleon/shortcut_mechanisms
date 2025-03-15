import torch 
import plotly.graph_objects as go
from typing import Optional, List

from graphviz import Digraph

class Node:
    def __init__(
        self,
        layer_id: Optional[int] = None,
        head_id: Optional[int] = None,
        token_idx: Optional[int] = None,
        token_idx_looksat_id: Optional[int] = None
    ):
        self.layer_id = layer_id
        self.head_id = head_id
        self.token_idx = token_idx
        self.token_idx_looksat_id = token_idx_looksat_id
        
        # Tree structure attributes
        self.children: List[Node] = []
        self.parent: Optional[Node] = None
    
    def add_child(self, child_node: 'Node') -> None:
        """Add a child node to the current node."""
        child_node.parent = self
        self.children.append(child_node)
        
    def is_leaf(self) -> bool:
        """Check if the current node is a leaf node."""
        return len(self.children) == 0
    
    def get_all_nodes(self) -> List['Node']:
        """Get all nodes in the tree starting from this node."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.get_all_nodes())
        return nodes

    def get_input_tokens_mask(self, sequence_length: int) -> List[int]:
        """
        Create a binary mask over input tokens that were used in the attention path.
        
        Args:
            sequence_length: The total length of the input sequence
            
        Returns:
            List[int]: Binary mask where 1 indicates the token was used in the attention path
        """
        # Initialize mask with zeros
        # mask = [0] * sequence_length
        # use a torch.Tensor
        mask = torch.zeros(sequence_length, dtype=torch.int).bool()
        
        # Get all nodes in the tree
        all_nodes = self.get_all_nodes()
        
        for node in all_nodes:
            # Mark tokens that are being looked at
            if node.token_idx_looksat_id is not None and node.token_idx_looksat_id >= 0:
                mask[node.token_idx_looksat_id] = 1
                
            # Mark source tokens
            if node.token_idx is not None and node.token_idx >= 0:
                mask[node.token_idx] = 1
                
        return mask


def visualize_tree(root: Node) -> None:
    """
    Visualize the tree structure using graphviz.
    Shows layer_id and token_idx for each node.
    Returns a graphviz object that automatically displays in Jupyter.
    """
    dot = Digraph(comment='Tree Visualization')
    dot.attr(rankdir='TB')  # Top to bottom direction
    
    # Create a unique identifier for each node
    def get_node_id(node: Node, prefix: str = "node") -> str:
        # Use object id to ensure uniqueness
        return f"{prefix}_{id(node)}"
    
    # Add all nodes to the graph
    def add_nodes_edges(node: Node):
        node_id = get_node_id(node)
        
        # Create label with layer_id and token_idx
        label = f"Layer: {node.layer_id}\nToken: {node.token_idx}"
        dot.node(node_id, label)
        
        # Add edges to children
        for child in node.children:
            child_id = get_node_id(child)
            add_nodes_edges(child)
            dot.edge(node_id, child_id)
    
    # Build the graph
    add_nodes_edges(root)
    
    return dot


    

def visualize_tree_grid(root: Node, token_strings: Optional[List[str]] = None) -> go.Figure:
    """
    Visualize the tree structure in a grid layout showing layers vs token positions.
    
    Parameters:
    root: Root node of the tree
    token_strings: Optional list of strings corresponding to tokens
    
    Returns:
    plotly.graph_objects.Figure
    """
    # Collect all nodes and their information
    nodes = root.get_all_nodes()
    
    # Extract coordinates and information
    layers = []
    token_idxs = []
    head_ids = []
    edges_x = []
    edges_y = []
    
    # First pass: collect node information
    for node in nodes:
        if node.layer_id is not None and node.token_idx is not None:
            layers.append(node.layer_id)
            token_idxs.append(node.token_idx)
            head_ids.append(str(node.head_id) if node.head_id is not None else "")
    
    # Second pass: collect edge information
    for node in nodes:
        for child in node.children:
            if (node.layer_id is not None and node.token_idx is not None and 
                child.layer_id is not None and child.token_idx is not None):
                # Add the line connecting the points
                edges_x.extend([node.token_idx, child.token_idx, None])
                edges_y.extend([node.layer_id, child.layer_id, None])
    
    # Create figure
    fig = go.Figure()
    
    # Add edges (lines connecting nodes)
    if edges_x and edges_y:
        fig.add_trace(go.Scatter(
            x=edges_x,
            y=edges_y,
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False
        ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=token_idxs,
        y=layers,
        mode='markers+text',
        text=head_ids,
        textposition="middle right",
        marker=dict(
            size=15,
            color='blue',
            symbol='circle'
        ),
        name='Attention heads'
    ))
    
    # Calculate axis ranges
    y_min = min(layers) - 0.5 if layers else 0
    y_max = max(layers) + 0.5 if layers else 1
    x_min = 0
    x_max = max(max(token_idxs) if token_idxs else 0,
                len(token_strings)-1 if token_strings else 0) + 1
    
    # Update layout
    fig.update_layout(
        title="Transformer Information Flow",
        xaxis=dict(
            title="Token Position",
            tickmode='array',
            ticktext=token_strings if token_strings else list(range(x_max)),
            tickvals=list(range(x_max)),
            gridwidth=0.1,
            gridcolor='lightgray',
            range=[x_min - 0.5, x_max - 0.5]
        ),
        yaxis=dict(
            title="Layer",
            gridwidth=1,
            gridcolor='lightgray',
            range=[y_max, y_min],  # Reversed range for top-down layout
            dtick=1
        ),
        showlegend=True,
        width=max(800, len(token_strings) * 30 if token_strings else 800),
        height=600,
        template="simple_white",
        plot_bgcolor='white'
    )
    
    return fig