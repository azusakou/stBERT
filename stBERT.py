import torch
import torch.nn as nn
import torch.nn.init as init

from torch_geometric.nn import GCNConv, global_add_pool, TransformerConv, global_mean_pool, SGConv, GATConv
from torch_scatter import scatter_mean
from torch import Generator

class GraphEmbedding(nn.Module):
    def __init__(self, n_node, node_in_features, node_out_features):
        super(GraphEmbedding, self).__init__()
        self.edge_emb = False
        self.node_in_features = node_in_features

        # Node embeddings
        #self.node_conv = GCNConv(node_in_features, node_out_features)
        self.node_conv = SGConv(node_in_features, node_out_features)

        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(n_node, node_out_features))  # Assuming fixed number of nodes as 34

        if self.edge_emb:
            # Edge embeddings
            self.edge_embedding = nn.Linear(node_in_features, node_out_features)

    def forward(self, x, edge_index):
        if self.edge_emb:
            # Edge embeddings
            edge_attr = self.generate_edge_attributes(x, edge_index, edge_feature_dim=self.node_in_features)
            edge_embeddings = self.edge_embedding(edge_attr)
            row, col = edge_index
            agg_edge_features = scatter_mean(edge_embeddings, col, dim=0, dim_size=x.size(0))

            x = self.node_conv(x, edge_index)# node embeddings
            x = x + self.pos_embedding # Add positional embeddings to node features #[:x.size(0)]
            return x, agg_edge_features

        else:
            x = self.node_conv(x, edge_index)
            x = x + self.pos_embedding
            return x, None

    def generate_edge_attributes(self, node_features, edge_index, edge_feature_dim=10):
        # Example: edge attributes as the absolute difference between connected node features
        start_features = node_features[edge_index[0]]
        end_features = node_features[edge_index[1]]

        # Compute the absolute difference or any other meaningful relation
        edge_attr = torch.abs(start_features - end_features)

        # Optionally, reduce or expand dimensions to match the expected edge_feature_dim
        if edge_attr.shape[1] > edge_feature_dim:
            # Reduce dimension via a linear transformation or simple averaging
            edge_attr = torch.mean(edge_attr, dim=1, keepdim=True).expand(-1, edge_feature_dim)
        elif edge_attr.shape[1] < edge_feature_dim:
            # Expand dimension by repeating features
            edge_attr = edge_attr.repeat(1, edge_feature_dim // edge_attr.shape[1])
        return edge_attr

class GraphTransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.1):
        super(GraphTransformerLayer, self).__init__()
        complex=0
        if complex:
            self.attention = TransformerConv(in_channels, out_channels, heads=heads, concat=False, beta=True)
        else:
            self.attention = GATConv(in_channels, out_channels, heads=heads, concat=False, dropout=dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(out_channels, 2 * out_channels),
            nn.ReLU(),
            nn.Linear(2 * out_channels, out_channels)
        )
        self.layer_norm1 = nn.LayerNorm(out_channels)
        self.layer_norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # Multi-head graph attention layer
        attention_out = self.attention(x, edge_index)
        x = self.layer_norm1(attention_out + x)  # Add residual connection
        x = self.dropout(x)

        # Feed-forward layer
        ff_out = self.feed_forward(x)
        x = self.layer_norm2(ff_out + x)  # Add residual connection
        x = self.dropout(x)
        return x

class GraphTransformerEncoder(torch.nn.Module):
    def __init__(self, in_features, out_features, num_layers=1, heads=8, dropout=0.1):
        super(GraphTransformerEncoder, self).__init__()
        self.layers = torch.nn.ModuleList([
            GraphTransformerLayer(in_features if i == 0 else out_features, out_features, heads, dropout)
            for i in range(num_layers)
        ])

    def forward(self, x, edge_index):
        hiddens = [x]
        for layer in self.layers:
            x = layer(x, edge_index)
            hiddens.append(x)
        return x, hiddens

class GraphBERT(nn.Module):
    def __init__(self, config):
        super(GraphBERT, self).__init__()

        self.embedding = GraphEmbedding(n_node=config.n_nodes,
                                        node_in_features=config.feat_dim,
                                        node_out_features=config.bert_dim,
                                        )

        self.encoder = GraphTransformerEncoder(in_features=config.bert_dim,
                                               out_features=config.bert_dim,
                                               num_layers=config.bert_layers,
                                               heads=config.bert_heads, dropout=0.1)

        self.conv = SGConv(config.bert_dim, config.embed_dim)
        self.reconstruction_head = SGConv(config.embed_dim, config.feat_dim)  # To predict original features
        self.mlm_recon = self._mlm_recon

        self.max_position_embeddings = 512

        self.init_weights()
        self.all_hidden = True
        #self.forward = self._forward_chunk # stBERT
        self.forward =  self._forward_complete # ablation


    def _forward_complete(self, x, edge_index):
        node_embeddings, edge_embeddings = self.embedding(x, edge_index)  #node_embeddings += edge_embeddings
        node_embeddings, hiddens = self.encoder(node_embeddings, edge_index)

        if self.all_hidden:
           node_embeddings = torch.stack(hiddens).mean(dim=0)
        return self.conv(node_embeddings, edge_index)

    def _forward_chunk(self, x, edge_index):
        """
        Process embeddings in chunks to adapt to the BERT model's maximum position limitations,
        and handle edge conditions if required for node embeddings.

        Args:
            embeddings (torch.Tensor): Tensor of shape (total_length, embedding_dim) containing
                                       the embeddings for all sequences or nodes.
            edge_index (torch.Tensor): Tensor containing the indices of edges in the graph.

        Returns:
            torch.Tensor: Processed embeddings after passing through the BERT model and concatenation.
        """
        embeddings, edge_embeddings = self.embedding(x, edge_index)
        total_length = embeddings.size(0)
        all_outputs = []

        for i in range(0, total_length, self.max_position_embeddings):
            # Extract embeddings for the current chunk and add a batch dimension
            chunk_embeddings = embeddings[i:i + self.max_position_embeddings]#.unsqueeze(0)
            if chunk_embeddings.shape[0] < self.max_position_embeddings:
                padding_size = self.max_position_embeddings - chunk_embeddings.shape[0]
                pad = torch.nn.ConstantPad1d((0, 0, 0, padding_size), 0)  # padding last dim
                chunk_embeddings = pad(chunk_embeddings)

            max_pos = self.max_position_embeddings  # max number of nodes in a chunk

            # Edge filtering
            mask = (edge_index[0] >= i) & (edge_index[0] < i + max_pos) & (edge_index[1] >= i) & (edge_index[1] < i + max_pos)
            chunk_edges = edge_index[:, mask]

            # Adjust indices to local chunk coordinates
            chunk_edges[0], chunk_edges[1] = chunk_edges[0] - i, chunk_edges[1] - i

            node_embeddings, hiddens = self.encoder(chunk_embeddings, chunk_edges)
            processed_embeddings = torch.stack(hiddens).mean(dim=0) if self.all_hidden else node_embeddings

            all_outputs.append(processed_embeddings)

        return self.conv(torch.cat(all_outputs, dim=0)[:total_length], edge_index)

    def _mlm_recon(self, z, edge_index):
        return self.reconstruction_head(z, edge_index)

    def init_weights(self):
        # Initialize Graph Embedding and Transformer Layers
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

        # Initialize Linear Transformation Layer
        # init.xavier_normal_(self.embedding_transformation.weight)
        # init.zeros_(self.embedding_transformation.bias)

def mask_features(x, mask_rate=0.15, rand_e=0):
    gen = Generator()
    gen.manual_seed(rand_e)
    batch_size, num_features = x.size()
    mask = torch.rand(batch_size, num_features, generator=gen) < mask_rate
    masked_x = x.detach().clone()
    masked_x[mask] = 0  # Zero out the masked features
    return masked_x, mask

if __name__ == '__main__':
    from torch_geometric.datasets import KarateClub
    dataset = KarateClub()
    data = dataset[0]  # Load the Karate Club graph

    # Assign random features to the nodes for the sake of the example
    data.x = torch.randn(data.num_nodes, 108)  # 34 features
    data.batch = None  # For simplicity in this example

    # Model configuration
    class cfg:
        n_nodes=34
        feat_dim = 108
        edge_in_features =156
        embed_dim = 32
        num_layers = 3

    model = GraphBERT(cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.CrossEntropyLoss()
    out = model(data.x, data.edge_index); print(out.size())

