"""
This file inclues the modules of the Vanilla MHNfs and the Label Cleaning MHNfs.

Some parts are taken and adapted from "Autoregressive activity prediction for low-data drug discovery" 
URL https://github.com/ml-jku/autoregressive_activity_prediction
"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from functools import partial
import math
from torch.nn import functional as F
import random
import numpy as np

current_loc = __file__.rsplit("/",3)[0]
import sys
import os
sys.path.append(current_loc)

from src.mhnfs.hopfield.modules import Hopfield
from src.mhnfs.initialization import init_weights


# Mappings
activation_function_mapping = {
    "relu": nn.ReLU(),
    "selu": nn.SELU(),
    "sigmoid": nn.Sigmoid(),
}

dropout_mapping = {"relu": nn.Dropout, "selu": nn.AlphaDropout}


# Modules
class EncoderBlock(nn.Module):
    """
    Fully connected molecule encoder block.
    - Takes molecular descriptors, e.g., ECFPs and RDKit fps as inputs
    - returns a molecular representation
    """

    def __init__(self, cfg: OmegaConf):
        super(EncoderBlock, self).__init__()

        # Input layer
        self.dropout = dropout_mapping[cfg.model.encoder.activation](
            cfg.model.encoder.regularization.input_dropout
        )
        self.fc = nn.Linear(
            cfg.model.encoder.input_dim, cfg.model.encoder.number_hidden_neurons
        )
        self.act = activation_function_mapping[cfg.model.encoder.activation]

        # Hidden layer
        self.hidden_linear_layers = nn.ModuleList([])
        self.hidden_dropout_layers = nn.ModuleList([])
        self.hidden_activations = nn.ModuleList([])

        for _ in range(cfg.model.encoder.number_hidden_layers):
            self.hidden_dropout_layers.append(
                dropout_mapping[cfg.model.encoder.activation](
                    cfg.model.encoder.regularization.dropout
                )
            )
            self.hidden_linear_layers.append(
                nn.Linear(
                    cfg.model.encoder.number_hidden_neurons,
                    cfg.model.encoder.number_hidden_neurons,
                )
            )
            self.hidden_activations.append(
                activation_function_mapping[cfg.model.encoder.activation]
            )

        # Output layer
        self.dropout_o = dropout_mapping[cfg.model.encoder.activation](
            cfg.model.encoder.regularization.dropout
        )
        self.fc_o = nn.Linear(
            cfg.model.encoder.number_hidden_neurons,
            cfg.model.associationSpace_dim,
        )
        self.act_o = activation_function_mapping[cfg.model.encoder.activation]

        # Initialization
        encoder_initialization = partial(init_weights, cfg.model.encoder.activation)
        self.apply(encoder_initialization)

    def forward(self, molecule_representation: torch.Tensor) -> torch.Tensor:
        # Input layer
        x = self.dropout(molecule_representation)
        x = self.fc(x)
        x = self.act(x)

        # Hidden layer
        for hidden_dropout, hidden_layer, hidden_activation_function in zip(
            self.hidden_dropout_layers,
            self.hidden_linear_layers,
            self.hidden_activations,
        ):
            x = hidden_dropout(x)
            x = hidden_layer(x)
            x = hidden_activation_function(x)

        # Output layer
        x = self.dropout_o(x)
        x = self.fc_o(x)
        x = self.act_o(x)

        return x


class ContextModule(nn.Module):
    """
    Allows for mutual information sharing.
    Enriches the query and support set embeddings with context by associating a query or
    support set molecule with the context set, i.e., large set of training molecules:
    - The context set can be seen as an external memory
    - For a given molecule embedding, a Modern Hopfield Network retrieves a representa-
      tion from the external memory

    Since we have to retrieve representations for all query and support set molecules we
    stack all embeddings together and perform a "batch-retrieval".
    """

    def __init__(self, cfg: OmegaConf):
        super(ContextModule, self).__init__()

        self.cfg = cfg

        self.hopfield = Hopfield(
            input_size=self.cfg.model.associationSpace_dim,
            hidden_size=cfg.model.hopfield.dim_QK,
            stored_pattern_size=self.cfg.model.associationSpace_dim,
            pattern_projection_size=self.cfg.model.associationSpace_dim,
            output_size=self.cfg.model.associationSpace_dim,
            num_heads=self.cfg.model.hopfield.heads,
            scaling=self.cfg.model.hopfield.beta,
            dropout=self.cfg.model.hopfield.dropout,
        )

        # Initialization
        hopfield_initialization = partial(init_weights, "linear")
        self.hopfield.apply(hopfield_initialization)

    def forward(
        self,
        query_embedding: torch.Tensor,
        support_actives_embedding: torch.Tensor,
        support_inactives_embedding: torch.Tensor,
        context_set_embedding: torch.Tensor,
    ) -> tuple:
        """
        inputs:
        - query; torch.tensor;
          dim: [batch-size, 1, initial-embedding-dimension]
            * e.g.: [512, 1, 1024]
            * initial-embedding-dimension: defined by molecule encoder block
        - active support set molecules; torch.tensor;
          dim: [batch-size, active-padding-dim, initial-embedding-dimension]
          * e.g.: [512, 9, 1024]
        - inactive support set molecules; torch.tensor;
          dim: [batch-size, inactive-padding-dim, initial-embedding-dimension]
          * e.g.: [512, 11, 1024]
        - context set molecules; torch.tensor;
          dim: [1, number-of-context-molecules, initial-embedding-dimension]
          * e.g.: [1, 512, 1024]

        return:
        tuple which includes the updated representations for query, active, and inactive
        support set molecules:
        (query, active support set molecules, inactive support set molecules)
        """
        # Stack embeddings together to perform a "batch-retrieval"
        s = torch.cat(
            (query_embedding, support_actives_embedding, support_inactives_embedding),
            dim=1,
        )
        s_flattend = s.reshape(1, s.shape[0] * s.shape[1], s.shape[2])

        # Retrieval
        s_h = self.hopfield((context_set_embedding, s_flattend, context_set_embedding))

        # Combine retrieval with skip connection
        s_updated = s_flattend + s_h
        s_updated_inputShape = s_updated.reshape(
            s.shape[0], s.shape[1], s.shape[2]
        )  # reshape tensor back to input shape

        query_embedding = s_updated_inputShape[:, 0, :]
        query_embedding = torch.unsqueeze(query_embedding, 1)

        # Split query, active and inactive support set embeddings
        padding_size_actives = support_actives_embedding.shape[1]

        support_actives_embedding = s_updated_inputShape[
            :, 1 : (padding_size_actives + 1), :
        ]
        support_inactives_embedding = s_updated_inputShape[
            :, (padding_size_actives + 1) :, :
        ]

        return query_embedding, support_actives_embedding, support_inactives_embedding


class LayerNormalizingBlock(nn.Module):
    """
    Layernorm-block which scales/transforms the representations for query, ac-
    tive, and inactive support set molecules.
    """

    def __init__(self, cfg: OmegaConf):
        super(LayerNormalizingBlock, self).__init__()

        self.cfg = cfg

        if cfg.model.layerNormBlock.usage:
            self.layernorm_query = nn.LayerNorm(
                cfg.model.associationSpace_dim,
                elementwise_affine=cfg.model.layerNormBlock.affine,
            )
            self.layernorm_support_actives = nn.LayerNorm(
                cfg.model.associationSpace_dim,
                elementwise_affine=cfg.model.layerNormBlock.affine,
            )
            self.layernorm_support_inactives = nn.LayerNorm(
                cfg.model.associationSpace_dim,
                elementwise_affine=cfg.model.layerNormBlock.affine,
            )

    def forward(
        self,
        query_embedding: torch.Tensor,
        support_actives_embedding: torch.Tensor,
        support_inactives_embedding: torch.Tensor,
    ) -> tuple:
        """
        inputs:
        - query; torch.tensor;
          dim: [batch-size, 1, embedding-dim]
            * e.g.: [512, 1, 1024]
        - active support set molecules; torch.tensor;
          dim: [batch-size, active-padding-dim, embedding-dim]
          * e.g.: [512, 9, 1024]
        - inactive support set molecules; torch.tensor;
          dim: [batch-size, inactive-padding-dim, initial-embedding-dim]
          * e.g.: [512, 11, 1024]

        return:
        tuple which includes the updated representations for query, active, and inactive
        support set molecules:
        (query, active support set molecules, inactive support set molecules)
        """

        # Layer normalization
        # Since the layernorm operations are optional the module just updates represen-
        # tations if the the referring option is set in the config.
        if self.cfg.model.layerNormBlock.usage:
            query_embedding = self.layernorm_query(query_embedding)
            support_actives_embedding = self.layernorm_support_actives(
                support_actives_embedding
            )
            if support_inactives_embedding is not None:
                support_inactives_embedding = self.layernorm_support_inactives(
                    support_inactives_embedding
                )
        return query_embedding, support_actives_embedding, support_inactives_embedding


class CrossAttentionModule(nn.Module):
    """
    The cross-attention module allows for information sharing between query and support
    set molecules.

    Altae-Tran et al. [1] showed that representations can be enriched by making the
    query molecule aware of the support set molecules and making the support set mole-
    cules aware of each other and of the query molecule. We enable information sharing
    with a transformer.

    Overview of the cross-attention module:
    1) The query and support set molecules are concatenated such that one joint matrix
       emerges which includes both query and support set molecules.
    2) The joint matrix is fed into a transformer
       - Self-attention enables information sharing between query and support set mole-
         cules

    [1] Altae-Tran, H., Ramsundar, B., Pappu, A. S., & Pande, V. (2017). Low data drug
        discovery with one-shot learning. ACS central science, 3(4), 283-293.
    """

    def __init__(self, cfg: OmegaConf):

        super(CrossAttentionModule, self).__init__()

        self.cfg = cfg

        cfg_gpt = self.GPTConfig()
        self.transformer_block = self.TranformerBlock(cfg_gpt)
        
        # Initialization
        encoder_initialization = partial(init_weights, 'relu')
        self.apply(encoder_initialization)

    def forward(
        self,
        query_embedding: torch.Tensor,
        support_actives_embedding: torch.Tensor,
        support_inactives_embedding: torch.Tensor,
    ) -> tuple:
        """
        inputs:
        - query; torch.tensor;
          dim: [batch-size, 1, embedding-dim]
            * e.g.: [512, 1, 1024]
        - active support set molecules; torch.tensor;
          dim: [batch-size, active-padding-dim, embedding-dim]
          * e.g.: [512, 9, 1024]
        - inactive support set molecules; torch.tensor;
          dim: [batch-size, inactive-padding-dim, initial-embedding-dim]
          * e.g.: [512, 11, 1024]
        - number of active molecules in support set; torch.tensor;
          dim: [batch-size]
        - number of inactive molecules in support set; torch.tensor;
          dim: [batch-size]

        return:
        tuple which includes the updated representations for query, active, and inactive
        support set molecules:
        (query, active support set molecules, inactive support set molecules)
        query_embedding, support_actives_embedding, support_inactives_embedding
        """

        # Embedding dim of query and support set molecules
        embedding_dim = support_actives_embedding.shape[2]

        # Add activity encoding to representations
        # Activity encoding:
        # - active: 1
        # - inactive: -1
        # - unknown (query): 0
        query_embedding = torch.cat(
            [
                query_embedding,
                torch.zeros_like(
                    query_embedding[
                        :, :, : self.cfg.model.transformer.activity_embedding_dim
                    ]
                ),
            ],
            dim=2,
        )

        support_actives_embedding = torch.cat(
            [
                support_actives_embedding,
                torch.ones_like(
                    support_actives_embedding[
                        :, :, : self.cfg.model.transformer.activity_embedding_dim
                    ]
                ),
            ],
            dim=2,
        )

        support_inactives_embedding = torch.cat(
            [
                support_inactives_embedding,
                (-1.0)
                * torch.ones_like(
                    support_inactives_embedding[
                        :, :, : self.cfg.model.transformer.activity_embedding_dim
                    ]
                ),
            ],
            dim=2,
        )

        # Concatenate query and support set molecules
        s = torch.cat(
            [query_embedding, support_actives_embedding, support_inactives_embedding],
            dim=1,
        )

        # Run transformer and update representations
        s_h = self.transformer_block(s)
        s_updated = s + s_h

        # Split representations into query, active, and inactive support set molecules
        query_embedding = s_updated[:, 0, :embedding_dim]
        query_embedding = torch.unsqueeze(query_embedding, 1)
        support_actives_embedding = s_updated[
            :, 1 : (support_actives_embedding.shape[1] + 1), :embedding_dim
        ]
        support_inactives_embedding = s_updated[
            :, (support_actives_embedding.shape[1] + 1) :, :embedding_dim
        ]

        return query_embedding, support_actives_embedding, support_inactives_embedding

    #-----------------------------------------------------------------------------------
    # Sub-modules
    class TranformerBlock(nn.Module):

        def __init__(self, config):
            super().__init__()
            self.ln_1 = self.LayerNorm(config.n_embd, bias=config.bias)
            self.attn = self.SelfAttention(config)
            self.ln_2 = self.LayerNorm(config.n_embd, bias=config.bias)
            self.mlp = self.MLP(config)

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x
        
        class LayerNorm(nn.Module):
            """
            LayerNorm but with an optional bias. PyTorch doesn't support simply
            bias=False
            """

            def __init__(self, ndim, bias):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(ndim))
                self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

            def forward(self, input):
                return F.layer_norm(input, self.weight.shape, self.weight, self.bias,
                                    1e-5)
        
        class SelfAttention(nn.Module):
            """
            Self Attention Block
            """

            def __init__(self, config):
                super().__init__()
                
                self.cfg = config
                
                # query, key, value projections
                self.q_proj = nn.Linear(config.n_embd, config.n_qk_proj*config.n_head,
                                        bias=config.bias)
                self.k_proj = nn.Linear(config.n_embd, config.n_qk_proj*config.n_head,
                                        bias=config.bias)
                self.v_proj = nn.Linear(config.n_embd, config.n_v_proj*config.n_head,
                                        bias=config.bias)
                
                # output projection
                self.c_proj = nn.Linear(config.n_v_proj*config.n_head, config.n_embd,
                                        bias=config.bias)
                
                # regularization
                self.attn_dropout = nn.Dropout(config.dropout)
                self.resid_dropout = nn.Dropout(config.dropout)
                self.n_head = config.n_head
                self.n_embd = config.n_embd
                self.dropout = config.dropout

            def forward(self, x):
                B, T, C = x.size() # batch size, sequence length, embedding dim (n_embd)
                
                # Calculate queries, keys, and values
                q = self.q_proj(x).view(B, T, self.n_head, self.cfg.n_qk_proj
                                        ).transpose(1, 2) # (B, nh, T, hs)
                k = self.k_proj(x).view(B, T, self.n_head, self.cfg.n_qk_proj
                                        ).transpose(1, 2) # (B, nh, T, hs)
                v = self.v_proj(x).view(B, T, self.n_head, self.cfg.n_v_proj
                                        ).transpose(1, 2) # (B, nh, T, hs)

                # Calculate self-attentions
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                
                # Activations
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
                
                # Re-assemble all head outputs side by side
                y = y.transpose(1, 2).contiguous().view(y.shape[0],
                                                        y.shape[2],
                                                        -1)

                # output projection
                y = self.resid_dropout(self.c_proj(y))
                return y

        class MLP(nn.Module):

            def __init__(self, config):
                super().__init__()
                self.c_fc    = nn.Linear(config.n_embd, 567, bias=config.bias)
                self.relu    = nn.ReLU()
                self.c_proj  = nn.Linear(567, config.n_embd, bias=config.bias)
                self.dropout = nn.Dropout(config.dropout)
            
            def forward(self, x):
                x = self.c_fc(x)
                x = self.relu(x)
                x = self.c_proj(x)
                x = self.dropout(x)
                return x
      
    class GPTConfig:
        n_head: int = 8
        n_embd: int = 1088
        dropout: float = 0.
        bias: bool = True
        n_qk_proj: int = 136
        n_v_proj: int = 136

    
def SimilarityModule(
    query_embedding: torch.Tensor,
    support_set_embeddings: torch.Tensor,
    support_set_size: torch.Tensor,
    cfg: OmegaConf,
) -> torch.Tensor:
    """
    The similarity module builds the activity prediction for the query molecule from a
    weighted sum over the support set labels. Pair-wise similarity values between query
    and support set molecules are used as weights for the weighted sum.

    Since the similarity module is applied twice within the MHNfs model - once for the
    active and once for the inactive support set molecules, the support_set_embeddings
    here mean ether active or inactive support set molecule embeddings.

    inputs:
    - query; torch.tensor;
      dim: [batch-size, 1, embedding-dimension]
        * e.g.: [512, 1, 1024]
    - support set molecules; torch.tensor;
      dim: [batch-size, padding-dim, embedding-dimension]
        * e.g.: [512, 9, 1024]
    - padding mask; torch.tensor; boolean
      dim: [batch-size, padding-dim] 
        * e.g.: [512, 9]
    - support set size; torch.tensor;
      dim: [batch-size]
        * e.g.: [512]
    """

    # Optional L2-norm
    if cfg.model.similarityModule.l2Norm:
        query_embedding_div = torch.unsqueeze(
            query_embedding.pow(2).sum(dim=2).sqrt(), 2
        )
        query_embedding_div[query_embedding_div == 0] = 1
        support_set_embeddings_div = torch.unsqueeze(
            support_set_embeddings.pow(2).sum(dim=2).sqrt(), 2
        )
        support_set_embeddings_div[support_set_embeddings_div == 0] = 1

        query_embedding = query_embedding / query_embedding_div
        support_set_embeddings = support_set_embeddings / support_set_embeddings_div

    # support_set_size (just dim batch_size with entries all number of support_set actives/inactives (starting with shots until max-1s in last iteration))
    # support_set_embeddings batch_size dim all equal entries, padding-dim = # support_set actives/inactives (starting with shots until max-1s in last iteration)
    # padding-dim = # of support_set actives/inactives

    # Compute similarity values
    similarities = query_embedding @ torch.transpose(support_set_embeddings, 1, 2)
    # dim:
    # [batch-size, 1, padding-dim] =
    # [batch-size, 1, emb-dim] x [batch-size, emb-dim, padding-dim]

    # Compute similarity values
    similarities[torch.isnan(similarities)] = 0.0
    similarity_sums = similarities.sum(
        dim=2
    )  # For every query molecule: Sum over support set molecules

    # Scaling
    if cfg.model.similarityModule.scaling == "1/N":
        stabilizer = torch.tensor(1e-8).float()
        similarity_sums = (
            1 / (2.0 * support_set_size.reshape(-1, 1) + stabilizer) * similarity_sums
        )
    if cfg.model.similarityModule.scaling == "1/sqrt(N)":
        stabilizer = torch.tensor(1e-8).float()
        similarity_sums = (
            1
            / (2.0 * torch.sqrt(support_set_size.reshape(-1, 1).float()) + stabilizer)
            * similarity_sums
        )

    return similarity_sums

def SimilarityModule_similarities(
    query_embedding: torch.Tensor,
    support_set_embeddings: torch.Tensor,
    support_set_size: torch.Tensor,
    cfg: OmegaConf,
) -> torch.Tensor:
    """
    The similarity module builds the activity prediction for the query molecule from a
    weighted sum over the support set labels. Pair-wise similarity values between query
    and support set molecules are used as weights for the weighted sum.
    
    Since the similarity module is applied twice within the MHNfs model - once for the
    active and once for the inactive support set molecules, the support_set_embeddings
    here mean ether active or inactive support set molecule embeddings.
    
    inputs:
    - query; torch.tensor;
      dim: [batch-size, 1, embedding-dimension]
        * e.g.: [512, 1, 1024]
    - support set molecules; torch.tensor;
      dim: [batch-size, padding-dim, embedding-dimension]
        * e.g.: [512, 9, 1024]
    - padding mask; torch.tensor; boolean
      dim: [batch-size, padding-dim] 
        * e.g.: [512, 9]
    - support set size; torch.tensor;
      dim: [batch-size]
        * e.g.: [512]
    """

    # Optional L2-norm
    if cfg.model.similarityModule.l2Norm:
        query_embedding_div = torch.unsqueeze(
            query_embedding.pow(2).sum(dim=2).sqrt(), 2
        )
        query_embedding_div[query_embedding_div == 0] = 1
        support_set_embeddings_div = torch.unsqueeze(
            support_set_embeddings.pow(2).sum(dim=2).sqrt(), 2
        )
        support_set_embeddings_div[support_set_embeddings_div == 0] = 1

        query_embedding = query_embedding / query_embedding_div
        support_set_embeddings = support_set_embeddings / support_set_embeddings_div

    # support_set_size (just dim batch_size with entries all number of support_set actives/inactives (starting with shots until max-1s in last iteration))
    # support_set_embeddings batch_size dim all equal entries, padding-dim = # support_set actives/inactives (starting with shots until max-1s in last iteration)
    # padding-dim = # of support_set actives/inactives

    # Compute similarity values
    similarities = query_embedding @ torch.transpose(support_set_embeddings, 1, 2)
    # dim:
    # [batch-size, 1, padding-dim] =
    # [batch-size, 1, emb-dim] x [batch-size, emb-dim, padding-dim]

    # Compute similarity values
    similarities[torch.isnan(similarities)] = 0.0
    similarities = torch.squeeze(similarities)

    return similarities

#### Modules for Label Cleaning MHNfs

def NearestNeighborGraphModule(
    query_embedding: torch.Tensor,
    support_set_embedding_active: torch.Tensor,
    support_set_embedding_inactive: torch.Tensor,
    k: int = None, 
    norm: str = 'l1',
    plus_method: str = 'moving'
) -> torch.Tensor:
    """
    inputs:
    - query_embedding; torch.tensor;
      dim: [batch-size, embedding-dimension]
        * e.g.: [54, 1024]
    - support_set_embedding_active; torch.tensor;
      dim: [padding-dim, embedding-dimension] (active support set samples in padding-dim)
        * e.g.: [5, 1024]
    - support_set_embedding_inactive; torch.tensor;
      dim: [padding-dim, embedding-dimension] (inactive support set samples in padding-dim)
        * e.g.: [5, 1024]
    - k; int;
        hyperparamter
    """
    V = torch.cat((support_set_embedding_active, support_set_embedding_inactive, query_embedding), 0)

    # Apply L1-norm
    if norm == 'l1':
        p = 1
        l1_norms = V.abs().sum(dim=1, keepdim=True)
        V = V / l1_norms
    elif norm == 'l2':
        p = 2
        l2_norms = V.pow(2).sum(dim=1, keepdim=True).sqrt()
        V = V / l2_norms    
    
    # calculate the affinity matrix A:
    A = torch.matmul(V, torch.transpose(V, 0, 1))
    
    if plus_method == 'clipping':
        A = torch.relu(A)
    elif plus_method == 'moving':
        A = A + torch.abs(A.min()) + 1e-8

    A = A.fill_diagonal_(0)

    # just take the k nearest Neighbors into acount:
    if k is not None and k and k < A.shape[0]-1:
        mask = torch.zeros_like(A)
        _, idx = torch.topk(A, k, dim=1, sorted=False)
        mask = mask.scatter_(1, idx, 1)
        A = A*mask
    
    if A.min() < 0:
        print("Warning! A < 0: ", A.min())

    # symmetric adjacency matrix:
    W = 0.5 * (A + torch.transpose(A, 0, 1))

    #symmetrically normalize:
    D = torch.diag(W.sum(dim=1).pow(-0.5))
    W = D @ W @ D
    return W

def LabelPropagationModule(
    W: torch.Tensor,
    query_size: int,
    alpha: float = 0.5
) -> torch.Tensor:
    """
    inputs:
    - W; torch.tensor;
      symmetrically normalized adjacency matrix
      dim: [samples-size, samples-size] (all samples, support and query set)
        * e.g.: [64, 64]
    - query_size; int;
      # of query samples
    -alpha; float;
      hyperparameter

    """
    if alpha >= 1.0:
        alpha = 0.99
    elif alpha < 0.0:
        alpha = 0.0
    # generate Y matrix for a concatination of actives, inactives and queries:
    Y = torch.zeros((W.shape[0], 2), device=W.device)
    support_size = W.shape[0] - query_size
    Y[:int(support_size/2), 0] = 1
    Y[int(support_size/2):support_size, 1] = 1
    
    # Calculate Z:
    Z = torch.inverse(torch.eye(W.shape[0], device=W.device)-alpha*W) @ Y

    return Z

def ClassBalancingModule(
    Z: torch.Tensor,
    query_size: int,
    tau: float = 3.0
) -> torch.Tensor:
    """
    inputs:
    - Z; torch.tensor;
      matrix for predictions before class balancing
      dim: [samples-size, classes] (all samples, support and query set) (classes: active, inactive)
        * e.g.: [64, 2]
    - query_size; int;
      # of query samples
    - tau; float;
      hyperparameter
    """
    # go on with just the query samples:
    P = Z[Z.shape[0]-query_size:]
    P = P.pow(tau)
    p = torch.ones(P.shape[0], device=Z.device)
    q = 1 / P.shape[1] * p.sum() * torch.ones(P.shape[1], device=Z.device)

    if P.min() < 0:
        print("Warning! P_start < 0: ", P.min())
    
    # Sinkhorn-Knopp-Algorithm:
    while True:
        P_old = P
        # normalize P to row-wise sum p = 1
        P = F.normalize(P, p=1)
        # normalize P to column-wise sum q
        P = P @ torch.inverse(torch.diag(P.sum(dim=0))) @ torch.diag(q)
        if torch.allclose(P, P_old):
            break

    if P.min() < 0:
        print("Warning! P_end < 0: ", P.min())

    # make predictions 0 = active and 1 = inactive
    yhat = torch.argmax(P, dim=1)
    # special case handling for all queries predicted the same label
    if(torch.all(yhat == 0)):
        print(f"Warning! All {yhat.shape[0]} predictions after balancing were equal to active (0). Change the most unclear to inactive (1).")
        if torch.all(torch.std(P, dim=1) == torch.std(P, dim=1)[0]):
            yhat[random.randint(0, yhat.shape[0]-1)] = 1
        else:
            yhat[torch.std(P, dim=1) <= (torch.std(P,dim=1).min() + (torch.abs(torch.std(P,dim=1).max() - torch.std(P,dim=1).min())/10))] = 1
    elif(torch.all(yhat == 1)):
        print(f"Warning! All {yhat.shape[0]} predictions after balancing were equal to inactive (1). Change the most unclear to inactive (0).")
        if torch.all(torch.std(P, dim=1) == torch.std(P, dim=1)[0]):
            yhat[random.randint(0, yhat.shape[0]-1)] = 0
        else:
            yhat[torch.std(P, dim=1) <= (torch.std(P,dim=1).min() + (torch.abs(torch.std(P,dim=1).max() - torch.std(P,dim=1).min())/10))] = 0
    return yhat

def LabelCleaningModule(
    yhat: torch.Tensor,
    query_embedding: torch.Tensor,
    support_actives_embedding: torch.Tensor,
    support_inactives_embedding: torch.Tensor,
    n_epochs: int = 1000,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    train_individual: bool = False
) -> tuple:
    """
    inputs:
    - yhat; torch.tensor;
      predictions of the querys
      dim: [batch-size] (query samples, 0 = active, 1 = inactive)
    - query_embedding; torch.tensor;
      dim: [batch-size, embedding-dimension]
        * e.g.: [54, 1024]
    - support_actives_embedding; torch.tensor;
      dim: [padding-dim, embedding-dimension] (active support set samples in padding-dim)
        * e.g.: [5, 1024]
    - support_inactives_embedding; torch.tensor;
      dim: [padding-dim, embedding-dimension] (inactive support set samples in padding-dim)
        * e.g.: [5, 1024]
    - train_individual; boolean;
      decision about using the variant of training for each query sample an individual classifier or not
    """
    #Create the samples and lables tensors for training
    samples = torch.cat((support_actives_embedding, support_inactives_embedding, query_embedding), 0)
    samples = samples.detach()
    labels = torch.zeros((samples.shape[0]), dtype=torch.int64, device=samples.device)
    labels[:support_actives_embedding.shape[0]] = 0
    labels[support_actives_embedding.shape[0]:support_actives_embedding.shape[0]+support_inactives_embedding.shape[0]] = 1
    labels[support_actives_embedding.shape[0]+support_inactives_embedding.shape[0]:] = yhat

    if train_individual:
        losses = torch.zeros((query_embedding.shape[0]), device=samples.device)
        # train a classifier for each query sample individually
        for i in range(query_embedding.shape[0]):
            samplesact = torch.zeros((support_actives_embedding.shape[0]+support_inactives_embedding.shape[0]+1, samples.shape[1]), device=samples.device)
            samplesact[:support_actives_embedding.shape[0]+support_inactives_embedding.shape[0]] = samples[:support_actives_embedding.shape[0]+support_inactives_embedding.shape[0]]
            samplesact[-1] = samples[support_actives_embedding.shape[0]+support_inactives_embedding.shape[0]+i]
            labelsact = torch.zeros((samplesact.shape[0]), dtype=torch.int64, device=samplesact.device)
            labelsact[:support_actives_embedding.shape[0]+support_inactives_embedding.shape[0]] = labels[:support_actives_embedding.shape[0]+support_inactives_embedding.shape[0]]
            labelsact[-1] = labels[support_actives_embedding.shape[0]+support_inactives_embedding.shape[0]+i]

            # Create the 2-way classifier g
            g = classifierLabelCleaning(query_embedding.shape[1], 2)
            g = g.to(samplesact.device)
            g.train()
            optimizer = torch.optim.SGD(g.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss(reduction='none')

            # train g for n_epochs and collecting the loss for the samples
            for e in range(n_epochs):
                optimizer.zero_grad()
                pred = g(samplesact)
                loss = criterion(pred, labelsact)
                losses[i] = losses[i]+loss[-1]
                loss = loss.sum()/samplesact.shape[0]
                loss.backward()
                optimizer.step()
    else:
        # Create the 2-way classifier g
        g = classifierLabelCleaning(query_embedding.shape[1], 2)
        g = g.to(samples.device)
        g.train()
        optimizer = torch.optim.SGD(g.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(reduction='none')

        # train g for n_epochs and collecting the loss for the samples
        losses = torch.zeros((query_embedding.shape[0]), device=samples.device)
        for e in range(n_epochs):
            optimizer.zero_grad()
            pred = g(samples)
            loss = criterion(pred, labels)
            losses = losses+loss[support_actives_embedding.shape[0]+support_inactives_embedding.shape[0]:]
            loss = loss.sum()/samples.shape[0]
            loss.backward()
            optimizer.step()

    losses = losses/n_epochs
    # move losses to values between -1, 1 (-1 best inactive, 1 best active)
    losses[yhat == 1] = (losses[yhat == 1] / losses[yhat == 1].max()) * (-1) + 1 + 1e-8
    losses[yhat == 0] = (losses[yhat == 0] / losses[yhat == 0].max()) * (-1) + 1 + 1e-8
    losses[yhat == 1] = losses[yhat == 1] * (-1)
    best_active_id = losses.argmax()
    best_inactive_id = losses.argmin()

    return best_active_id, best_inactive_id, losses

class classifierLabelCleaning(nn.Module):
    """
    classifier used while the Label Cleaning process
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        ## Define the layers
        self.linear = nn.Linear(in_dim, out_dim)
      
    
    def forward(self, x):
        pred = self.linear(x)
        return pred

def ClassificationModule(    # Just works for supportset >= 2 each
    query_embedding: torch.Tensor,
    support_actives_embedding: torch.Tensor,
    support_inactives_embedding: torch.Tensor,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0005
) -> tuple:
    """
    inputs:
    - query_embedding; torch.tensor;
      dim: [batch-size, embedding-dimension]
        * e.g.: [54, 1024]
    - support_actives_embedding; torch.tensor;
      dim: [padding-dim, embedding-dimension] (active support set samples in padding-dim)
        * e.g.: [5, 1024]
    - support_inactives_embedding; torch.tensor;
      dim: [padding-dim, embedding-dimension] (inactive support set samples in padding-dim)
        * e.g.: [5, 1024]
    """
    
    if support_actives_embedding.shape[0] < 5:
        numberofclassifiers = support_actives_embedding.shape[0]
    else:
        numberofclassifiers = 5

    if numberofclassifiers == 1:
        print("Not Enough samples in the support set for this method!!!!!")
        print(supportsettoosmall)

    # Create the samples and lables tensors for training
    samples = torch.cat((support_actives_embedding, support_inactives_embedding), 0)
    samples = samples.detach()
    labels = torch.zeros((samples.shape[0]), dtype=torch.int64, device=samples.device)
    labels[:support_actives_embedding.shape[0]] = 0
    labels[support_actives_embedding.shape[0]:support_actives_embedding.shape[0]+support_inactives_embedding.shape[0]] = 1

    indicesActive = list(range(support_actives_embedding.shape[0]))
    random.shuffle(indicesActive)
    indicesInactive = list(range(support_inactives_embedding.shape[0]))
    random.shuffle(indicesInactive)
    indicesInactive = [x + len(indicesActive) for x in indicesInactive]

    valsizes = [int(len(indicesActive)/5)]*5
    valsizes[:len(indicesActive)%5] = [x + 1 for x in valsizes[:len(indicesActive)%5]]
    sectors = [valsizes[0], valsizes[0]+valsizes[1], valsizes[0]+valsizes[1]+valsizes[2],
                valsizes[0]+valsizes[1]+valsizes[2]+valsizes[3], valsizes[0]+valsizes[1]+valsizes[2]+valsizes[3], valsizes[4]
    ]
    groupedIndicesActive = [indicesActive[:sectors[0]], indicesActive[sectors[0]:sectors[1]], 
                            indicesActive[sectors[1]:sectors[2]], indicesActive[sectors[2]:sectors[3]], 
                            indicesActive[sectors[3]:]
    ]
    groupedIndicesInactive = [indicesInactive[:sectors[0]], indicesInactive[sectors[0]:sectors[1]], 
                            indicesInactive[sectors[1]:sectors[2]], indicesInactive[sectors[2]:sectors[3]], 
                            indicesInactive[sectors[3]:]
    ]
    groupedIndicesActive = groupedIndicesActive[:numberofclassifiers]
    groupedIndicesInactive = groupedIndicesInactive[:numberofclassifiers]

    # Train 5 Classifier
    classifiers = list(range(numberofclassifiers))
    for i in range(numberofclassifiers):
        inxsval = groupedIndicesActive[i] + groupedIndicesInactive[i]
        inxstrain = groupedIndicesActive[:i] + groupedIndicesActive[i+1:] + groupedIndicesInactive[:i] + groupedIndicesInactive[i+1:]
        inxstrain = sum(inxstrain, [])
        inxsval = torch.tensor(inxsval, device=samples.device)
        inxstrain = torch.tensor(inxstrain, device=samples.device)

        trainsamples = samples.index_select(0, inxstrain)
        trainlabels = labels.index_select(0, inxstrain)
        valsamples = samples.index_select(0, inxsval)
        vallabels = labels.index_select(0, inxsval)

        # Create the 2-way classifier g
        g = classifierLabelCleaning(samples.shape[1], 2)
        g = g.to(samples.device)
        optimizer = torch.optim.SGD(g.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        # train g
        g.eval()
        pred = g(valsamples)
        evalloss = criterion(pred, vallabels)
        classifiers[i] = g
        counter = 0
        while True:
            g.train()
            optimizer.zero_grad()
            pred = g(samples)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            g.eval()
            pred = g(valsamples)
            evallossnew = criterion(pred, vallabels)
            counter = counter + 1
            if evallossnew < evalloss:
                evalloss = evallossnew
                classifiers[i] = g
                counter = 0
            if counter == 50:
                break
    
    # use classifiers for predictions:
    preds = torch.zeros((query_embedding.shape[0]), device=samples.device)
    for g in classifiers:
        pred = g(query_embedding)
        _, pred = pred.max(1)
        preds = preds + pred

    # propataedLabelsMatrix support set part just 0 as not used anymore later
    propagatedLabelsMatrix = torch.zeros((query_embedding.shape[0]+support_actives_embedding.shape[0]+support_inactives_embedding.shape[0], 2), device=query_embedding.device)
    propagatedLabelsMatrix[support_actives_embedding.shape[0]+support_inactives_embedding.shape[0]:, 0] = numberofclassifiers - preds + 1e-8
    propagatedLabelsMatrix[support_actives_embedding.shape[0]+support_inactives_embedding.shape[0]:, 1] = preds + 1e-8

    return propagatedLabelsMatrix

#---------------------------------------------------------------------------------------

    
    
    
    