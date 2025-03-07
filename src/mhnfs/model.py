"""
This file inclues the Vanilla MHNfs and the Label Cleaning MHNfs model

Some parts are taken and adapted from "Autoregressive activity prediction for low-data drug discovery" 
URL https://github.com/ml-jku/autoregressive_activity_prediction
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from cfg import Config as cfgExperiment

from src.mhnfs.modules import (
    EncoderBlock,
    ContextModule,
    LayerNormalizingBlock,
    CrossAttentionModule,
    SimilarityModule,
    SimilarityModule_similarities,
    NearestNeighborGraphModule,
    LabelPropagationModule,
    ClassBalancingModule,
    LabelCleaningModule,
    ClassificationModule
)

class MHNfs(pl.LightningModule):
    """
    The MHNfs is a few-shot drug-discovery model for activity prediction.

    For a requested query molecule, MHNfs predicts activity, while known knowledge from
    the support set is used.

    MHNfs can be seen as an embedding-based few-shot method since the prediction is
    based on similarities of molecule representations in a learned "representation
    space". Being able to build rich, expressive molecule representations is the key for
    a predictive model.

    MHNfs consists of
    three consecutive modules:
    - the context module,
    - the cross attention module, and
    - the similarity module.

    The context module associates the query and support set molecules with context -
    i.e., a large set of training molecules.

    The cross-attention module allows for information sharing between query and support
    set molecules.

    The similirity modules computes pair-wise similarity values between query and sup-
    port set molecules and uses these similarity values to build a prediction from a
    weighted sum over the support set labels.
    """

    def __init__(self, cfg):
        super(MHNfs, self).__init__()

        # Config
        self.cfg = cfg

        # Load context set
        current_loc = __file__.rsplit("/",3)[0]
        self.context = (
            torch.unsqueeze(
                torch.from_numpy(
                    np.load(current_loc + "/assets/mhnfs_data/full_context_set.npy")
                ),
                0,
            )
            .float()
        )

        self.context_embedding = torch.ones(1, 512, 1024)

        self.layerNorm_context = torch.nn.LayerNorm(
            cfg.model.associationSpace_dim,
            elementwise_affine=cfg.model.layerNormBlock.affine,
        )

        # Encoder
        self.encoder = EncoderBlock(cfg)

        # Context module
        self.contextModule = ContextModule(self.cfg)

        # Layernormalizing-block
        self.layerNormBlock = LayerNormalizingBlock(cfg)

        # Cross-attention module
        self.crossAttentionModule = CrossAttentionModule(self.cfg)

        # Similarity module
        self.similarity_function = SimilarityModule
        self.similarity_function_sims = SimilarityModule_similarities

        # Output function
        self.sigmoid = torch.nn.Sigmoid()
        self.prediction_scaling = cfg.model.prediction_scaling

    def forward(
        self,
        query_molecules: torch.Tensor,
        support_molecules_active: torch.Tensor,
        support_molecules_inactive: torch.Tensor,
        support_set_actives_size: torch.Tensor,
        support_set_inactives_size: torch.Tensor,
    ) -> torch.Tensor:
        # Get embeddings from molecule encoder
        query_embedding = self.encoder(query_molecules)
        support_actives_embedding = self.encoder(support_molecules_active)
        support_inactives_embedding = self.encoder(support_molecules_inactive)

        # Retrieve updated representations from the context module
        # - Layernorm
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.layerNormBlock(
            query_embedding, support_actives_embedding, support_inactives_embedding
        )

        # - Expand support set related tensors
        support_actives_embedding = support_actives_embedding.expand(
                                                    query_embedding.shape[0], -1, -1)
        support_inactives_embedding = support_inactives_embedding.expand(
                                                    query_embedding.shape[0], -1, -1)
        support_set_actives_size = support_set_actives_size.expand(
                                                    query_embedding.shape[0])
        support_set_inactives_size = support_set_inactives_size.expand(
                                                    query_embedding.shape[0])
        
        # - Context module
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.contextModule(
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
            self.context_embedding,
        )

        # Allow for information sharing between query and support set
        # - Layernorm
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.layerNormBlock(
            query_embedding, support_actives_embedding, support_inactives_embedding
        )

        # - Cross-attention module
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.crossAttentionModule(
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        )

        # Build predictions from a weighted sum over support set labels
        # - Layernorm:
        if self.cfg.model.layerNormBlock.usage:
            (
                query_embedding,
                support_actives_embedding,
                support_inactives_embedding,
            ) = self.layerNormBlock(
                query_embedding, support_actives_embedding, support_inactives_embedding
            )

        # - Similarity module:
        predictions_support_actives = self.similarity_function(
            query_embedding,
            support_actives_embedding,
            support_set_actives_size,
            self.cfg,
        )

        predictions_support_inactives = self.similarity_function(
            query_embedding,
            support_inactives_embedding,
            support_set_inactives_size,
            self.cfg,
        )

        predictions = predictions_support_actives - predictions_support_inactives

        predictions = self.sigmoid(self.prediction_scaling * predictions)

        return predictions

    def get_similarities_in_embedding_space(
        self,
        query_molecules: torch.Tensor,
        support_molecules_active: torch.Tensor,
        support_molecules_inactive: torch.Tensor,
        support_set_actives_size: torch.Tensor,
        support_set_inactives_size: torch.Tensor,
    ) -> torch.Tensor:
        # Get embeddings from molecule encoder
        query_embedding = self.encoder(query_molecules)
        support_actives_embedding = self.encoder(support_molecules_active)
        support_inactives_embedding = self.encoder(support_molecules_inactive)

        # Retrieve updated representations from the context module
        # - Layernorm
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.layerNormBlock(
            query_embedding, support_actives_embedding, support_inactives_embedding
        )

        # - Expand support set related tensors
        support_actives_embedding = support_actives_embedding.expand(
                                                    query_embedding.shape[0], -1, -1)
        support_inactives_embedding = support_inactives_embedding.expand(
                                                    query_embedding.shape[0], -1, -1)
        support_set_actives_size = support_set_actives_size.expand(
                                                    query_embedding.shape[0])
        support_set_inactives_size = support_set_inactives_size.expand(
                                                    query_embedding.shape[0])

        # - Context module
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.contextModule(
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
            self.context_embedding,
        )

        # Allow for information sharing between query and support set
        # - Layernorm
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.layerNormBlock(
            query_embedding, support_actives_embedding, support_inactives_embedding
        )

        # - Cross-attention module
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.crossAttentionModule(
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        )

        # Build predictions from a weighted sum over support set labels
        # - Layernorm:
        if self.cfg.model.layerNormBlock.usage:
            (
                query_embedding,
                support_actives_embedding,
                support_inactives_embedding,
            ) = self.layerNormBlock(
                query_embedding, support_actives_embedding, support_inactives_embedding
            )

        # - Similarity module:
        active_outs = self.similarity_function_sims(
            query_embedding,
            support_actives_embedding,
            support_set_actives_size,
            self.cfg,
        )

        inactive_outs = self.similarity_function_sims(
            query_embedding,
            support_inactives_embedding,
            support_set_inactives_size,
            self.cfg,
        )

        return active_outs.cpu().detach().numpy(), inactive_outs.cpu().detach().numpy()

    @torch.no_grad()
    def _update_context_set_embedding(self):
        """
        This function randomly samples the context set as a subset of all available
        training molecules
        """
        max_rows = self.context.shape[1]
        number_requested_rows = int(
            np.rint(self.cfg.model.context.ratio_training_molecules * max_rows)
        )

        sampled_rows = torch.randperm(max_rows)[:number_requested_rows]

        self.context_embedding = self.layerNorm_context(
            self.encoder(self.context[:, sampled_rows, :])
        )

class MHNfsLabelCleaning(MHNfs):
    """
    NHNFs with implementation of the iterative label cleaning proposed in paper https://arxiv.org/pdf/2012.07962.pdf
    """

    def __init__(self, cfg):
        super(MHNfsLabelCleaning, self).__init__(cfg)

        # NearestNeighborGraph
        self.NearestNeighborGraphModule = NearestNeighborGraphModule

        # LabelPropagation
        self.LabelPropagationModule = LabelPropagationModule

        # ClassBalancing
        self.ClassBalancingModule = ClassBalancingModule

        # LabelCleaning
        self.LabelCleaningModule = LabelCleaningModule

        # Classification Replacement of LabelPropagation
        self.ClassificationModule = ClassificationModule
    
    def forwardEmbedding(
        self,
        query_molecules: torch.Tensor,
        support_molecules_active: torch.Tensor,
        support_molecules_inactive: torch.Tensor,
        support_set_actives_size: torch.Tensor,
        support_set_inactives_size: torch.Tensor, 
    ) -> tuple:

        # Everything used from the original MHNFs until the output of the Cross-Attention-Module:
        # Used as the Embedding from where the new methode starts

        # Get embeddings from molecule encoder
        query_embedding = self.encoder(query_molecules)
        support_actives_embedding = self.encoder(support_molecules_active)
        support_inactives_embedding = self.encoder(support_molecules_inactive)

        # Retrieve updated representations from the context module
        # - Layernorm
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.layerNormBlock(
            query_embedding, support_actives_embedding, support_inactives_embedding
        )

        # - Expand support set related tensors
        support_actives_embedding = support_actives_embedding.expand(
                                                    query_embedding.shape[0], -1, -1)
        support_inactives_embedding = support_inactives_embedding.expand(
                                                    query_embedding.shape[0], -1, -1)
        support_set_actives_size = support_set_actives_size.expand(
                                                    query_embedding.shape[0])
        support_set_inactives_size = support_set_inactives_size.expand(
                                                    query_embedding.shape[0])
        
        # - Context module
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.contextModule(
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
            self.context_embedding,
        )

        # Allow for information sharing between query and support set
        # - Layernorm
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.layerNormBlock(
            query_embedding, support_actives_embedding, support_inactives_embedding
        )

        # - Cross-attention module
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.crossAttentionModule(
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        )

        # - Layernorm:
        if self.cfg.model.layerNormBlock.usage:
            (
                query_embedding,
                support_actives_embedding,
                support_inactives_embedding,
            ) = self.layerNormBlock(
                query_embedding, support_actives_embedding, support_inactives_embedding
            )

        return query_embedding, support_actives_embedding, support_set_actives_size, support_inactives_embedding, support_set_inactives_size

    def forwardIterativeLabelCleaning(
        self,
        query_embedding: torch.Tensor,
        support_actives_embedding: torch.Tensor,
        support_inactives_embedding: torch.Tensor
    ) -> tuple:

        # Find best active and best inactive sample in the query set by following iterative label cleaning:
        
        # Reshape vectors:
        # query_molecules: [batch-size, 1, embedding-dimension] -> [batch-size, embedding-dimension]
        # support_actives_embedding: [batch-size, padding-dim, embedding-dimension] -> [padding-dim, embedding-dimension]
        # support_inactives_embedding: [batch-size, padding-dim, embedding-dimension] -> [padding-dim, embedding-dimension]
        query_embedding = torch.squeeze(query_embedding, 1)
        support_actives_embedding = support_actives_embedding[0]
        support_inactives_embedding = support_inactives_embedding[0] 

        # NearestNeighborGraph:
        adjacencyMatrix = self.NearestNeighborGraphModule(
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
            k = cfgExperiment.k,
            norm = cfgExperiment.norm
        )

        # LabelPropagation:
        propagatedLabelsMatrix = self.LabelPropagationModule(
            adjacencyMatrix,
            query_embedding.shape[0],
            alpha = cfgExperiment.alpha
        )

        # ClassBalancing:
        predictions = self.ClassBalancingModule(
            propagatedLabelsMatrix,
            query_embedding.shape[0],
            tau = cfgExperiment.tau
        )

        # LabelCleaning:
        best_active_id, best_inactive_id, losses = self.LabelCleaningModule(
            predictions,
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
            train_individual = cfgExperiment.individualLabelClean
        )

        return best_active_id, best_inactive_id, losses

    def forwardIterativeLabelCleaningReplaceLPwithSimilarityModule(
        self,
        query_embedding: torch.Tensor,
        support_actives_embedding: torch.Tensor,
        support_inactives_embedding: torch.Tensor,
        support_set_actives_size,
        support_set_inactives_size
    ) -> tuple:

        # Find best active and best inactive sample in the query set by following iterative label cleaning.
        # Instead of Label Propagation the original Similarity Module from Vanilla MHNfs is used:
        
        # - Similarity module:
        predictions_support_actives = self.similarity_function(
            query_embedding,
            support_actives_embedding,
            support_set_actives_size,
            self.cfg,
        )

        predictions_support_inactives = self.similarity_function(
            query_embedding,
            support_inactives_embedding,
            support_set_inactives_size,
            self.cfg,
        )

        predictions = predictions_support_actives - predictions_support_inactives

        predictions = self.sigmoid(self.prediction_scaling * predictions)
        predictions = torch.squeeze(predictions, 1)

        # propataedLabelsMatrix (support set part just 0 as not used anymore later)
        propagatedLabelsMatrix = torch.zeros((query_embedding.shape[0]+support_actives_embedding.shape[1]+support_inactives_embedding.shape[1], 2), device=query_embedding.device)
        propagatedLabelsMatrix[support_actives_embedding.shape[1]+support_inactives_embedding.shape[1]:, 0] = predictions
        propagatedLabelsMatrix[support_actives_embedding.shape[1]+support_inactives_embedding.shape[1]:, 1] = 1 - predictions

        # Reshape vectors:
        # query_molecules: [batch-size, 1, embedding-dimension] -> [batch-size, embedding-dimension]
        # support_actives_embedding: [batch-size, padding-dim, embedding-dimension] -> [padding-dim, embedding-dimension]
        # support_inactives_embedding: [batch-size, padding-dim, embedding-dimension] -> [padding-dim, embedding-dimension]

        query_embedding = torch.squeeze(query_embedding, 1)
        support_actives_embedding = support_actives_embedding[0]
        support_inactives_embedding = support_inactives_embedding[0] 

        # ClassBalancing:
        predictions = self.ClassBalancingModule(
            propagatedLabelsMatrix,
            query_embedding.shape[0],
            tau = cfgExperiment.tau
        )

        # LabelCleaning:
        best_active_id, best_inactive_id, losses = self.LabelCleaningModule(
            predictions,
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
            train_individual = cfgExperiment.individualLabelClean
        )

        return best_active_id, best_inactive_id, losses

    def forwardIterativeLabelCleaningReplaceLPwithClassifier(
        self,
        query_embedding: torch.Tensor,
        support_actives_embedding: torch.Tensor,
        support_inactives_embedding: torch.Tensor
    ) -> tuple:

        # Find best active and best inactive sample in the query set by following iterative label cleaning.
        # Instead of Label Propagation a Classifier is trained on the support samples and used for predicting
        # the lables of the query samples:

        # Reshape vectors:
        # query_molecules: [batch-size, 1, embedding-dimension] -> [batch-size, embedding-dimension]
        # support_actives_embedding: [batch-size, padding-dim, embedding-dimension] -> [padding-dim, embedding-dimension]
        # support_inactives_embedding: [batch-size, padding-dim, embedding-dimension] -> [padding-dim, embedding-dimension]

        query_embedding = torch.squeeze(query_embedding, 1)
        support_actives_embedding = support_actives_embedding[0]
        support_inactives_embedding = support_inactives_embedding[0] 

        if support_actives_embedding.shape[0] < 2: # Just works with at least 2 samples in the support set
            # NearestNeighborGraph:
            adjacencyMatrix = self.NearestNeighborGraphModule(
                query_embedding,
                support_actives_embedding,
                support_inactives_embedding,
                k = cfgExperiment.k,
                norm = cfgExperiment.norm
            )

            # LabelPropagation:
            propagatedLabelsMatrix = self.LabelPropagationModule(
                adjacencyMatrix,
                query_embedding.shape[0],
                alpha = cfgExperiment.alpha
            )

        else:
            # - Classification module:
            propagatedLabelsMatrix = self.ClassificationModule(
                query_embedding,
                support_actives_embedding,
                support_inactives_embedding
            )

        # ClassBalancing:
        predictions = self.ClassBalancingModule(
            propagatedLabelsMatrix,
            query_embedding.shape[0],
            tau = cfgExperiment.tau
        )

        # LabelCleaning:
        best_active_id, best_inactive_id, losses = self.LabelCleaningModule(
            predictions,
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
            train_individual = cfgExperiment.individualLabelClean
        )

        return best_active_id, best_inactive_id, losses

    def forwardLabelCleaning(
        self,
        query_molecules: torch.Tensor,
        support_molecules_active: torch.Tensor,
        support_molecules_inactive: torch.Tensor,
        support_set_actives_size: torch.Tensor,
        support_set_inactives_size: torch.Tensor
    ) -> torch.Tensor:

        # in case of ensemble True the few-shot scenario is devided into few one-shot scenarios
        # the losses which are the selection base are added up for the queries
        if cfgExperiment.ensemble:
            rounds = support_set_actives_size
        else:
            rounds = 1
        orig_support_molecules_active = support_molecules_active
        orig_support_molecules_inactive = support_molecules_inactive
        orig_support_set_actives_size = support_set_actives_size
        orig_support_set_inactives_size = support_set_inactives_size
        endlosses = torch.zeros((query_molecules.shape[0]), device=query_molecules.device)
        for i in range(rounds):
            if cfgExperiment.ensemble:
                support_molecules_active = orig_support_molecules_active[:, i, :].unsqueeze(dim=1)
                support_molecules_inactive = orig_support_molecules_inactive[:, i, :].unsqueeze(dim=1)
                support_set_actives_size = orig_support_set_actives_size / orig_support_set_actives_size
                support_set_inactives_size = orig_support_set_inactives_size / orig_support_set_inactives_size

            # Get embedding from where the new methode starts by using everything until the Similarity Module from the origninal MHNFs:
            query_embedding, support_actives_embedding, support_set_actives_size, support_inactives_embedding, support_set_inactives_size = self.forwardEmbedding(
                query_molecules,
                support_molecules_active,
                support_molecules_inactive,
                support_set_actives_size,
                support_set_inactives_size,
            )

            # Find best active and best inactive sample in the query set based on their losses
            best_active_id = None
            best_inactive_id = None
            losses = None

            # Using the basic idea of Iterative Label Cleaning:
            if cfgExperiment.variantPred == 'BasicPred':
                best_active_id, best_inactive_id, losses = self.forwardIterativeLabelCleaning(
                    query_embedding,
                    support_actives_embedding,
                    support_inactives_embedding
                )

            # Using the Iterative Label Cleaning, but replace the Label Propagation with the Similarity Module
            elif cfgExperiment.variantPred == 'SimilarityModPred':
                best_active_id, best_inactive_id, losses = self.forwardIterativeLabelCleaningReplaceLPwithSimilarityModule(
                    query_embedding,
                    support_actives_embedding,
                    support_inactives_embedding,
                    support_set_actives_size,
                    support_set_inactives_size
                )

            # Using the Iterative Label Cleaning, but replace the Label Propagation with the Classification Module
            elif cfgExperiment.variantPred == 'ClassifierPred':
                best_active_id, best_inactive_id, losses = self.forwardIterativeLabelCleaningReplaceLPwithClassifier(
                    query_embedding,
                    support_actives_embedding,
                    support_inactives_embedding
                )
            else:
                print(NoValidVariantChosen)

            # add up the losses for the rounds of the ensemble
            endlosses = endlosses + losses

        return endlosses