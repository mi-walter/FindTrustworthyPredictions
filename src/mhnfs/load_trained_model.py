"""
This file is used for loading the models.

Some parts are taken and adapted from "Autoregressive activity prediction for low-data drug discovery" 
URL https://github.com/ml-jku/autoregressive_activity_prediction
"""

#---------------------------------------------------------------------------------------
# Dependencies
import pytorch_lightning as pl
import torch
from src.mhnfs.model import MHNfs, MHNfsLabelCleaning

#---------------------------------------------------------------------------------------
# Define function

def load_trained_model():
    """
    This function loads a trained MHNfs model from a checkpoint file.
        * Training on FS-Mol training set
        * Hyperparameter search on FS-Mol validation set 
    """
    
    pl.seed_everything(1234)
    current_loc = __file__.rsplit("/",3)[0]
    model = MHNfs.load_from_checkpoint(current_loc +
                                       "/assets/mhnfs_data/mhnfs_checkpoint.ckpt")
    model._update_context_set_embedding()
    model.eval()
    
    return model 

class MHNfs_inference_module:
    """
    This module is a wrapper for the pre-trained MHNfs model and is suppesed to be used
    for inference.
    """
    def __init__(self, device:['cpu', 'gpu']='cpu'):
        
        if device == 'cpu':
            self.device = device
        elif device == 'gpu':
            self.device = 'cuda'
        
        # Load model
        self.model = load_trained_model()
        
        # Move to GPU if requested
        if device == 'gpu':
            self.model = self.model.to('cuda')
            self.model.context_embedding = self.model.context_embedding.to('cuda')
    
    def predict(self, query_tensor, support_actives_tensor, support_inactives_tensor):
        """
        This function creates support-set-size tensor and feeeds all tensors into the
        model forward function.
        """
        
        # Create support set size tensors
        support_actives_size = torch.tensor(support_actives_tensor.shape[1])
        support_inactives_size = torch.tensor(support_inactives_tensor.shape[1])
        
        # Make predictions
        predictions = self.model(
            query_tensor.to(self.device),
            support_actives_tensor.to(self.device),
            support_inactives_tensor.to(self.device),
            support_actives_size.to(self.device),
            support_inactives_size.to(self.device)
        ).detach().cpu()
        
        return predictions

    def get_similarities_in_embedding_space(self, query_tensor, support_actives_tensor, support_inactives_tensor):
        """
        This function creates support-set-size tensor and feeeds all tensors into the
        model get_similarities_in_embedding_space function to get similarities in
        embedding space for Learned MHNfs.
        """

        # Create support set size tensors
        support_actives_size = torch.tensor(support_actives_tensor.shape[1])
        support_inactives_size = torch.tensor(support_inactives_tensor.shape[1])

        # Make predictions
        outs_active, outs_inactive = self.model.get_similarities_in_embedding_space(
            query_tensor.to(self.device),
            support_actives_tensor.to(self.device),
            support_inactives_tensor.to(self.device),
            support_actives_size.to(self.device),
            support_inactives_size.to(self.device)
        )

        return outs_active, outs_inactive 

def load_trained_model_LabelCleaning():
    """
    This function loads a trained MHNfs model from a checkpoint file.
        * Training on FS-Mol training set
        * Hyperparameter search on FS-Mol validation set 
    """
    
    pl.seed_everything(1234)
    current_loc = __file__.rsplit("/",3)[0]
    model = MHNfsLabelCleaning.load_from_checkpoint(current_loc +
                                       "/assets/mhnfs_data/mhnfs_checkpoint.ckpt")
    model._update_context_set_embedding()
    model.eval()
    
    return model 
        
class MHNfs_inference_module_LabelCleaning(MHNfs_inference_module):
    """
    This module is a wrapper for the pre-trained Label Cleaning MHNfs model and is suppesed to be used
    for inference.
    """
    def __init__(self, device:['cpu', 'gpu']='cpu'):
        
        super(MHNfs_inference_module_LabelCleaning, self).__init__(device)

        # Load model
        self.model = load_trained_model_LabelCleaning()
        
        # Move to GPU if requested
        if device == 'gpu':
            self.model = self.model.to('cuda')
            self.model.context_embedding = self.model.context_embedding.to('cuda')

    def predictLabelCleaning(self, query_tensor, support_actives_tensor, support_inactives_tensor,
            kNearestNeighborGraph = 60, alphaLabelPropagation = 0.5, tauClassBalancing = 3.0
        ):
        """
        This function creates support-set-size tensor and feeeds all tensors into the
        model forwardLabelCleaning function.
        """
        
        # Create support set size tensors
        support_actives_size = torch.tensor(support_actives_tensor.shape[1])
        support_inactives_size = torch.tensor(support_inactives_tensor.shape[1])
        
        # Make predictions
        predictions = self.model.forwardLabelCleaning(
            query_tensor.to(self.device),
            support_actives_tensor.to(self.device),
            support_inactives_tensor.to(self.device),
            support_actives_size.to(self.device),
            support_inactives_size.to(self.device)
        ).detach().cpu()
        
        return predictions
        
        