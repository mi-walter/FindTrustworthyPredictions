"""
This file includes the config for the autoregressive inference experiments of Vanilla MHNfs and Label Cleaning MHNfs,
e.g. path to the data ...
(Model hyperparameters are not included (see assets/mhnfs_data))

Some parts are taken and adapted from "Autoregressive activity prediction for low-data drug discovery" 
URL https://github.com/ml-jku/autoregressive_activity_prediction
"""
from dataclasses import dataclass

@dataclass
class Config:
    
    #-----------------------------------------------------------------------------------
    # Base settings
    seed: int = 1234
    
    # Data
    data_path: str =   # TODO set path to data folder
    nbr_ss_start: int = 8 # TODO set nuber of shots 
    nbr_support_set_candidates: int = 31+nbr_ss_start
    inference_batch_size: int = 64   # inference_batch_size has to be larger than query set ([nbr_support_set_candidates - nbr_ss_start] * 2)! Outputs are not batched together!!!
    
    # LabelCleaningMHNfs (Variant selection and hyperparameters)
    variantPred = 'BasicPred' # Variants: ['BasicPred', 'ClassifierPred', 'SimilarityModPred']
    ensemble: bool = False 
    individualLabelClean: bool = False
    k: int = 30
    norm = 'l1' # options: ['l1', 'l2']
    alpha = 0.5
    tau = 3.0

    # Experiment
    device='gpu'
    
    # Results
    results_path: str =   # TODO set path to result folder
