"""
This file includes the config for the training of the learned selection policy and 
the autoregressive inference experiment,
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
    
    # Experiment
    device='gpu'
    
    # Data
    data_path: str =  #TODO set path to basic data folder

    statistic_features_size: int = 10
    query_sizes = [5]
    support_sizes = [8]
    
    nbr_different_sample_set_train_cases: int = 10000
    nbr_different_sample_set_eval_cases: int = 1000
    task_ids_train = list(range(20)) # for classifier training
    task_ids_eval = list(range(20, 40)) # for classifier training

    # LabelCleaningMHNfs (Variant selection and hyperparameters)
    variantPred = 'BasicPred' # Variants: ['BasicPred', 'ClassifierPred', 'SimilarityModPred']
    ensemble = False 
    individualLabelClean = False
    k = 30
    norm = 'l1' # options: ['l1', 'l2']
    alpha = 0.5
    tau = 3.0

    # training hyperparameters
    layers = 6 # options: [3, 6]
    hidden_dim: int = 4096
    dropout: float = 0.3
    convergingSteps: int = 500
    learning_rate: float = 0.0001
    max_epochs: int = 10000
    batch_size: int = 4096

    # evaluation
    usetrainlast: bool = False
    nbr_ss_start: int = 8 # TODO set nuber of shots 
    nbr_support_set_candidates: int = 31+nbr_ss_start
    inference_batch_size: int = 64  # inference_batch_size has to be larger than query set ([nbr_support_set_candidates - nbr_ss_start] * 2)! Outputs are not batched together!!!
    
    # Results
    results_path: str =  # TODO set path to result folder
