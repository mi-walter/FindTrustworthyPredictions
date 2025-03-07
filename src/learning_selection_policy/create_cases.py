"""
This file is used for creating the data for training the classifiers of the Learned MHNfs.
In the init function the the data that should be created can be specified.
"""

#---------------------------------------------------------------------------------------
# Dependencies
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import torch.nn as nn
import pickle
from cfg import Config
from dataloader import DataModule_for_learning_selection_policy_on_FSMol_testset
current_loc = __file__.rsplit("/",3)[0]
import sys
import os
sys.path.append(current_loc)

from src.mhnfs.load_trained_model import MHNfs_inference_module, MHNfs_inference_module_LabelCleaning

#---------------------------------------------------------------------------------------
# Create Cases

class CasesCreator():
    def __init__(self, cfg=Config()):
        np.random.seed(cfg.seed)
        self.modelMHNfs = MHNfs_inference_module(device=cfg.device)  # wrapper for the pretrained MHNfs model
        self.modelLabelCleaning = MHNfs_inference_module_LabelCleaning(device=cfg.device)  # wrapper for the pretrained MHNfs model
        self.cfg = cfg
        
        self.task_ids = list(range(20)) # the tasks on which the data should be created; tasks taken from cfg
        self.experiment_seed = 1234
        self.nbr_different_sample_set_cases = 10000 # number of different scenarios for each task that should be created
        
    def perform_creation(self):
        # prepare all of the sample cases:
        if len(self.cfg.query_sizes) != len(self.cfg.support_sizes):
            print("Different number of query_sizes and support_sizes!")
            print(error)
        cases_labels_active_classifier = []
        cases_labels_inactive_classifier = []
        cases_inputs_classifier = []
        print("generating cases...")
        for task in self.task_ids:
            print(f"...for task {task}")
            # generate data module:
            data_module = DataModule_for_learning_selection_policy_on_FSMol_testset(task, self.cfg)
            for i in range(len(self.cfg.query_sizes)): # if different query and support set sizes should be used for creation
                query_size = self.cfg.query_sizes[i]
                support_size = self.cfg.support_sizes[i]
                
                # generate sample cases:
                data_module.generate_sample_cases(self.experiment_seed, support_size=support_size, query_size=query_size, nbr_different_sample_set_cases=self.nbr_different_sample_set_cases)

                for case in range(self.nbr_different_sample_set_cases):

                    support_actives_input = data_module.cases_support_set_actives_inputs[case]
                    support_inactives_input = data_module.cases_support_set_inactives_inputs[case]
                    query_actives_input = data_module.cases_query_set_actives_inputs[case]
                    query_inactives_input = data_module.cases_query_set_inactives_inputs[case]
                    calcGain_dataloader = data_module.cases_calcGain_dataloader[case] # used for evaluating the gain in case of augmentation

                    # generate the labels and inputs for the classifier:
                    labels_active_classifier, labels_inactive_classifier = calc_labels(support_actives_input, support_inactives_input, query_actives_input, query_inactives_input, calcGain_dataloader, self.modelMHNfs, query_size)
                    cases_labels_active_classifier.append(labels_active_classifier)
                    cases_labels_inactive_classifier.append(labels_inactive_classifier)

                    inputs_classifier = calc_inputs(support_actives_input, support_inactives_input, torch.cat([query_actives_input, query_inactives_input], dim=1).squeeze(dim=0).unsqueeze(dim=1), self.modelMHNfs, self.modelLabelCleaning, query_size)
                    cases_inputs_classifier.append(inputs_classifier)
            # Save cases:
            current_loc = __file__.rsplit("/",3)[0]
            with open(current_loc + f"/src/learning_selection_policy/cases/cases_task{self.task_ids[0]}-{self.task_ids[-1]}_{self.nbr_different_sample_set_cases}_support_size{self.cfg.support_sizes[0]}-{self.cfg.support_sizes[-1]}_query_size{self.cfg.query_sizes[0]}-{self.cfg.query_sizes[-1]}_notcomplete.pkl", 'wb') as f:
                pickle.dump([cases_labels_active_classifier, cases_labels_inactive_classifier, cases_inputs_classifier], f)
            print("saved task ", task)
        # Save cases:
        current_loc = __file__.rsplit("/",3)[0]
        with open(current_loc + f"/src/learning_selection_policy/cases/cases_task{self.task_ids[0]}-{self.task_ids[-1]}_{self.nbr_different_sample_set_cases}_support_size{self.cfg.support_sizes[0]}-{self.cfg.support_sizes[-1]}_query_size{self.cfg.query_sizes[0]}-{self.cfg.query_sizes[-1]}.pkl", 'wb') as f:
            pickle.dump([cases_labels_active_classifier, cases_labels_inactive_classifier, cases_inputs_classifier], f)

def calc_labels(support_actives_input, support_inactives_input, query_actives_input, query_inactives_input, calcGain_dataloader, modelMHNfs, query_size):
    # Prediction for calcGain set
    predictions = []
    labels = []
    for batch in calcGain_dataloader:
        query_inputs = torch.unsqueeze(batch['inputs'],1)
        query_labels = batch['labels']
        prediction = modelMHNfs.predict(
            query_inputs,
            support_actives_input.expand(query_inputs.shape[0], -1, -1),
            support_inactives_input.expand(query_inputs.shape[0], -1, -1)
        )
        predictions = (predictions + list(prediction.numpy()))
        labels = (labels + list(query_labels.numpy()))
    # Performance on calcGain set
    calcGain_base_auc = roc_auc_score(labels, predictions)
    
    # Prediction for calcGain set with one added query for active and inactive independet and search each best
    best_gain_active = -np.inf
    best_gain_inactive = -np.inf
    best_gain_active_id = None
    best_gain_inactive_id = None
    for i in range(query_size):
        # Prediction for calcGain set with one active query added
        predictions = []
        labels = []
        for batch in calcGain_dataloader:
            query_inputs = torch.unsqueeze(batch['inputs'],1)
            query_labels = batch['labels']
            prediction = modelMHNfs.predict(
                query_inputs,
                torch.cat([support_actives_input, query_actives_input[:, i, :].unsqueeze(dim=1)], dim=1).expand(query_inputs.shape[0], -1, -1),
                support_inactives_input.expand(query_inputs.shape[0], -1, -1)
            )
            predictions = (predictions + list(prediction.numpy()))
            labels = (labels + list(query_labels.numpy()))
        # Performance on calcGain set with one active query added
        calcGain_auc = roc_auc_score(labels, predictions)
        gain = calcGain_auc - calcGain_base_auc
        if gain > best_gain_active:
            best_gain_active = gain
            best_gain_active_id = i

        # Prediction for calcGain set with one inactive query added
        predictions = []
        labels = []
        for batch in calcGain_dataloader:
            query_inputs = torch.unsqueeze(batch['inputs'],1)
            query_labels = batch['labels']
            prediction = modelMHNfs.predict(
                query_inputs,
                support_actives_input.expand(query_inputs.shape[0], -1, -1),
                torch.cat([support_inactives_input, query_inactives_input[:, i, :].unsqueeze(dim=1)], dim=1).expand(query_inputs.shape[0], -1, -1)
            )
            predictions = (predictions + list(prediction.numpy()))
            labels = (labels + list(query_labels.numpy()))
        # Performance on calcGain set with one inactive query added
        calcGain_auc = roc_auc_score(labels, predictions)
        gain = calcGain_auc - calcGain_base_auc
        if gain > best_gain_inactive:
            best_gain_inactive = gain
            best_gain_inactive_id = i

    labels_active_classifier = torch.zeros((1, 2*query_size))
    labels_inactive_classifier = torch.zeros((1, 2*query_size))
    labels_active_classifier[:, best_gain_active_id] = 1
    labels_inactive_classifier[:, query_size+best_gain_inactive_id] = 1

    return labels_active_classifier, labels_inactive_classifier

def calc_inputs(support_actives_input, support_inactives_input, query_inputs, modelMHNfs, modelLabelCleaning, query_size):
    # generate features from the models for the queries
    predictions_MHNfs = modelMHNfs.predict(
        query_inputs,
        support_actives_input.expand(query_inputs.shape[0], -1, -1),
        support_inactives_input.expand(query_inputs.shape[0], -1, -1)
    )
    predictions_MHNfs = list(predictions_MHNfs.numpy())

    predictions_LabelCleaning = modelLabelCleaning.predictLabelCleaning(
        query_inputs,
        support_actives_input.expand(query_inputs.shape[0], -1, -1),
        support_inactives_input.expand(query_inputs.shape[0], -1, -1)
    )
    predictions_LabelCleaning = list(predictions_LabelCleaning.numpy())

    predictions_MHNfs = np.array(predictions_MHNfs).reshape(1,-1)
    predictions_LabelCleaning = np.array(predictions_LabelCleaning).reshape(1,-1)

    # generate statistical features for the queries
    sims_active, sims_inactive = modelMHNfs.get_similarities_in_embedding_space(
        query_inputs,
        support_actives_input.expand(query_inputs.shape[0], -1, -1),
        support_inactives_input.expand(query_inputs.shape[0], -1, -1)
    )
    
    actives_max = np.max(sims_active, 1).reshape(1,-1)
    actives_mean = np.mean(sims_active, 1).reshape(1,-1)
    actives_min = np.min(sims_active, 1).reshape(1,-1)
    actives_std = np.std(sims_active, 1).reshape(1,-1)

    inactives_max = np.max(sims_inactive, 1).reshape(1,-1)
    inactives_mean = np.mean(sims_inactive, 1).reshape(1,-1)
    inactives_min = np.min(sims_inactive, 1).reshape(1,-1)
    inactives_std = np.std(sims_inactive, 1).reshape(1,-1)

    # create input tensor from the models and statistics
    statistic_features_size = 10
    inputs_classifier = torch.zeros((1, 2*query_size, statistic_features_size))
    # add feature mean from similarity to active support set:
    inputs_classifier[:, :, 0] = torch.from_numpy(actives_mean)
    # add feature max from similarity to active support set:
    inputs_classifier[:, :, 1] = torch.from_numpy(actives_max)
    # add feature min from similarity to active support set:
    inputs_classifier[:, :, 2] = torch.from_numpy(actives_min)
    # add feature std from similarity to active support set:
    inputs_classifier[:, :, 3] = torch.from_numpy(actives_std)
    # add feature mean from similarity to inactive support set:
    inputs_classifier[:, :, 4] = torch.from_numpy(inactives_mean)
    # add feature max from similarity to inactive support set:
    inputs_classifier[:, :, 5] = torch.from_numpy(inactives_max)
    # add feature min from similarity to inactive support set:
    inputs_classifier[:, :, 6] = torch.from_numpy(inactives_min)
    # add feature std from similarity to inactive support set:
    inputs_classifier[:, :, 7] = torch.from_numpy(inactives_std)
    # add feature prediction of MHNfs:
    inputs_classifier[:, :, 8] = torch.from_numpy(predictions_MHNfs)
    # add feature prediction of LabelCleaning:
    inputs_classifier[:, :, 9] = torch.from_numpy(predictions_LabelCleaning)
    
    return inputs_classifier
#---------------------------------------------------------------------------------------
if __name__ == '__main__':
    cases_creator = CasesCreator()
    cases_creator.perform_creation()
        
            
                