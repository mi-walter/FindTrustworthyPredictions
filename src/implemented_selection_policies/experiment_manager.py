"""
This file include experiment manager modules.
They take the pretrained model and the data module as inputs and perform the
autoregressive inference experiments.

Some parts are taken and adapted from "Autoregressive activity prediction for low-data drug discovery" 
URL https://github.com/ml-jku/autoregressive_activity_prediction
"""

#---------------------------------------------------------------------------------------
# Dependencies
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
from cfg import Config
from dataloader import DataModule_for_autregressive_inference_on_FSMol_testset
current_loc = __file__.rsplit("/",3)[0]
import sys
import os
sys.path.append(current_loc)

from src.mhnfs.load_trained_model import MHNfs_inference_module, MHNfs_inference_module_LabelCleaning

#---------------------------------------------------------------------------------------
# Experiment Manager

class ExperimentManager_AutoregressiveInference(): # Vanilla MHNfs
    def __init__(self, cfg=Config()):
        self.model = MHNfs_inference_module(device=cfg.device) # wrapper for the pretrained MHNfs model
        self.cfg = cfg
        
        task_ids_file = np.load(cfg.data_path + "task_ids.npy")
        task_ids_unique = np.unique(task_ids_file)
        self.task_ids = list(range(len(task_ids_unique))) # 40 for eval 157 for test
        self.experiment_rerun_seeds = [8915, 1318, 7221, 7540,  664, 6137, 6833, 8471,
                                       9449, 7322]  # fixed seeds on which to run the experiment
        
    def perform_experiment(self):
        auc_dict = dict()
        dauc_pr_dict = dict()
        
        # for tracking errors of iteration steps:
        failsActives = np.zeros((len(self.task_ids), len(self.experiment_rerun_seeds), 31)) # tasks x reruns x iterations
        failsInactives = np.zeros((len(self.task_ids), len(self.experiment_rerun_seeds), 31)) # tasks x reruns x iterations
        
        # Loop over tasks
        for task_id in self.task_ids:
            print(f'for {self.cfg.results_path}:')
            print(f'Processing task {task_id} ...')
            data_module = DataModule_for_autregressive_inference_on_FSMol_testset(task_id,
                                                                                  self.cfg)
            
            # Loop over reruns
            aucs_rerun_dict = dict()
            dauc_prs_rerun_dict = dict()
            for seed in self.experiment_rerun_seeds:
                print(f'... rerun seed {seed} ...')
                data_module.sample_initial_support_set_and_candidate_pool(seed)
                
                # Iteratively add members to support set with predicting pseudo labels
                
                aucs = list()
                dauc_prs = list()

                for i in range(self.cfg.nbr_support_set_candidates):
                    support_actives_input = data_module.support_set_actives_inputs
                    support_inactives_input = data_module.support_set_inactives_inputs
                    
                    # Prediction evaluation set
                    evaluation_predictions = []
                    evaluation_labels = []
                    for batch in data_module.eval_dataloader:
                        query_inputs = torch.unsqueeze(batch['inputs'],1)
                        query_labels = batch['labels']
                        
                        predictions = self.model.predict(
                            query_inputs,
                            support_actives_input.expand(query_inputs.shape[0], -1, -1),
                            support_inactives_input.expand(query_inputs.shape[0], -1, -1)
                        )
                        evaluation_predictions = (evaluation_predictions + 
                                                  list(predictions.numpy())
                                                  )
                        evaluation_labels = (evaluation_labels + 
                                               list(query_labels.numpy())
                                               )
                    
                    # Performance on evaluation set
                    auc = roc_auc_score(evaluation_labels, evaluation_predictions)
                    auc_pr = average_precision_score(evaluation_labels,
                                                     evaluation_predictions)
                    nbr_inactives, nbr_actives = np.unique(evaluation_labels,
                                                           return_counts=True)[1]
                    random_clf = nbr_actives / (nbr_actives + nbr_inactives)
                    dauc_pr = auc_pr - random_clf
                    
                    aucs.append(auc)
                    dauc_prs.append(dauc_pr)
                    
                    # Prediction support set candidates (query set)
                    if i == (self.cfg.nbr_support_set_candidates-self.cfg.nbr_ss_start):
                        break
                    candidate_predictions = []
                    for batch in data_module.support_candidate_pool_dataloader:
                        query_inputs = torch.unsqueeze(batch['inputs'], 1)
                        predictions = self.model.predict(
                            query_inputs,
                            support_actives_input.expand(query_inputs.shape[0], -1, -1),
                            support_inactives_input.expand(query_inputs.shape[0], -1, -1)
                        )
                        candidate_predictions = (candidate_predictions + 
                                                  list(predictions.numpy())
                                                  )
                            
                    # Select candidates for augmenting support set
                    raw_active_cand_id = np.argmax(candidate_predictions)
                    raw_inactive_cand_id = np.argmin(candidate_predictions)
                    
                    # analysizing the methode for tracking errors of iteration steps
                    if data_module.support_candidate_pool[raw_active_cand_id] not in data_module.mol_ids_active_ss_candidates:
                        failsActives[self.task_ids.index(task_id), self.experiment_rerun_seeds.index(seed), i] = failsActives[self.task_ids.index(task_id), self.experiment_rerun_seeds.index(seed), i] + 1
                    if data_module.support_candidate_pool[raw_inactive_cand_id] not in data_module.mol_ids_inactive_ss_candidates:
                        failsInactives[self.task_ids.index(task_id), self.experiment_rerun_seeds.index(seed), i] = failsInactives[self.task_ids.index(task_id), self.experiment_rerun_seeds.index(seed), i] + 1

                    # augmente support set with selected candidates
                    data_module.add_candidate_to_support_set_and_remove_from_pool(
                        raw_active_cand_id,
                        raw_inactive_cand_id
                    )
            
                aucs_rerun_dict[seed] = aucs
                dauc_prs_rerun_dict[seed] = dauc_prs
            
            auc_dict[task_id] = aucs_rerun_dict
            dauc_pr_dict[task_id] = dauc_prs_rerun_dict

            # print wrongly added samples statistic of the task
            print("task"+str(task_id)+": \nwrongly added actives (mean over reruns): "+str(failsActives.mean(axis=1)[self.task_ids.index(task_id),:])+"\nwrongly added inactives (mean over reruns): "+ str(failsInactives.mean(axis=1)[self.task_ids.index(task_id),:]))
        
        # Save results
        current_loc = __file__.rsplit("/",3)[0]
        with open(current_loc + self.cfg.results_path + "autoreg_inf_fsmol.pkl", 'wb') as f:
            pickle.dump([auc_dict, dauc_pr_dict], f)
        print(f'for {self.cfg.results_path} results saved.')

        # plot and print statistics of wrongly added samples over tasks
        failsActivesMean = failsActives.mean(axis=1)
        failsActivesStd = failsActives.std(axis=1)
        failsInactivesMean = failsInactives.mean(axis=1)
        failsInactivesStd = failsInactives.std(axis=1)
        if len(self.task_ids) == 1:
            plt.figure(figsize=(15,7))
            x_values = np.arange(1, failsActivesMean.shape[1]+1, 1)
            y_values = failsActivesMean[0, :]
            y_error = failsActivesStd[0, :]
            plt.errorbar(x_values, y_values, y_error, linestyle='None', marker='o', color='b')
            x_values = np.arange(1, failsInactivesMean.shape[1]+1, 1)
            y_values = failsInactivesMean[0,:]
            y_error = failsInactivesStd[0, :]
            plt.errorbar(x_values, y_values, y_error, linestyle='None', marker='^', color = 'r')
            plt.grid()
            plt.xticks(np.arange(1, failsActivesMean.shape[1]+1, 1))
            plt.title(f'Wrongly added actives (blue) and inactives (red) (mean over reruns) of task {self.task_ids[0]}')
            plt.ylabel(f'ration of wrongly added actives/inactives')
            plt.xlabel(f'iteration')
            current_loc = __file__.rsplit("/",3)[0]
            plt.savefig(current_loc + self.cfg.results_path + 
                        f'wrongly_added_actives_and_inactives_task{self.task_ids[0]}.pdf',
                        bbox_inches='tight',pad_inches=0.1)
            plt.close()
        else:
            plt.figure(figsize=(15,7))
            x_values = np.arange(1, failsActivesMean.shape[1]+1, 1)
            y_values = failsActivesMean.mean(axis=0)
            y_error = failsActivesMean.std(axis=0)
            plt.errorbar(x_values, y_values, y_error, linestyle='None', marker='o', color='b')
            x_values = np.arange(1, failsInactivesMean.shape[1]+1, 1)
            y_values = failsInactivesMean.mean(axis=0)
            y_error = failsInactivesMean.std(axis=0)
            plt.errorbar(x_values, y_values, y_error, linestyle='None', marker='^', color = 'r')
            plt.grid()
            plt.xticks(np.arange(1, failsActivesMean.shape[1]+1, 1))
            plt.title(f'Wrongly added actives (blue) and inactives (red) (mean over tasks)')
            plt.ylabel(f'ration of wrongly added actives/inactives')
            plt.xlabel(f'iteration')
            current_loc = __file__.rsplit("/",3)[0]
            plt.savefig(current_loc + self.cfg.results_path +
                        f'wrongly_added_actives_and_inactives.pdf',
                        bbox_inches='tight',pad_inches=0.1)

        np.set_printoptions(threshold=np.inf)
        print("wrongly added actives (mean over reruns):")
        print(failsActivesMean)
        print("Mean for each iteration: ", failsActivesMean.mean(axis=0))
        print("Mean for each task: ", failsActivesMean.mean(axis=1))
        print("\nwrongly added inactives (mean over reruns):")
        print(failsInactivesMean)
        print("Mean for each iteration: ", failsInactivesMean.mean(axis=0))
        print("Mean for each task: ", failsInactivesMean.mean(axis=1))

class ExperimentManager_AutoregressiveInference_LabelCleaning(): # Label Cleaning MHNfs
    def __init__(self, cfg=Config()):
        self.model = MHNfs_inference_module_LabelCleaning(device=cfg.device)  # wrapper for the pretrained MHNfs model
        self.cfg = cfg
        
        task_ids_file = np.load(cfg.data_path + "task_ids.npy")
        task_ids_unique = np.unique(task_ids_file)
        self.task_ids = list(range(len(task_ids_unique))) # 40 for eval 157 for test
        self.experiment_rerun_seeds = [8915, 1318, 7221, 7540,  664, 6137, 6833, 8471,
                                       9449, 7322]  # fixed seeds on which to run the experiment
        
    def perform_experiment(self):
        auc_dict = dict()
        dauc_pr_dict = dict()
        
        # for tracking errors of iteration steps:
        failsActives = np.zeros((len(self.task_ids), len(self.experiment_rerun_seeds), 31)) # tasks x reruns x iterations
        failsInactives = np.zeros((len(self.task_ids), len(self.experiment_rerun_seeds), 31)) # tasks x reruns x iterations
        
        # Loop over tasks
        for task_id in self.task_ids:
            print(f'for {self.cfg.results_path}:')
            print(f'Processing task {task_id} ...')
            data_module = DataModule_for_autregressive_inference_on_FSMol_testset(task_id,
                                                                                  self.cfg)
            
            # Loop over reruns
            aucs_rerun_dict = dict()
            dauc_prs_rerun_dict = dict()
            for seed in self.experiment_rerun_seeds:
                print(f'... rerun seed {seed} ...')
                data_module.sample_initial_support_set_and_candidate_pool(seed)

                # Iteratively add members to support set with predicting pseudo labels
                
                aucs = list()
                dauc_prs = list()

                for i in range(self.cfg.nbr_support_set_candidates):
                    support_actives_input = data_module.support_set_actives_inputs
                    support_inactives_input = data_module.support_set_inactives_inputs
                    
                    # Prediction evaluation set
                    evaluation_predictions = []
                    evaluation_labels = []
                    for batch in data_module.eval_dataloader:
                        query_inputs = torch.unsqueeze(batch['inputs'],1)
                        query_labels = batch['labels']
                        
                        predictions = self.model.predict(
                            query_inputs,
                            support_actives_input.expand(query_inputs.shape[0], -1, -1),
                            support_inactives_input.expand(query_inputs.shape[0], -1, -1)
                        )
                        evaluation_predictions = (evaluation_predictions + 
                                                  list(predictions.numpy())
                                                  )
                        evaluation_labels = (evaluation_labels + 
                                               list(query_labels.numpy())
                                               )
                    
                    # Performance on evaluation set
                    auc = roc_auc_score(evaluation_labels, evaluation_predictions)
                    auc_pr = average_precision_score(evaluation_labels,
                                                     evaluation_predictions)
                    nbr_inactives, nbr_actives = np.unique(evaluation_labels,
                                                           return_counts=True)[1]
                    random_clf = nbr_actives / (nbr_actives + nbr_inactives)
                    dauc_pr = auc_pr - random_clf
                    
                    aucs.append(auc)
                    dauc_prs.append(dauc_pr)
                    
                    plt.close('all')
                    # Prediction support set candidates (query set)
                    if i == (self.cfg.nbr_support_set_candidates-self.cfg.nbr_ss_start):
                        break
                    predictions = 0
                    for batch in data_module.support_candidate_pool_dataloader:
                        query_inputs = torch.unsqueeze(batch['inputs'], 1)
                        # support_actives/inactives_input(1 x supportsetactives(start with initial goes up to max-1 in last iteration) x 2248)
                        predictions = self.model.predictLabelCleaning(
                            query_inputs,
                            support_actives_input.expand(query_inputs.shape[0], -1, -1),
                            support_inactives_input.expand(query_inputs.shape[0], -1, -1)
                        )

                    # Select candidates for augmenting support set
                    best_active_id = predictions.argmax()
                    best_inactive_id = predictions.argmin()

                    raw_active_cand_id = np.int64(best_active_id.numpy())
                    raw_inactive_cand_id = np.int64(best_inactive_id.numpy())

                    # analysizing the methode for tracking errors of iteration steps
                    if data_module.support_candidate_pool[raw_active_cand_id] not in data_module.mol_ids_active_ss_candidates:
                        failsActives[self.task_ids.index(task_id), self.experiment_rerun_seeds.index(seed), i] = failsActives[self.task_ids.index(task_id), self.experiment_rerun_seeds.index(seed), i] + 1
                    if data_module.support_candidate_pool[raw_inactive_cand_id] not in data_module.mol_ids_inactive_ss_candidates:
                        failsInactives[self.task_ids.index(task_id), self.experiment_rerun_seeds.index(seed), i] = failsInactives[self.task_ids.index(task_id), self.experiment_rerun_seeds.index(seed), i] + 1

                    # augmente support set with selected candidates
                    data_module.add_candidate_to_support_set_and_remove_from_pool(
                        raw_active_cand_id,
                        raw_inactive_cand_id
                    )
                    
                aucs_rerun_dict[seed] = aucs
                dauc_prs_rerun_dict[seed] = dauc_prs
            
            auc_dict[task_id] = aucs_rerun_dict
            dauc_pr_dict[task_id] = dauc_prs_rerun_dict

            # print wrongly added samples statistic of the task
            print("task"+str(task_id)+": \nwrongly added actives (mean over reruns): "+str(failsActives.mean(axis=1)[self.task_ids.index(task_id),:])+"\nwrongly added inactives (mean over reruns): "+ str(failsInactives.mean(axis=1)[self.task_ids.index(task_id),:]))

        # Save results
        current_loc = __file__.rsplit("/",3)[0]
        with open(current_loc + self.cfg.results_path + "autoreg_inf_fsmol.pkl", 'wb') as f:
            pickle.dump([auc_dict, dauc_pr_dict], f)
        print(f'for {self.cfg.results_path} results saved.')

        # plot and print statistics of wrongly added samples over tasks
        failsActivesMean = failsActives.mean(axis=1)
        failsActivesStd = failsActives.std(axis=1)
        failsInactivesMean = failsInactives.mean(axis=1)
        failsInactivesStd = failsInactives.std(axis=1)
        if len(self.task_ids) == 1:
            plt.figure(figsize=(15,7))
            x_values = np.arange(1, failsActivesMean.shape[1]+1, 1)
            y_values = failsActivesMean[0, :]
            y_error = failsActivesStd[0, :]
            plt.errorbar(x_values, y_values, y_error, linestyle='None', marker='o', color='b')
            plt.plot(x_values, np.ones_like(y_values)*y_values.mean(), color='b', alpha=0.3)
            x_values = np.arange(1, failsInactivesMean.shape[1]+1, 1)
            y_values = failsInactivesMean[0,:]
            y_error = failsInactivesStd[0, :]
            plt.errorbar(x_values, y_values, y_error, linestyle='None', marker='^', color = 'r')
            plt.plot(x_values, np.ones_like(y_values)*y_values.mean(), color = 'r', alpha=0.3)
            plt.grid()
            plt.xticks(np.arange(1, failsActivesMean.shape[1]+1, 1))
            plt.title(f'Wrongly added actives (blue) and inactives (red) (mean over reruns) of task {self.task_ids[0]}')
            plt.ylabel(f'ration of wrongly added actives/inactives')
            plt.xlabel(f'iteration')
            current_loc = __file__.rsplit("/",3)[0]
            plt.savefig(current_loc + self.cfg.results_path + 
                        f'wrongly_added_actives_and_inactives.pdf',
                        bbox_inches='tight',pad_inches=0.1)
            plt.close()
        else:
            plt.figure(figsize=(15,7))
            x_values = np.arange(1, failsActivesMean.shape[1]+1, 1)
            y_values = failsActivesMean.mean(axis=0)
            y_error = failsActivesMean.std(axis=0)
            plt.errorbar(x_values, y_values, y_error, linestyle='None', marker='o', color='b')
            x_values = np.arange(1, failsInactivesMean.shape[1]+1, 1)
            y_values = failsInactivesMean.mean(axis=0)
            y_error = failsInactivesMean.std(axis=0)
            plt.errorbar(x_values, y_values, y_error, linestyle='None', marker='^', color = 'r')
            plt.grid()
            plt.xticks(np.arange(1, failsActivesMean.shape[1]+1, 1))
            plt.title(f'Wrongly added actives (blue) and inactives (red) (mean over tasks)')
            plt.ylabel(f'ration of wrongly added actives/inactives')
            plt.xlabel(f'iteration')
            current_loc = __file__.rsplit("/",3)[0]
            plt.savefig(current_loc + self.cfg.results_path + 
                        f'wrongly_added_actives_and_inactives.pdf',
                        bbox_inches='tight',pad_inches=0.1)
        
        np.set_printoptions(threshold=np.inf)
        print("wrongly added actives (mean over reruns):")
        print(failsActivesMean)
        print("Mean for each iteration: ", failsActivesMean.mean(axis=0))
        print("Mean for each task: ", failsActivesMean.mean(axis=1))
        print("\nwrongly added inactives (mean over reruns):")
        print(failsInactivesMean)
        print("Mean for each iteration: ", failsInactivesMean.mean(axis=0))
        print("Mean for each task: ", failsInactivesMean.mean(axis=1))

#---------------------------------------------------------------------------------------
if __name__ == '__main__':
    #experiment_manager = ExperimentManager_AutoregressiveInference()
    experiment_manager = ExperimentManager_AutoregressiveInference_LabelCleaning()
    experiment_manager.perform_experiment()
        
            
                