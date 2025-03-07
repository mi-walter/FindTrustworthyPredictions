"""
This file includes the dataloader which provides the data for the autoregressive
inference experiment.
The tasks are the ones included in the FS-Mol test set.

For each task, the available datapoints, i.e. a molecular structure with
associated labels, are split into a) a support set candidates set, and
b) an evaluation set. For each rerun c) a initial support set is drawn from the support
set candidates set. The rest forms d) the query set.

    a. Support set candidates set
    * consists of both 31+xshots active and 31+shots inactive molecules
    * randomly drawn

    b. Evaluation set
    * Includes all molecules which are not in [a.]
    * For every iteration, MHNfs predicts the activity for all molecules in the
      evaluation set
    * Performance is reported in terms of AUC and ΔAUC-PR.

    c. Initial support set
    * consists of shots active and shots inactive molecule
    * randomly drawn from [a.] for each rerun
    * based on **real labels**

    d. Query set / Candidate set
    * molecules of [a.] not in [c.]
    * MHNfs predicts pseudo-labels for these molecules, and chooses, for every
      iteartion, one molecule with active pseudo-label and one molecule with inactive
      pseudo-label to add to the support set.

# Preprocessed FS-Mol data
The pre-processed FS-Mol test data is stored in triplets (molecule-id, task-id, label).
Also the molecular inputs for the MHNfs model, i.e. feature vectors, are available.
We load this data, split the data points as described above, and build a dataloader from
it.

Further it also includes the dataloaders for creating the data for learning and for 
training the classifiers.

Some parts are taken and adapted from "Autoregressive activity prediction for low-data drug discovery" 
URL https://github.com/ml-jku/autoregressive_activity_prediction
"""

#---------------------------------------------------------------------------------------
# Dependencies
import numpy as np
import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from cfg import Config

#---------------------------------------------------------------------------------------
# Data-Module

class DataModule_for_autregressive_inference_on_FSMol_testset:
    
    def __init__(self, task_id, cfg=Config()):
      
      #-------------------------------------------------------------------------------
      # Helper functions
      def _split_data(mol_ids, labels, seed, nbr_support_set_candidates):
        """
        This function takes the mol-ids and labels (from the tripets) and creates the
        support set candidates, and evaluation set.
        (The initial support set is created later).
        """
        
        # set seed
        np.random.seed(seed)
        
        # Get indices of active and inactive molecules
        mol_ids_active = mol_ids[labels == 1].flatten()
        mol_ids_inactive = mol_ids[labels == 0].flatten()
        
        # Randomly shuffle the indices
        np.random.shuffle(mol_ids_active)
        np.random.shuffle(mol_ids_inactive)
        
        # Split the indices into support set candidates and evaluation set
        mol_ids_active_ss_candidates = mol_ids_active[:nbr_support_set_candidates]
        mol_ids_inactive_ss_candidates = mol_ids_inactive[:nbr_support_set_candidates]
        
        mol_ids_active_eval = mol_ids_active[nbr_support_set_candidates:]
        mol_ids_inactive_eval = mol_ids_inactive[nbr_support_set_candidates:]
        
        labels_eval = np.array([1] * mol_ids_active_eval.shape[0] +
                                [0] * mol_ids_inactive_eval.shape[0])
        mol_ids_eval = np.array(list(mol_ids_active_eval) + 
                                list(mol_ids_inactive_eval))
        
        return (mol_ids_active_ss_candidates, mol_ids_inactive_ss_candidates,
                mol_ids_eval, labels_eval)
      
      class EvalDataset():
        def __init__(self, mol_ids, labels, mol_inputs):
          self.mol_ids = mol_ids
          self.labels = labels
          self.mol_inputs = mol_inputs
          self.len = mol_ids.shape[0]
        def __getitem__(self, index):
          sample_labels = self.labels[index]
          feature_mtx_ids = self.mol_ids[index]
          sample_inputs = self.mol_inputs[feature_mtx_ids]
          
          sample = {'inputs': sample_inputs, 'labels': sample_labels}
          return sample
        def __len__(self):
            return self.len
        
      
      #-------------------------------------------------------------------------------
      # Main part of the function
      self.cfg = cfg
      self.task_id = task_id
      
      # Initialize task specific data storage
      self.task_statistics = dict()
      
      # Load FS-Mol triplets and features
      mol_ids = np.load(cfg.data_path + "mol_ids.npy")
      task_ids = np.load(cfg.data_path + "task_ids.npy")
      labels = np.load(cfg.data_path + "labels.npy").astype('float32')
      self.mol_inputs = np.load(cfg.data_path + "mol_inputs.npy").astype('float32')
      
      # We filter the datapoints relevant for the requested task
      bool_filter = (task_ids == task_id)
      mol_ids = mol_ids[bool_filter]
      labels = labels[bool_filter]
      
      # Compute nbr of active and inactive molecules and store info in task_statistics
      nbr_inactives, nbr_actives = np.unique(labels, return_counts=True)[1]
      self.task_statistics['nbr_inactives_all'] = nbr_inactives
      self.task_statistics['nbr_actives_all'] = nbr_actives
      
      # Split data into evaluation set and support set pool
      (self.mol_ids_active_ss_candidates, self.mol_ids_inactive_ss_candidates,
        self.mol_ids_eval, self.labels_eval
        ) = _split_data(mol_ids,
                        labels,
                        cfg.seed,
                        cfg.nbr_support_set_candidates)
        
      # Create Evaluation Dataset
      self.eval_dataset = EvalDataset(self.mol_ids_eval,
                                      self.labels_eval,
                                      self.mol_inputs)
      self.eval_dataloader = DataLoader(self.eval_dataset,
                                          batch_size=cfg.inference_batch_size,
                                          shuffle=False,)
    
    def sample_initial_support_set_and_candidate_pool(self, seed):
      """
      This function samples nbr_ss_start active and nbr_ss_start inactive molecules from the support set
      canidate pool to create an initial support set.
      
      seed: "experiment rerun seed"
      """
      
      np.random.seed(seed)
      np.random.shuffle(self.mol_ids_active_ss_candidates)
      np.random.shuffle(self.mol_ids_inactive_ss_candidates)
      
      self.support_set_actives_ids = self.mol_ids_active_ss_candidates[:self.cfg.nbr_ss_start]
      self.support_set_inactives_ids = self.mol_ids_inactive_ss_candidates[:self.cfg.nbr_ss_start]
      
      # Create a support set candidate pool which does not include the drawn indices
      # We omit label information here
      self.support_candidate_pool = np.concatenate(
        [self.mol_ids_active_ss_candidates[self.cfg.nbr_ss_start:],
         self.mol_ids_inactive_ss_candidates[self.cfg.nbr_ss_start:]],
        0)
      
      # Create support set and candite pool inputs
      self.support_set_actives_inputs = torch.from_numpy(
        self.mol_inputs[self.support_set_actives_ids]).unsqueeze(0)
      self.support_set_inactives_inputs = torch.from_numpy(
        self.mol_inputs[self.support_set_inactives_ids]).unsqueeze(0)
      
      candidate_dataset = self.CandidateDataSet(self.support_candidate_pool,
                                               self.mol_inputs)
      self.support_candidate_pool_dataloader = DataLoader(
        candidate_dataset,
        batch_size=self.cfg.inference_batch_size,
        shuffle=False,
        )
    
    def add_candidate_to_support_set_and_remove_from_pool(self,
                                                          raw_active_candidate_ids,
                                                          raw_inactive_candidate_ids):
      """
      This function adds a member from the candidate pool to the support set and removes
      it from the pool.
      """
      assert type(raw_active_candidate_ids)==np.int64
      assert type(raw_inactive_candidate_ids)==np.int64
      
      raw_active_candidate_ids = list([raw_active_candidate_ids])
      raw_inactive_candidate_ids = list([raw_inactive_candidate_ids])
      
      active_cand_ids = self.support_candidate_pool[raw_active_candidate_ids]
      inactive_cand_ids = self.support_candidate_pool[raw_inactive_candidate_ids]
      
      # add members to support set
      self.support_set_actives_ids = np.concatenate([self.support_set_actives_ids,
                                                     active_cand_ids], 0)
      self.support_set_inactives_ids = np.concatenate([self.support_set_inactives_ids,
                                                       inactive_cand_ids], 0)
      
      # remove members from candidate pool      
      ids_to_remove = (list(raw_active_candidate_ids) +
                       list(raw_inactive_candidate_ids))
      self.support_candidate_pool = np.delete(self.support_candidate_pool,
                                              ids_to_remove)
      
      # update inputs
      self.support_set_actives_inputs = torch.from_numpy(
        self.mol_inputs[self.support_set_actives_ids]).unsqueeze(0)
      self.support_set_inactives_inputs = torch.from_numpy(
        self.mol_inputs[self.support_set_inactives_ids]).unsqueeze(0)
      
      candidate_dataset = self.CandidateDataSet(self.support_candidate_pool,
                                               self.mol_inputs)
      self.support_candidate_pool_dataloader = DataLoader(
        candidate_dataset,
        batch_size=self.cfg.inference_batch_size,
        shuffle=False,
        )
    
    class CandidateDataSet():
      def __init__(self, mol_ids, mol_inputs):
        self.mol_ids = mol_ids
        self.mol_inputs = mol_inputs
        self.len = mol_ids.shape[0]
      def __getitem__(self, index):
        feature_mtx_ids = self.mol_ids[index]
        sample_inputs = self.mol_inputs[feature_mtx_ids]
        sample = {'inputs': sample_inputs}
        return sample
      def __len__(self):
        return self.len

class DataModule_for_learning_selection_policy_on_FSMol_testset:  # used in create_cases
    
    def __init__(self, task_id, cfg=Config()):
      self.cfg = cfg
      self.task_id = task_id
      
      # Initialize task specific data storage
      self.task_statistics = dict()
      
      # Load FS-Mol triplets and features
      mol_ids = np.load(cfg.data_path + "mol_ids.npy")
      task_ids = np.load(cfg.data_path + "task_ids.npy")
      labels = np.load(cfg.data_path + "labels.npy").astype('float32')
      self.mol_inputs = np.load(cfg.data_path + "mol_inputs.npy").astype('float32')
      
      # We filter the datapoints relevant for the requested task
      bool_filter = (task_ids == task_id)
      mol_ids = mol_ids[bool_filter]
      labels = labels[bool_filter]
      
      # Compute nbr of active and inactive molecules and store info in task_statistics
      nbr_inactives, nbr_actives = np.unique(labels, return_counts=True)[1]
      self.task_statistics['nbr_inactives_all'] = nbr_inactives
      self.task_statistics['nbr_actives_all'] = nbr_actives
      print(f"for this task there are {nbr_actives} active and {nbr_inactives} inactive molecules")
        
      # set seed
      np.random.seed(cfg.seed)
        
      # Get indices of active and inactive molecules
      self.mol_ids_active = mol_ids[labels == 1].flatten()
      self.mol_ids_inactive = mol_ids[labels == 0].flatten()
        
      # Randomly shuffle the indices
      np.random.shuffle(self.mol_ids_active)
      np.random.shuffle(self.mol_ids_inactive)

    def sample_support_set_and_query_set(self, seed: int = 1234, support_size: int = 8, query_size: int = 5):
      """
      This function samples support_size active and support_size inactive molecules for the support set and
      query_size active and query_size inactive molecules for the query set. The rest is used for evaluating the labels for the gain.
      """
      np.random.seed(seed)
      np.random.shuffle(self.mol_ids_active)
      np.random.shuffle(self.mol_ids_inactive)

      support_set_actives_ids = self.mol_ids_active[: support_size]
      support_set_inactives_ids = self.mol_ids_inactive[: support_size]
      query_set_actives_ids = self.mol_ids_active[support_size: support_size+query_size]
      query_set_inactives_ids = self.mol_ids_inactive[support_size: support_size+query_size]
      calcGain_set_actives_ids = self.mol_ids_active[support_size+query_size:]
      calcGain_set_inactives_ids = self.mol_ids_inactive[support_size+query_size:]
      labels_calcGain = np.array([1] * calcGain_set_actives_ids.shape[0] +
                              [0] * calcGain_set_inactives_ids.shape[0])
      mol_ids_calcGain = np.array(list(calcGain_set_actives_ids) + 
                              list(calcGain_set_actives_ids))

      # Create dataset for calculating the gain
      calcGain_dataset = self.CalcGainDataset(mol_ids_calcGain,
                                      labels_calcGain,
                                      self.mol_inputs)
      calcGain_dataloader = DataLoader(calcGain_dataset,
                                          batch_size=self.cfg.inference_batch_size,
                                          shuffle=False,)

      # Create support set and query set inputs
      support_set_actives_inputs = torch.from_numpy(
        self.mol_inputs[support_set_actives_ids]).unsqueeze(0)
      support_set_inactives_inputs = torch.from_numpy(
        self.mol_inputs[support_set_inactives_ids]).unsqueeze(0)
      query_set_actives_inputs = torch.from_numpy(
        self.mol_inputs[query_set_actives_ids]).unsqueeze(0)
      query_set_inactives_inputs = torch.from_numpy(
        self.mol_inputs[query_set_inactives_ids]).unsqueeze(0)
      
      return support_set_actives_inputs, support_set_inactives_inputs, query_set_actives_inputs, query_set_inactives_inputs, calcGain_dataloader

    def generate_sample_cases(self, seed: int = 1234, support_size: int = 8, query_size: int = 5, nbr_different_sample_set_cases: int = 10):
      """
      This function samples support_size active and support_size inactive molecules for the support set and
      query_size active and query_size inactive molecules for the query set. Such cases are generated nbr_different_sample_set_cases times.
      """
      self.cases_support_set_actives_inputs = []
      self.cases_support_set_inactives_inputs = []
      self.cases_query_set_actives_inputs = []
      self.cases_query_set_inactives_inputs = []
      self.cases_calcGain_dataloader = []

      for i in range(nbr_different_sample_set_cases):
        support_set_actives_inputs, support_set_inactives_inputs, query_set_actives_inputs, query_set_inactives_inputs, calcGain_dataloader = self.sample_support_set_and_query_set(seed=seed, support_size=support_size, query_size=query_size)

        self.cases_support_set_actives_inputs.append(support_set_actives_inputs)
        self.cases_support_set_inactives_inputs.append(support_set_inactives_inputs)
        self.cases_query_set_actives_inputs.append(query_set_actives_inputs)
        self.cases_query_set_inactives_inputs.append(query_set_inactives_inputs)
        self.cases_calcGain_dataloader.append(calcGain_dataloader)

    class CalcGainDataset():
      def __init__(self, mol_ids, labels, mol_inputs):
        self.mol_ids = mol_ids
        self.labels = labels
        self.mol_inputs = mol_inputs
        self.len = mol_ids.shape[0]
      def __getitem__(self, index):
        sample_labels = self.labels[index]
        feature_mtx_ids = self.mol_ids[index]
        sample_inputs = self.mol_inputs[feature_mtx_ids]
        
        sample = {'inputs': sample_inputs, 'labels': sample_labels}
        return sample
      def __len__(self):
          return self.len

class DataModule_for_learning_selection_policy_training_with_batches:

    def __init__(self, cfg=Config()):

      class TrainingDataset():
        def __init__(self, inputs_classifier, labels_active_classifier, labels_inactive_classifier):
          self.labels_active_classifier = labels_active_classifier
          self.labels_inactive_classifier = labels_inactive_classifier
          self.inputs_classifier = inputs_classifier
          self.len = inputs_classifier.shape[0]
        def __getitem__(self, index):
          sample_label_active_classifier = self.labels_active_classifier[index]
          sample_label_inactive_classifier = self.labels_inactive_classifier[index]
          sample_input = self.inputs_classifier[index, :]
          
          sample = {'input': sample_input, 'label_active_classifier': sample_label_active_classifier, 'label_inactive_classifier': sample_label_inactive_classifier}
          return sample
        def __len__(self):
            return self.len

      # main part of the function
      self.cfg = cfg

      current_loc = __file__.rsplit("/",3)[0]
      with open(current_loc + f"/src/learning_selection_policy/cases/cases_task{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_support_size{self.cfg.support_sizes[0]}-{self.cfg.support_sizes[-1]}_query_size{self.cfg.query_sizes[0]}-{self.cfg.query_sizes[-1]}.pkl", "rb") as f:
          cases_train_labels_active_classifier, cases_train_labels_inactive_classifier, cases_train_inputs_classifier = pickle.load(f)
      with open(current_loc + f"/src/learning_selection_policy/cases/cases_task{self.cfg.task_ids_eval[0]}-{self.cfg.task_ids_eval[-1]}_{self.cfg.nbr_different_sample_set_eval_cases}_support_size{self.cfg.support_sizes[0]}-{self.cfg.support_sizes[-1]}_query_size{self.cfg.query_sizes[0]}-{self.cfg.query_sizes[-1]}.pkl", "rb") as f:
          cases_eval_labels_active_classifier, cases_eval_labels_inactive_classifier, cases_eval_inputs_classifier = pickle.load(f)
      
      # extract the queries from the cases:
      self.train_labels_active_classifier = torch.cat(cases_train_labels_active_classifier, dim = 1).squeeze(dim = 0)
      self.train_labels_inactive_classifier = torch.cat(cases_train_labels_inactive_classifier, dim = 1).squeeze(dim = 0)
      self.train_inputs_classifier = torch.cat(cases_train_inputs_classifier, dim = 1).squeeze(dim = 0)
      self.eval_labels_active_classifier = torch.cat(cases_eval_labels_active_classifier, dim = 1).squeeze(dim = 0)
      self.eval_labels_inactive_classifier = torch.cat(cases_eval_labels_inactive_classifier, dim = 1).squeeze(dim = 0)
      self.eval_inputs_classifier = torch.cat(cases_eval_inputs_classifier, dim = 1).squeeze(dim = 0)

      # generate scaler for normalization on train set and save it for usage on further data
      self.scaler = StandardScaler()
      self.scaler.fit(self.train_inputs_classifier)
      current_loc = __file__.rsplit("/",3)[0]
      with open(current_loc + f'/src/learning_selection_policy/normScaler/scaler_task{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_support_size{self.cfg.support_sizes[0]}-{self.cfg.support_sizes[-1]}_query_size{self.cfg.query_sizes[0]}-{self.cfg.query_sizes[-1]}.pkl','wb') as f:
        pickle.dump(self.scaler, f)

      # normalize the data
      self.train_inputs_classifier = torch.from_numpy(self.scaler.transform(self.train_inputs_classifier).astype(np.float32))
      self.eval_inputs_classifier = torch.from_numpy(self.scaler.transform(self.eval_inputs_classifier).astype(np.float32))

      # Create Training Dataset
      self.train_dataset = TrainingDataset(self.train_inputs_classifier,
                                      self.train_labels_active_classifier,
                                      self.train_labels_inactive_classifier)
      self.train_dataloader = DataLoader(self.train_dataset,
                                          batch_size=cfg.batch_size,
                                          shuffle=True)
      self.train_dataloader_notShuffled_oneTaskBatches = DataLoader(self.train_dataset,
                                          batch_size=cfg.batch_size,
                                          shuffle=False)
      # Create Evaluation Dataset
      self.eval_dataset = TrainingDataset(self.eval_inputs_classifier,
                                      self.eval_labels_active_classifier,
                                      self.eval_labels_inactive_classifier)
      self.eval_dataloader = DataLoader(self.eval_dataset,
                                          batch_size=cfg.batch_size,
                                          shuffle=False)

#---------------------------------------------------------------------------------------

    