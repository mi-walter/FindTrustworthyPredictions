"""
This file include experiment manager modules.
The one takes the created data and trains two classifiers for the Learned MHNfs.
The other takes the pretrained model and the data module as inputs and performs the
autoregressive inference experiments.

Some parts are taken and adapted from "Autoregressive activity prediction for low-data drug discovery" 
URL https://github.com/ml-jku/autoregressive_activity_prediction
"""

#---------------------------------------------------------------------------------------
# Dependencies
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
from cfg import Config
from dataloader import DataModule_for_autregressive_inference_on_FSMol_testset, DataModule_for_learning_selection_policy_training_with_batches
current_loc = __file__.rsplit("/",3)[0]
import sys
import os
sys.path.append(current_loc)

from src.mhnfs.load_trained_model import MHNfs_inference_module, MHNfs_inference_module_LabelCleaning

#---------------------------------------------------------------------------------------
# Experiment Manager

class ExperimentManager_LearningSelectionPolicy():
    def __init__(self, cfg=Config()):
        np.random.seed(cfg.seed)
        self.cfg = cfg
        self.experiment_seed = 1234
        
    def perform_experiment(self):
        # Train classifiers
        print(f'training with {self.cfg.layers}layers_hidden{self.cfg.hidden_dim}_dropout{self.cfg.dropout}_batch{self.cfg.batch_size}_converge{self.cfg.convergingSteps}steps_lr{self.cfg.learning_rate}_trainTask{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_evalTask{self.cfg.task_ids_eval[0]}-{self.cfg.task_ids_eval[-1]}_{self.cfg.nbr_different_sample_set_eval_cases}')

        # Load the evaluation and traing cases:
        training_data_module = DataModule_for_learning_selection_policy_training_with_batches(self.cfg)

        # generate the classifiers to train:
        if self.cfg.layers == 3: 
            classifier_active = classifierSelectionPolicy3Layer(self.cfg.statistic_features_size, self.cfg.hidden_dim, 1, self.cfg.dropout, self.cfg.device)
            classifier_inactive = classifierSelectionPolicy3Layer(self.cfg.statistic_features_size, self.cfg.hidden_dim, 1, self.cfg.dropout, self.cfg.device)
        elif self.cfg.layers == 6: 
            classifier_active = classifierSelectionPolicy6Layer(self.cfg.statistic_features_size, self.cfg.hidden_dim, 1, self.cfg.dropout, self.cfg.device)
            classifier_inactive = classifierSelectionPolicy6Layer(self.cfg.statistic_features_size, self.cfg.hidden_dim, 1, self.cfg.dropout, self.cfg.device)
        # for loading a pretrained classifier
        """current_loc = __file__.rsplit("/",3)[0]
        classifier_active.load_state_dict(torch.load(current_loc + f'/src/learning_selection_policy/classifiers/classifier_active/classifier_active_{self.cfg.layers}layers_hidden{self.cfg.hidden_dim}_dropout{self.cfg.dropout}_batch{self.cfg.batch_size}_converge{self.cfg.convergingSteps}steps_lr{self.cfg.learning_rate}_trainTask{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_evalTask{self.cfg.task_ids_eval[0]}-{self.cfg.task_ids_eval[-1]}_{self.cfg.nbr_different_sample_set_eval_cases}.pth'))
        classifier_inactive.load_state_dict(torch.load(current_loc + f'/src/learning_selection_policy/classifiers/classifier_inactive/classifier_inactive_{self.cfg.layers}layers_hidden{self.cfg.hidden_dim}_dropout{self.cfg.dropout}_batch{self.cfg.batch_size}_converge{self.cfg.convergingSteps}steps_lr{self.cfg.learning_rate}_trainTask{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_evalTask{self.cfg.task_ids_eval[0]}-{self.cfg.task_ids_eval[-1]}_{self.cfg.nbr_different_sample_set_eval_cases}.pth'))"""
        
        lr = self.cfg.learning_rate
        optimizer_classifier_active = torch.optim.Adam(classifier_active.parameters(), lr=lr)
        optimizer_classifier_inactive = torch.optim.Adam(classifier_inactive.parameters(), lr=lr)
        criterion = nn.BCELoss(reduction='mean')

        print("Evaluating...")
        # evaluate the classifiers on the training set:
        train_loss_active = 0
        train_loss_inactive = 0
        train_active_preds = []
        train_active_labels = []
        train_inactive_preds = []
        train_inactive_labels = []
        counter = 0
        for batch in training_data_module.train_dataloader:
            counter = counter + 1
            inputs_classifier = batch['input'].to(device=classifier_active.device)
            labels_active_classifier = batch['label_active_classifier'].to(device=classifier_active.device)
            classifier_active.eval()
            with torch.no_grad():
                pred = classifier_active(inputs_classifier).squeeze(dim=1)
                train_loss_active = train_loss_active + criterion(pred, labels_active_classifier)
                train_active_preds = train_active_preds + list(pred.detach().cpu().numpy())
                train_active_labels = train_active_labels + list(labels_active_classifier.detach().cpu().numpy())

            inputs_classifier = batch['input'].to(device=classifier_inactive.device)
            labels_inactive_classifier = batch['label_inactive_classifier'].to(device=classifier_inactive.device)
            classifier_inactive.eval()
            with torch.no_grad():
                pred = classifier_inactive(inputs_classifier).squeeze(dim=1)
                train_loss_inactive = train_loss_inactive + criterion(pred, labels_inactive_classifier)
                train_inactive_preds = train_inactive_preds + list(pred.detach().cpu().numpy())
                train_inactive_labels = train_inactive_labels + list(labels_inactive_classifier.detach().cpu().numpy())

        train_loss_active = train_loss_active / counter
        train_loss_inactive = train_loss_inactive / counter

        # calc auc and daucPR on and trainset:
        train_auc_active = roc_auc_score(train_active_labels, train_active_preds)
        train_auc_inactive = roc_auc_score(train_inactive_labels, train_inactive_preds)
        train_daucPr_active = average_precision_score(train_active_labels, train_active_preds)-sum(train_active_labels)/len(train_active_labels)
        train_daucPr_inactive = average_precision_score(train_inactive_labels, train_inactive_preds)-sum(train_inactive_labels)/len(train_inactive_labels)

        print("Mean Trainingloss ActiveClassifier: ", train_loss_active)
        print("Mean Trainingloss InactiveClassifier: ", train_loss_inactive)
        print("AUC Trainingset ActiveClassifier: ", train_auc_active)
        print("AUC Trainingset InactiveClassifier: ", train_auc_inactive)
        print("ΔAUC_Pr Trainingset ActiveClassifier: ", train_daucPr_active)
        print("ΔAUC_Pr Trainingset InactiveClassifier: ", train_daucPr_inactive)

        # evaluate the classifiers on the evaluation cases:
        best_eval_loss_active = 0
        best_eval_loss_inactive = 0
        eval_active_preds = []
        eval_active_labels = []
        eval_inactive_preds = []
        eval_inactive_labels = []
        counter = 0
        for batch in training_data_module.eval_dataloader:
            counter = counter + 1
            inputs_classifier = batch['input'].to(device=classifier_active.device)
            labels_active_classifier = batch['label_active_classifier'].to(device=classifier_active.device)
            classifier_active.eval()
            with torch.no_grad():
                pred = classifier_active(inputs_classifier).squeeze(dim=1)
                best_eval_loss_active = best_eval_loss_active + criterion(pred, labels_active_classifier)
                eval_active_preds = eval_active_preds + list(pred.detach().cpu().numpy())
                eval_active_labels = eval_active_labels + list(labels_active_classifier.detach().cpu().numpy())

            inputs_classifier = batch['input'].to(device=classifier_inactive.device)
            labels_inactive_classifier = batch['label_inactive_classifier'].to(device=classifier_inactive.device)
            classifier_inactive.eval()
            with torch.no_grad():
                pred = classifier_inactive(inputs_classifier).squeeze(dim=1)
                best_eval_loss_inactive = best_eval_loss_inactive + criterion(pred, labels_inactive_classifier)
                eval_inactive_preds = eval_inactive_preds + list(pred.detach().cpu().numpy())
                eval_inactive_labels = eval_inactive_labels + list(labels_inactive_classifier.detach().cpu().numpy())

        best_eval_loss_active = best_eval_loss_active / counter
        best_eval_loss_inactive = best_eval_loss_inactive / counter

        # calc auc and daucPr on and evalset:
        eval_auc_active = roc_auc_score(eval_active_labels, eval_active_preds)
        eval_auc_inactive = roc_auc_score(eval_inactive_labels, eval_inactive_preds)
        best_eval_daucPr_active = average_precision_score(eval_active_labels, eval_active_preds)-sum(eval_active_labels)/len(eval_active_labels)
        best_eval_daucPr_inactive = average_precision_score(eval_inactive_labels, eval_inactive_preds)-sum(eval_inactive_labels)/len(eval_inactive_labels)

        print("Mean Evaluationloss ActiveClassifier: ", best_eval_loss_active)
        print("Mean Evaluationloss InactiveClassifier: ", best_eval_loss_inactive)
        print("AUC Evaluationset ActiveClassifier: ", eval_auc_active)
        print("AUC Evaluationset InactiveClassifier: ", eval_auc_inactive)
        print("ΔAUC_Pr Evaluationset ActiveClassifier: ", best_eval_daucPr_active)
        print("ΔAUC_Pr Evaluationset InactiveClassifier: ", best_eval_daucPr_inactive)
        
        # evaluate accuracy on Scenarios of the evaluation set (not possible for trainset as it is shuffled)
        best_eval_accuracyScen_active = 0
        best_eval_accuracyScen_inactive = 0
        for i in range(int(len(eval_active_preds)/10)):
            ind = np.argmax(eval_active_preds[i*10:(i+1)*10])
            if eval_active_labels[i*10 + ind] == 1:
                best_eval_accuracyScen_active = best_eval_accuracyScen_active + 1
            ind = np.argmax(eval_inactive_preds[i*10:(i+1)*10])
            if eval_inactive_labels[i*10 + ind] == 1:
                best_eval_accuracyScen_inactive = best_eval_accuracyScen_inactive + 1
        best_eval_accuracyScen_active = best_eval_accuracyScen_active / (len(eval_active_preds)/10)
        best_eval_accuracyScen_inactive = best_eval_accuracyScen_inactive / (len(eval_inactive_preds)/10)

        print("Accuracy for scenarios Evaluationset ActiveClassifier: ", best_eval_accuracyScen_active)
        print("Accuracy for scenarios Evaluationset InactiveClassifier: ", best_eval_accuracyScen_inactive)

        # save clasifiers:
        current_loc = __file__.rsplit("/",3)[0]
        torch.save(classifier_active.state_dict(), current_loc + f'/src/learning_selection_policy/classifiers/classifier_active/classifier_active_{self.cfg.layers}layers_hidden{self.cfg.hidden_dim}_dropout{self.cfg.dropout}_batch{self.cfg.batch_size}_converge{self.cfg.convergingSteps}steps_lr{self.cfg.learning_rate}_trainTask{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_evalTask{self.cfg.task_ids_eval[0]}-{self.cfg.task_ids_eval[-1]}_{self.cfg.nbr_different_sample_set_eval_cases}.pth')
        torch.save(classifier_inactive.state_dict(), current_loc + f'/src/learning_selection_policy/classifiers/classifier_inactive/classifier_inactive_{self.cfg.layers}layers_hidden{self.cfg.hidden_dim}_dropout{self.cfg.dropout}_batch{self.cfg.batch_size}_converge{self.cfg.convergingSteps}steps_lr{self.cfg.learning_rate}_trainTask{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_evalTask{self.cfg.task_ids_eval[0]}-{self.cfg.task_ids_eval[-1]}_{self.cfg.nbr_different_sample_set_eval_cases}.pth')
        
        print(f"Classifier Active saved with daucPr {best_eval_daucPr_active}!")
        print(f"Classifier Inactive saved with daucPr {best_eval_daucPr_inactive}!")

        convergingActive = 0
        convergingInactive = 0
        train_losses_active = [train_loss_active]
        train_losses_inactive = [train_loss_inactive]
        eval_losses_active = [best_eval_loss_active]
        eval_losses_inactive = [best_eval_loss_inactive]
        train_aucs_active = [train_auc_active]
        train_aucs_inactive = [train_auc_inactive]
        train_daucPrs_active = [train_daucPr_active]
        train_daucPrs_inactive = [train_daucPr_inactive]
        eval_aucs_active = [eval_auc_active]
        eval_aucs_inactive = [eval_auc_inactive]
        eval_daucPrs_active = [best_eval_daucPr_active]
        eval_daucPrs_inactive = [best_eval_daucPr_inactive]
        eval_accuracyScens_active = [best_eval_accuracyScen_active]
        eval_accuracyScens_inactive = [best_eval_accuracyScen_inactive]

        # Training Epochs:
        for e in range(self.cfg.max_epochs):
            # check for convergance
            if (convergingActive > self.cfg.convergingSteps) and (convergingInactive > self.cfg.convergingSteps):
                print(f"Convergance of Classifier Active for {convergingActive} steps with best evaluation ΔAUC_Pr: {best_eval_daucPr_active}")
                print(f"Convergance of Classifier Inactive for {convergingInactive} steps with best evaluation ΔAUC_Pr: {best_eval_daucPr_inactive}")
                break
            convergingActive = convergingActive + 1
            convergingInactive = convergingInactive + 1

            # plot intermediate plots and save data:
            if (e % 50) == 0:
                # plot loss curve
                self.plot_losses(best_eval_loss_active, best_eval_loss_inactive, train_losses_active, train_losses_inactive, eval_losses_active, eval_losses_inactive, intermediate=True)
                # plot auc curve
                self.plot_aucs(train_aucs_active, train_aucs_inactive, eval_aucs_active, eval_aucs_inactive, metric="auc")
                # plot Δauc-pr curve
                self.plot_aucs(train_daucPrs_active, train_daucPrs_inactive, eval_daucPrs_active, eval_daucPrs_inactive, metric="daucPr")
                # plot accuracyScen curve
                self.plot_accuracyScen(eval_accuracyScens_active, eval_accuracyScens_inactive)
            # save datatable
            current_loc = __file__.rsplit("/",3)[0]
            with open(current_loc + 
                    f'/src/learning_selection_policy/classifiers/datatable/'
                    f'datatable_{self.cfg.layers}layers_hidden{self.cfg.hidden_dim}_dropout{self.cfg.dropout}_batch{self.cfg.batch_size}_converge{self.cfg.convergingSteps}steps_lr{self.cfg.learning_rate}_trainTask{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_evalTask{self.cfg.task_ids_eval[0]}-{self.cfg.task_ids_eval[-1]}_{self.cfg.nbr_different_sample_set_eval_cases}.pkl', 'wb') as f:
                pickle.dump([train_losses_active, train_losses_inactive, eval_losses_active, eval_losses_inactive, train_aucs_active, train_aucs_inactive, eval_aucs_active, eval_aucs_inactive, train_daucPrs_active, train_daucPrs_inactive, eval_daucPrs_active, eval_daucPrs_inactive, eval_accuracyScens_active, eval_accuracyScens_inactive], f)
        
            print(f"Epoch {e}:")
            print(f'training with {self.cfg.layers}layers_hidden{self.cfg.hidden_dim}_dropout{self.cfg.dropout}_batch{self.cfg.batch_size}_converge{self.cfg.convergingSteps}steps_lr{self.cfg.learning_rate}_trainTask{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_evalTask{self.cfg.task_ids_eval[0]}-{self.cfg.task_ids_eval[-1]}_{self.cfg.nbr_different_sample_set_eval_cases}')
            print("Training...")
            # train the classifiers on the training set
            for batch in training_data_module.train_dataloader:
                inputs_classifier = batch['input'].to(device=classifier_active.device)
                labels_active_classifier = batch['label_active_classifier'].to(device=classifier_active.device)
                classifier_active.train()
                optimizer_classifier_active.zero_grad()
                pred = classifier_active(inputs_classifier).squeeze(dim=1)
                loss = criterion(pred, labels_active_classifier)
                loss.backward()
                optimizer_classifier_active.step()

                inputs_classifier = batch['input'].to(device=classifier_inactive.device)
                labels_inactive_classifier = batch['label_inactive_classifier'].to(device=classifier_inactive.device)
                classifier_inactive.train()
                optimizer_classifier_inactive.zero_grad()
                pred = classifier_inactive(inputs_classifier).squeeze(dim=1)
                loss = criterion(pred, labels_inactive_classifier)
                loss.backward()
                optimizer_classifier_inactive.step()

            print("Evaluating...")
            # evaluate the classifiers on the training set:
            train_loss_active = 0
            train_loss_inactive = 0
            train_active_preds = []
            train_active_labels = []
            train_inactive_preds = []
            train_inactive_labels = []
            counter = 0
            for batch in training_data_module.train_dataloader:
                counter = counter + 1
                inputs_classifier = batch['input'].to(device=classifier_active.device)
                labels_active_classifier = batch['label_active_classifier'].to(device=classifier_active.device)
                classifier_active.eval()
                with torch.no_grad():
                    pred = classifier_active(inputs_classifier).squeeze(dim=1)
                    train_loss_active = train_loss_active + criterion(pred, labels_active_classifier)
                    train_active_preds = train_active_preds + list(pred.detach().cpu().numpy())
                    train_active_labels = train_active_labels + list(labels_active_classifier.detach().cpu().numpy())

                inputs_classifier = batch['input'].to(device=classifier_inactive.device)
                labels_inactive_classifier = batch['label_inactive_classifier'].to(device=classifier_inactive.device)
                classifier_inactive.eval()
                with torch.no_grad():
                    pred = classifier_inactive(inputs_classifier).squeeze(dim=1)
                    train_loss_inactive = train_loss_inactive + criterion(pred, labels_inactive_classifier)
                    train_inactive_preds = train_inactive_preds + list(pred.detach().cpu().numpy())
                    train_inactive_labels = train_inactive_labels + list(labels_inactive_classifier.detach().cpu().numpy())

            train_loss_active = train_loss_active / counter
            train_loss_inactive = train_loss_inactive / counter
            train_losses_active.append(train_loss_active)
            train_losses_inactive.append(train_loss_inactive)

            # calc auc and daucPr on trainset:
            train_auc_active = roc_auc_score(train_active_labels, train_active_preds)
            train_auc_inactive = roc_auc_score(train_inactive_labels, train_inactive_preds)
            train_daucPr_active = average_precision_score(train_active_labels, train_active_preds)-sum(train_active_labels)/len(train_active_labels)
            train_daucPr_inactive = average_precision_score(train_inactive_labels, train_inactive_preds)-sum(train_inactive_labels)/len(train_inactive_labels)
            train_aucs_active.append(train_auc_active)
            train_aucs_inactive.append(train_auc_inactive)
            train_daucPrs_active.append(train_daucPr_active)
            train_daucPrs_inactive.append(train_daucPr_inactive)

            print("Mean Trainingloss ActiveClassifier: ", train_loss_active)
            print("Mean Trainingloss InactiveClassifier: ", train_loss_inactive)
            print("AUC Trainingset ActiveClassifier: ", train_auc_active)
            print("AUC Trainingset InactiveClassifier: ", train_auc_inactive)
            print("ΔAUC_Pr Trainingset ActiveClassifier: ", train_daucPr_active)
            print("ΔAUC_Pr Trainingset InactiveClassifier: ", train_daucPr_inactive)
            
            # evaluate the classifiers on the evaluation cases:
            eval_loss_active = 0
            eval_loss_inactive = 0
            eval_active_preds = []
            eval_active_labels = []
            eval_inactive_preds = []
            eval_inactive_labels = []
            counter = 0
            for batch in training_data_module.eval_dataloader:
                counter = counter + 1
                inputs_classifier = batch['input'].to(device=classifier_active.device)
                labels_active_classifier = batch['label_active_classifier'].to(device=classifier_active.device)
                classifier_active.eval()
                with torch.no_grad():
                    pred = classifier_active(inputs_classifier).squeeze(dim=1)
                    eval_loss_active = eval_loss_active + criterion(pred, labels_active_classifier)
                    eval_active_preds = eval_active_preds + list(pred.detach().cpu().numpy())
                    eval_active_labels = eval_active_labels + list(labels_active_classifier.detach().cpu().numpy())

                inputs_classifier = batch['input'].to(device=classifier_inactive.device)
                labels_inactive_classifier = batch['label_inactive_classifier'].to(device=classifier_inactive.device)
                classifier_inactive.eval()
                with torch.no_grad():
                    pred = classifier_inactive(inputs_classifier).squeeze(dim=1)
                    eval_loss_inactive = eval_loss_inactive + criterion(pred, labels_inactive_classifier)
                    eval_inactive_preds = eval_inactive_preds + list(pred.detach().cpu().numpy())
                    eval_inactive_labels = eval_inactive_labels + list(labels_inactive_classifier.detach().cpu().numpy())

            eval_loss_active = eval_loss_active / counter
            eval_loss_inactive = eval_loss_inactive / counter
            eval_losses_active.append(eval_loss_active)
            eval_losses_inactive.append(eval_loss_inactive)

            # calc auc and daucPr on evalset:
            eval_auc_active = roc_auc_score(eval_active_labels, eval_active_preds)
            eval_auc_inactive = roc_auc_score(eval_inactive_labels, eval_inactive_preds)
            eval_daucPr_active = average_precision_score(eval_active_labels, eval_active_preds)-sum(eval_active_labels)/len(eval_active_labels)
            eval_daucPr_inactive = average_precision_score(eval_inactive_labels, eval_inactive_preds)-sum(eval_inactive_labels)/len(eval_inactive_labels)
            eval_aucs_active.append(eval_auc_active)
            eval_aucs_inactive.append(eval_auc_inactive)
            eval_daucPrs_active.append(eval_daucPr_active)
            eval_daucPrs_inactive.append(eval_daucPr_inactive)

            print("Mean Evaluationloss ActiveClassifier: ", eval_loss_active)
            print("Mean Evaluationloss InactiveClassifier: ", eval_loss_inactive)
            print("AUC Evaluationset ActiveClassifier: ", eval_auc_active)
            print("AUC Evaluationset InactiveClassifier: ", eval_auc_inactive)
            print("ΔAUC_Pr Evaluationset ActiveClassifier: ", eval_daucPr_active)
            print("ΔAUC_Pr Evaluationset InactiveClassifier: ", eval_daucPr_inactive)

            # evaluate accuracy on Scenarios of the evaluation set (not possible for trainset as it is shuffled)
            eval_accuracyScen_active = 0
            eval_accuracyScen_inactive = 0
            for i in range(int(len(eval_active_preds)/10)):
                ind = np.argmax(eval_active_preds[i*10:(i+1)*10])
                if eval_active_labels[i*10 + ind] == 1:
                    eval_accuracyScen_active = eval_accuracyScen_active + 1
                ind = np.argmax(eval_inactive_preds[i*10:(i+1)*10])
                if eval_inactive_labels[i*10 + ind] == 1:
                    eval_accuracyScen_inactive = eval_accuracyScen_inactive + 1
            eval_accuracyScen_active = eval_accuracyScen_active / (len(eval_active_preds)/10)
            eval_accuracyScen_inactive = eval_accuracyScen_inactive / (len(eval_inactive_preds)/10)
            eval_accuracyScens_active.append(eval_accuracyScen_active)
            eval_accuracyScens_inactive.append(eval_accuracyScen_inactive)

            print("Accuracy for scenarios Evaluationset ActiveClassifier: ", eval_accuracyScen_active)
            print("Accuracy for scenarios Evaluationset InactiveClassifier: ", eval_accuracyScen_inactive)

            # save classifier when improved on daucPr
            if eval_daucPr_active > best_eval_daucPr_active:
                best_eval_daucPr_active = eval_daucPr_active
                current_loc = __file__.rsplit("/",3)[0]
                torch.save(classifier_active.state_dict(), current_loc + f'/src/learning_selection_policy/classifiers/classifier_active/classifier_active_{self.cfg.layers}layers_hidden{self.cfg.hidden_dim}_dropout{self.cfg.dropout}_batch{self.cfg.batch_size}_converge{self.cfg.convergingSteps}steps_lr{self.cfg.learning_rate}_trainTask{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_evalTask{self.cfg.task_ids_eval[0]}-{self.cfg.task_ids_eval[-1]}_{self.cfg.nbr_different_sample_set_eval_cases}.pth')
                print(f"Classifier Active saved with daucPr {best_eval_daucPr_active}!")
                convergingActive = 0

            if eval_daucPr_inactive > best_eval_daucPr_inactive:
                best_eval_daucPr_inactive = eval_daucPr_inactive
                current_loc = __file__.rsplit("/",3)[0]
                torch.save(classifier_inactive.state_dict(), current_loc + f'/src/learning_selection_policy/classifiers/classifier_inactive/classifier_inactive_{self.cfg.layers}layers_hidden{self.cfg.hidden_dim}_dropout{self.cfg.dropout}_batch{self.cfg.batch_size}_converge{self.cfg.convergingSteps}steps_lr{self.cfg.learning_rate}_trainTask{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_evalTask{self.cfg.task_ids_eval[0]}-{self.cfg.task_ids_eval[-1]}_{self.cfg.nbr_different_sample_set_eval_cases}.pth')
                print(f"Classifier Inactive saved with daucPr {best_eval_daucPr_inactive}!")
                convergingInactive = 0
            
            # save classifier as traininglast
            torch.save(classifier_active.state_dict(), current_loc + f'/src/learning_selection_policy/classifiers/classifier_active/traininglast_classifier_active_{self.cfg.layers}layers_hidden{self.cfg.hidden_dim}_dropout{self.cfg.dropout}_batch{self.cfg.batch_size}_converge{self.cfg.convergingSteps}steps_lr{self.cfg.learning_rate}_trainTask{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_evalTask{self.cfg.task_ids_eval[0]}-{self.cfg.task_ids_eval[-1]}_{self.cfg.nbr_different_sample_set_eval_cases}.pth')
            torch.save(classifier_inactive.state_dict(), current_loc + f'/src/learning_selection_policy/classifiers/classifier_inactive/traininglast_classifier_inactive_{self.cfg.layers}layers_hidden{self.cfg.hidden_dim}_dropout{self.cfg.dropout}_batch{self.cfg.batch_size}_converge{self.cfg.convergingSteps}steps_lr{self.cfg.learning_rate}_trainTask{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_evalTask{self.cfg.task_ids_eval[0]}-{self.cfg.task_ids_eval[-1]}_{self.cfg.nbr_different_sample_set_eval_cases}.pth')

        print(f'training with {self.cfg.layers}layers_hidden{self.cfg.hidden_dim}_dropout{self.cfg.dropout}_batch{self.cfg.batch_size}_converge{self.cfg.convergingSteps}steps_lr{self.cfg.learning_rate}_trainTask{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_evalTask{self.cfg.task_ids_eval[0]}-{self.cfg.task_ids_eval[-1]}_{self.cfg.nbr_different_sample_set_eval_cases} done')
        
        # plot loss curve
        self.plot_losses(best_eval_loss_active, best_eval_loss_inactive, train_losses_active, train_losses_inactive, eval_losses_active, eval_losses_inactive, intermediate=False)
        # plot auc curve
        self.plot_aucs(train_aucs_active, train_aucs_inactive, eval_aucs_active, eval_aucs_inactive, metric="auc")
        # plot Δauc-pr curve
        self.plot_aucs(train_daucPrs_active, train_daucPrs_inactive, eval_daucPrs_active, eval_daucPrs_inactive, metric="daucPr")
        # plot accuracyScen curve
        self.plot_accuracyScen(eval_accuracyScens_active, eval_accuracyScens_inactive)

        # save datatable
        current_loc = __file__.rsplit("/",3)[0]
        with open(current_loc + 
                f'/src/learning_selection_policy/classifiers/datatable/'
                f'datatable_{self.cfg.layers}layers_hidden{self.cfg.hidden_dim}_dropout{self.cfg.dropout}_batch{self.cfg.batch_size}_converge{self.cfg.convergingSteps}steps_lr{self.cfg.learning_rate}_trainTask{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_evalTask{self.cfg.task_ids_eval[0]}-{self.cfg.task_ids_eval[-1]}_{self.cfg.nbr_different_sample_set_eval_cases}.pkl', 'wb') as f:
            pickle.dump([train_losses_active, train_losses_inactive, eval_losses_active, eval_losses_inactive, train_aucs_active, train_aucs_inactive, eval_aucs_active, eval_aucs_inactive, train_daucPrs_active, train_daucPrs_inactive, eval_daucPrs_active, eval_daucPrs_inactive, eval_accuracyScens_active, eval_accuracyScens_inactive], f)


    def plot_losses(self, best_eval_loss_active, best_eval_loss_inactive, train_losses_active, train_losses_inactive, eval_losses_active, eval_losses_inactive, intermediate=True):
        # plot loss curve
        train_losses_active = torch.stack(train_losses_active, dim=0).detach().cpu().numpy()
        train_losses_inactive = torch.stack(train_losses_inactive, dim=0).detach().cpu().numpy()
        eval_losses_active = torch.stack(eval_losses_active, dim=0).detach().cpu().numpy()
        eval_losses_inactive = torch.stack(eval_losses_inactive, dim=0).detach().cpu().numpy()

        title = f"Losses (train_active: blue, train_inactive: red, eval_active: purple, eval_inactive: green)\nbest eval_active = {best_eval_loss_active}, best eval_inactive = {best_eval_loss_inactive}"
        x_label = 'epochs'
        y_label = 'loss'
        plt.figure(figsize=(15,7))

        x_values = np.arange(0, len(train_losses_active), 1)
        y_values = train_losses_active
        plt.plot(x_values, y_values, color='tab:blue')

        y_values = train_losses_inactive
        plt.plot(x_values, y_values, color='tab:red')

        y_values = eval_losses_active
        plt.plot(x_values, y_values, color='tab:purple')

        y_values = eval_losses_inactive
        plt.plot(x_values, y_values, color='tab:green')

        plt.grid()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        current_loc = __file__.rsplit("/",3)[0]
        if intermediate:
            plt.savefig(current_loc +
                    f'/src/learning_selection_policy/classifiers/losses/'
                    f'losses_{self.cfg.layers}layers_hidden{self.cfg.hidden_dim}_dropout{self.cfg.dropout}_batch{self.cfg.batch_size}_converge{self.cfg.convergingSteps}steps_lr{self.cfg.learning_rate}_trainTask{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_evalTask{self.cfg.task_ids_eval[0]}-{self.cfg.task_ids_eval[-1]}_{self.cfg.nbr_different_sample_set_eval_cases}.pdf',
                    bbox_inches='tight',pad_inches=0.1)
        else:
            plt.savefig(current_loc +
                    f'/src/learning_selection_policy/classifiers/losses/'
                    f'losses_{self.cfg.layers}layers_hidden{self.cfg.hidden_dim}_dropout{self.cfg.dropout}_batch{self.cfg.batch_size}_converge{self.cfg.convergingSteps}steps_lr{self.cfg.learning_rate}_trainTask{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_evalTask{self.cfg.task_ids_eval[0]}-{self.cfg.task_ids_eval[-1]}_{self.cfg.nbr_different_sample_set_eval_cases}_best_eval_active_{best_eval_loss_active}_best_eval_inactive_{best_eval_loss_inactive}.pdf',
                    bbox_inches='tight',pad_inches=0.1)
        plt.close()

    def plot_aucs(self, train_aucs_active, train_aucs_inactive, eval_aucs_active, eval_aucs_inactive, metric="auc"):
        # plot auc or daucPR curve
        title = f"{metric}s (train_active: blue, train_inactive: red, eval_active: purple, eval_inactive: green)"
        x_label = 'epochs'
        y_label = f'{metric}s'
        plt.figure(figsize=(15,7))
        
        x_values = np.arange(0, len(train_aucs_active), 1)
        y_values = train_aucs_active
        plt.plot(x_values, y_values, color='tab:blue')

        y_values = train_aucs_inactive
        plt.plot(x_values, y_values, color='tab:red')

        y_values = eval_aucs_active
        plt.plot(x_values, y_values, color='tab:purple')

        y_values = eval_aucs_inactive
        plt.plot(x_values, y_values, color='tab:green')

        plt.grid()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        current_loc = __file__.rsplit("/",3)[0]
        plt.savefig(current_loc +
                f'/src/learning_selection_policy/classifiers/{metric}s/'
                f'{metric}s_{self.cfg.layers}layers_hidden{self.cfg.hidden_dim}_dropout{self.cfg.dropout}_batch{self.cfg.batch_size}_converge{self.cfg.convergingSteps}steps_lr{self.cfg.learning_rate}_trainTask{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_evalTask{self.cfg.task_ids_eval[0]}-{self.cfg.task_ids_eval[-1]}_{self.cfg.nbr_different_sample_set_eval_cases}.pdf',
                bbox_inches='tight',pad_inches=0.1)
        plt.close()
    
    def plot_accuracyScen(self, eval_accuracyScens_active, eval_accuracyScens_inactive):
        # plot accuracy of Scenarios curve
        title = f"Accuracy of selecting the right sample of the scenarios (eval_active: purple, eval_inactive: green)"
        x_label = 'epochs'
        y_label = f'accuracies'
        plt.figure(figsize=(15,7))
        
        x_values = np.arange(0, len(eval_accuracyScens_active), 1)
        y_values = eval_accuracyScens_active
        plt.plot(x_values, y_values, color='tab:purple')

        y_values = eval_accuracyScens_inactive
        plt.plot(x_values, y_values, color='tab:green')

        plt.grid()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        current_loc = __file__.rsplit("/",3)[0]
        plt.savefig(current_loc +
                f'/src/learning_selection_policy/classifiers/accuracyScens/'
                f'accuracyScens_{self.cfg.layers}layers_hidden{self.cfg.hidden_dim}_dropout{self.cfg.dropout}_batch{self.cfg.batch_size}_converge{self.cfg.convergingSteps}steps_lr{self.cfg.learning_rate}_trainTask{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_evalTask{self.cfg.task_ids_eval[0]}-{self.cfg.task_ids_eval[-1]}_{self.cfg.nbr_different_sample_set_eval_cases}.pdf',
                bbox_inches='tight',pad_inches=0.1)
        plt.close()

        
class ExperimentManager_AutoregressiveInference_Learned():
    def __init__(self, cfg=Config()):
        np.random.seed(cfg.seed)
        self.modelMHNfs = MHNfs_inference_module(device=cfg.device)  # wrapper for the pretrained MHNfs model
        self.modelLabelCleaning = MHNfs_inference_module_LabelCleaning(device=cfg.device)  # wrapper for the pretrained MHNfs model
        self.cfg = cfg
        self.experiment_seed = 1234
        self.trainlast = ''
        if self.cfg.usetrainlast:
            self.trainlast = 'traininglast_'

        task_ids_file = np.load(cfg.data_path + "task_ids.npy")
        task_ids_unique = np.unique(task_ids_file)
        self.experiment_task_ids = list(range(len(task_ids_unique)))  # 40 for eval 157 for test
        self.experiment_rerun_seeds = [8915, 1318, 7221, 7540,  664, 6137, 6833, 8471,
                                    9449, 7322]
        
    def perform_experiment(self):
        # Evaluate perforance of classifiers:
        # load best models:
        if self.cfg.layers == 3:
            classifier_active = classifierSelectionPolicy3Layer(self.cfg.statistic_features_size, self.cfg.hidden_dim, 1, self.cfg.dropout, self.cfg.device)
            classifier_inactive = classifierSelectionPolicy3Layer(self.cfg.statistic_features_size, self.cfg.hidden_dim, 1, self.cfg.dropout, self.cfg.device)
        elif self.cfg.layers == 6:
            classifier_active = classifierSelectionPolicy6Layer(self.cfg.statistic_features_size, self.cfg.hidden_dim, 1, self.cfg.dropout, self.cfg.device)
            classifier_inactive = classifierSelectionPolicy6Layer(self.cfg.statistic_features_size, self.cfg.hidden_dim, 1, self.cfg.dropout, self.cfg.device)
        current_loc = __file__.rsplit("/",3)[0]
        classifier_active.load_state_dict(torch.load(current_loc + f'/src/learning_selection_policy/classifiers/classifier_active/{self.trainlast}classifier_active_{self.cfg.layers}layers_hidden{self.cfg.hidden_dim}_dropout{self.cfg.dropout}_batch{self.cfg.batch_size}_converge{self.cfg.convergingSteps}steps_lr{self.cfg.learning_rate}_trainTask{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_evalTask{self.cfg.task_ids_eval[0]}-{self.cfg.task_ids_eval[-1]}_{self.cfg.nbr_different_sample_set_eval_cases}.pth'))
        classifier_inactive.load_state_dict(torch.load(current_loc + f'/src/learning_selection_policy/classifiers/classifier_inactive/{self.trainlast}classifier_inactive_{self.cfg.layers}layers_hidden{self.cfg.hidden_dim}_dropout{self.cfg.dropout}_batch{self.cfg.batch_size}_converge{self.cfg.convergingSteps}steps_lr{self.cfg.learning_rate}_trainTask{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_evalTask{self.cfg.task_ids_eval[0]}-{self.cfg.task_ids_eval[-1]}_{self.cfg.nbr_different_sample_set_eval_cases}.pth'))

        print(f'Evaluate performance for {self.cfg.layers}layers_hidden{self.cfg.hidden_dim}_dropout{self.cfg.dropout}_batch{self.cfg.batch_size}_converge{self.cfg.convergingSteps}steps_lr{self.cfg.learning_rate}_trainTask{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_evalTask{self.cfg.task_ids_eval[0]}-{self.cfg.task_ids_eval[-1]}_{self.cfg.nbr_different_sample_set_eval_cases}_onTask{self.experiment_task_ids[0]}-{self.experiment_task_ids[-1]}')
        print(f"usetrainlast = {self.cfg.usetrainlast}")

        auc_dict = dict()
        dauc_pr_dict = dict()
        
        failsActives = np.zeros((len(self.experiment_task_ids), len(self.experiment_rerun_seeds), 31)) # tasks x reruns x iterations
        failsInactives = np.zeros((len(self.experiment_task_ids), len(self.experiment_rerun_seeds), 31)) # tasks x reruns x iterations

        for task_increment, task in enumerate(self.experiment_task_ids):
            print(f'for {self.cfg.results_path}:')
            print(f'Processing task {task} ...')
            data_module = DataModule_for_autregressive_inference_on_FSMol_testset(task,
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
                        
                        predictions = self.modelMHNfs.predict(
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

                    # generate the inputs for the classifier:
                    query_input = torch.from_numpy(data_module.support_candidate_pool_dataloader.dataset[:]['inputs']).unsqueeze(dim=1)
                    shuffled_indices = np.arange(query_input.shape[0])
                    np.random.shuffle(shuffled_indices)
                    query_input = query_input[shuffled_indices, :, :]

                    inputs_classifier = calc_inputs(support_actives_input, support_inactives_input, torch.from_numpy(data_module.support_candidate_pool_dataloader.dataset[:]['inputs']).unsqueeze(dim=1), self.modelMHNfs, self.modelLabelCleaning, self.cfg.nbr_support_set_candidates-self.cfg.nbr_ss_start-i)
                    
                    # normalize the data 
                    current_loc = __file__.rsplit("/",3)[0]
                    with open(current_loc + f'/src/learning_selection_policy/normScaler/scaler_task{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_support_size{self.cfg.support_sizes[0]}-{self.cfg.support_sizes[-1]}_query_size{self.cfg.query_sizes[0]}-{self.cfg.query_sizes[-1]}.pkl','rb') as f:
                        scaler = pickle.load(f)
                    inputs_classifier = torch.from_numpy(scaler.transform(inputs_classifier.squeeze(dim=0)).astype(np.float32)).unsqueeze(dim=0)
                    
                    # create predictions for the active and inactive candidate
                    inputs_classifier = inputs_classifier.to(device=classifier_active.device)
                    classifier_active.eval()
                    pred_act = classifier_active(inputs_classifier).squeeze(dim=2).detach().cpu().numpy()

                    inputs_classifier = inputs_classifier.to(device=classifier_inactive.device)
                    classifier_inactive.eval()
                    pred_inact = classifier_inactive(inputs_classifier).squeeze(dim=2).detach().cpu().numpy()

                    # Select candidates for augmenting support set
                    raw_active_cand_id = shuffled_indices[np.argmax(pred_act)]
                    raw_inactive_cand_id = shuffled_indices[np.argmax(pred_inact)]
                    # avoid choice of the same sample (for this the shuffled_indices are needed to keep randomness)
                    if raw_active_cand_id == raw_inactive_cand_id:
                        raw_active_cand_id = shuffled_indices[np.max(np.argsort(pred_act, axis=1)[:, -2])]
                        if raw_active_cand_id == raw_inactive_cand_id: # case for just 2 left and both have the same prediction value
                            raw_active_cand_id = shuffled_indices[np.max(np.argsort(pred_act, axis=1)[:, -1])]
                        print(f"Prediction changed for active from shuffled {np.argmax(pred_act)} to {np.max(np.argsort(pred_act, axis=1)[:, -2])}")

                    # analysizing the methode for tracking errors of iteration steps
                    if data_module.support_candidate_pool[raw_active_cand_id] not in data_module.mol_ids_active_ss_candidates:
                        failsActives[task_increment, self.experiment_rerun_seeds.index(seed), i] = failsActives[task_increment, self.experiment_rerun_seeds.index(seed), i] + 1
                    if data_module.support_candidate_pool[raw_inactive_cand_id] not in data_module.mol_ids_inactive_ss_candidates:
                        failsInactives[task_increment, self.experiment_rerun_seeds.index(seed), i] = failsInactives[task_increment, self.experiment_rerun_seeds.index(seed), i] + 1

                    # augmente support set with selected candidates
                    data_module.add_candidate_to_support_set_and_remove_from_pool(
                        raw_active_cand_id,
                        raw_inactive_cand_id
                    )
            
                aucs_rerun_dict[seed] = aucs
                dauc_prs_rerun_dict[seed] = dauc_prs
            
            auc_dict[task] = aucs_rerun_dict
            dauc_pr_dict[task] = dauc_prs_rerun_dict
        
        # Save results
        current_loc = __file__.rsplit("/",3)[0]
        with open(current_loc + self.cfg.results_path + "autoreg_inf_fsmol.pkl", 'wb') as f:
            pickle.dump([auc_dict, dauc_pr_dict], f)
        print("performance data saved!")

        # plot and print statistics of wrongly added samples over tasks
        failsActivesMean = failsActives.mean(axis=1)
        failsActivesStd = failsActives.std(axis=1)
        failsInactivesMean = failsInactives.mean(axis=1)
        failsInactivesStd = failsInactives.std(axis=1)
        if len(self.experiment_task_ids) == 1:
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
            plt.title(f'Wrongly added actives (blue) and inactives (red) (mean over reruns) of task {self.experiment_task_ids[0]}')
            plt.ylabel(f'ration of wrongly added actives/inactives')
            plt.xlabel(f'iteration')
            current_loc = __file__.rsplit("/",3)[0]
            plt.savefig(current_loc + self.cfg.results_path +
                        f'wrongly_added_actives_and_inactives_task{self.experiment_task_ids[0]}.pdf',
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

        print(f'Evaluate performance for {self.cfg.layers}layers_hidden{self.cfg.hidden_dim}_dropout{self.cfg.dropout}_batch{self.cfg.batch_size}_converge{self.cfg.convergingSteps}steps_lr{self.cfg.learning_rate}_trainTask{self.cfg.task_ids_train[0]}-{self.cfg.task_ids_train[-1]}_{self.cfg.nbr_different_sample_set_train_cases}_evalTask{self.cfg.task_ids_eval[0]}-{self.cfg.task_ids_eval[-1]}_{self.cfg.nbr_different_sample_set_eval_cases}_onTask{self.experiment_task_ids[0]}-{self.experiment_task_ids[-1]} done')
        print(f"usetrainlast = {self.cfg.usetrainlast}")

class classifierSelectionPolicy3Layer(nn.Module):
    """
    classifier used for learning the selection policies
    """
    def __init__(self, in_dim: int = 10, hidden_dim: int = 20, out_dim: int = 1, dropout: float = 0.2, device: str = 'cpu'):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        if device == 'gpu':
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.initialize()
      
    def initialize(self):
        """nn.init.xavier_uniform(self.linear1.weight.data)
        self.linear1.bias.data.zero_()
        nn.init.xavier_uniform(self.linear2.weight.data)
        self.linear2.bias.data.zero_()
        nn.init.xavier_uniform(self.linear3.weight.data)
        self.linear3.bias.data.zero_()"""
        self = self.to(self.device)
    
    def forward(self, x):
        #x = self.dropout(x)
        pred = self.linear1(x)
        pred = self.relu(pred)
        pred = self.dropout(pred)
        pred = self.linear2(pred)
        pred = self.relu(pred)
        pred = self.dropout(pred)
        pred = self.linear3(pred)
        pred = self.sigmoid(pred)
        return pred

class classifierSelectionPolicy6Layer(nn.Module):
    """
    classifier used for learning the selection policies
    """
    def __init__(self, in_dim: int = 10, hidden_dim: int = 20, out_dim: int = 1, dropout: float = 0.2, device: str = 'cpu'):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        if device == 'gpu':
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.AlphaDropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()
        self.initialize()
      
    def initialize(self):
        nn.init.normal_(self.linear1.weight, mean=0.0, std=torch.sqrt(torch.tensor([1.0]) / self.linear1.in_features).numpy()[0])
        nn.init.zeros_(self.linear1.bias)
        nn.init.normal_(self.linear2.weight, mean=0.0, std=torch.sqrt(torch.tensor([1.0]) / self.linear2.in_features).numpy()[0])
        nn.init.zeros_(self.linear2.bias)
        nn.init.normal_(self.linear3.weight, mean=0.0, std=torch.sqrt(torch.tensor([1.0]) / self.linear3.in_features).numpy()[0])
        nn.init.zeros_(self.linear3.bias)
        nn.init.normal_(self.linear4.weight, mean=0.0, std=torch.sqrt(torch.tensor([1.0]) / self.linear4.in_features).numpy()[0])
        nn.init.zeros_(self.linear4.bias)
        nn.init.normal_(self.linear5.weight, mean=0.0, std=torch.sqrt(torch.tensor([1.0]) / self.linear5.in_features).numpy()[0])
        nn.init.zeros_(self.linear5.bias)
        nn.init.normal_(self.linear6.weight, mean=0.0, std=torch.sqrt(torch.tensor([1.0]) / self.linear6.in_features).numpy()[0])
        nn.init.zeros_(self.linear6.bias)
        self = self.to(self.device)
    
    def forward(self, x):
        #x = self.dropout(x)
        pred = self.linear1(x)
        pred = self.selu(pred)
        pred = self.dropout(pred)
        pred = self.linear2(pred)
        pred = self.selu(pred)
        pred = self.dropout(pred)
        pred = self.linear3(pred)
        pred = self.selu(pred)
        pred = self.dropout(pred)
        pred = self.linear4(pred)
        pred = self.selu(pred)
        pred = self.dropout(pred)
        pred = self.linear5(pred)
        pred = self.selu(pred)
        pred = self.dropout(pred)
        pred = self.linear6(pred)
        pred = self.sigmoid(pred)
        return pred

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

    # Run individual experiments:
    #experiment_manager = ExperimentManager_LearningSelectionPolicy()
    experiment_manager = ExperimentManager_AutoregressiveInference_Learned()
    experiment_manager.perform_experiment()