# How to find find trustworthy predictions.

This repository is based on the repository "Autoregressive activity prediction for low-data drug discovery" from  https://github.com/ml-jku/autoregressive_activity_prediction

 

There you can also find the links and discription for downloading the used data.

 

In src:

The folder implemented_selection_policies contains the experiments for Vanilla MHNfs and Label Cleaning MHNfs.

The folder learning_selection_policy contains the experiments for Learned MHNfs.

The folder Evaluation contains the file for evaluationg the autoregressive model performances.

The folder mhnfs contains the implemented models and their modules.

 

When using this repository create a results folder containing autoreg_inf_fsmol.pkl, performance_tables and plots.

Further in learning_selection_policy create the folders cases and normScaler, and the folder classifiers with classifier_active, classifier_inactive, accuracyScens, aucs, datatable, daucPrs and losses.
