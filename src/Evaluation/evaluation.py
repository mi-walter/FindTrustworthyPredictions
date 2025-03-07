"""
This file evaluates the the autoregressive inference experiment.
Some parts are taken and adapted from "Autoregressive activity prediction for low-data drug discovery" 
URL https://github.com/ml-jku/autoregressive_activity_prediction
"""

#---------------------------------------------------------------------------------------
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import ranksums

#---------------------------------------------------------------------------------------
def evaluate_experiment(experiment_name, compare="", dir=""):
    #-----------------------------------------------------------------------------------
    # Helper functions
    def load_results(experiment_name, compare="", dir=""):
        if dir == "":
            dir ="results"
        current_loc = __file__.rsplit("/",3)[0]
        #current_loc = os.getcwd().rsplit("/",2)[0] 
        if experiment_name == 'semi_supervised':
            with open(current_loc + '/' + dir + '/autoreg_inf_fsmol.pkl',
                    'rb') as f:
                aucs, dauc_prs = pickle.load(f)
            if compare != "":
                with open(current_loc + '/' + compare + '/autoreg_inf_fsmol.pkl',
                    'rb') as f:
                    aucsBase, dauc_prsBase = pickle.load(f)
        else:
            raise ValueError('Invalid experiment name')
        
        if compare != "":
            return aucs, dauc_prs, aucsBase, dauc_prsBase

        return aucs, dauc_prs
    
    def create_performance_array(performance_dict:dict) -> np.array:
        """
        The raw results are stored in a nested dictionary where the first level of keys
        corresponds to the task id and the second level to the reruns with different
        seeds. For each (task-id, rerun) pair, the performance values are stored in a
        list.
        This function creates an array with the performance values:
        - The first dimensions corresponds to the task id
        - The second dimension corresponds to the reruns
        - The third dimension corresponds to the different support set sizes
        """
        
        performance_values = np.zeros((157,10,32)) # test 157 validation 40 #TODO
        # Compute mean performances across reruns
        for task_nbr, task_id in enumerate(performance_dict.keys()):
            # Filter relevant values
            rerun_dict = performance_dict[task_id]
            
            for rerun_nbr,rerun_key in enumerate(rerun_dict.keys()):
                values = rerun_dict[rerun_key]
                
                performance_values[task_nbr,rerun_nbr,:] = values
        
        return performance_values
            
    def plot_mean_performance_across_tasks(performance_values:np.array,
                                                metric:['AUC', 'ΔAUC-PR'],
                                                experiment_name,
                                                compare_performance_values:np.array = None,
                                                dir = ""
        ):
        """
        This functions evalueates the performance of the model across all test tasks
        (analogous to the FS-Mol main benchmark experiment).
        mean performance:
        * Firstly the mean for a task is computed across the different reruns
        * Eventually the mean performance across tasks is computed
        """
        if dir == "":
            dir = "results"

        # Compute mean performance across reruns
        mean_rerun_performance = np.mean(performance_values, axis=1)
        
        # Compute mean and std across reruns
        mean_performance = np.mean(mean_rerun_performance, axis=0)
        std_performance = np.std(mean_rerun_performance, axis=0)
        
        # Create plot
        title = f"MHNfs performance [{metric}] with increasing support set"
        x_label = 'Nbr of both actives and inactives in the support set'
        y_label = f'{metric} (mean across test tasks)'
        
        x_values = np.arange(1, len(mean_performance)+1, 1)
        y_values = mean_performance
        y_upper_bound = mean_performance + std_performance
        y_lower_bound = mean_performance - std_performance
        
        plt.figure(figsize=(15,7))
        plt.plot(x_values, y_values, color='tab:blue')
        plt.fill_between(x_values,
                         y_lower_bound,
                         y_upper_bound,
                         color='tab:blue',
                         alpha=0.3)

        if compare_performance_values is not None:
            # Compute mean performance across reruns
            mean_rerun_compare_performance = np.mean(compare_performance_values, axis=1)
        
            # Compute mean and std across reruns
            mean_compare_performance = np.mean(mean_rerun_compare_performance, axis=0)
            std_compare_performance = np.std(mean_rerun_compare_performance, axis=0)

            x_values = np.arange(1, len(mean_performance)+1, 1)
            y_values = mean_compare_performance
            y_upper_bound = mean_compare_performance + std_compare_performance
            y_lower_bound = mean_compare_performance - std_compare_performance
        
            plt.plot(x_values, y_values, color='tab:red')
            plt.fill_between(x_values,
                             y_lower_bound,
                             y_upper_bound,
                             color='tab:red',
                             alpha=0.3)

        plt.grid()
        plt.xticks(np.arange(1, len(mean_performance)+1, 1))
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        current_loc = __file__.rsplit("/",3)[0]
        plt.savefig(current_loc +
                    f'/{dir}/plots/{experiment_name}/'
                    f'{metric}_mean_performance_across_tasks.pdf',
                    bbox_inches='tight',pad_inches=0.1)
        plt.close()
        
    def plot_mean_performance_across__reruns(
            performance_values:np.array,                                   
            metric:['AUC', 'ΔAUC-PR'],
            experiment_name='str',
            compare_performance_values:np.array = None,
            dir = ""
        ):
        """
        This functions evalueates the performance of the model across all test tasks
        (analogous to the FS-Mol main benchmark experiment).
        mean performance:
        * Firstly the mean across tasks, then the mean across reruns is computed
        """
        if dir == "":
            dir = "results"
        
        # Compute mean performance across tasks
        mean_task_performance = np.mean(performance_values, axis=0)
        #mean_task_performance = np.median(performance_values, axis=0)
        
        # Compute mean and std performance across reruns
        mean_performance = np.mean(mean_task_performance, axis=0)
        std_performance = np.std(mean_task_performance, axis=0)
        
        # Create plot
        title = f"MHNfs performance [{metric}] with increasing support set"
        x_label = 'Nbr of autoregressive inference iterations (1+/1- added in each round)'
        y_label = f'{metric}'
        
        x_values = np.arange(0, len(mean_performance), 1)
        y_values = mean_performance
        y_upper_bound = mean_performance + std_performance
        y_lower_bound = mean_performance - std_performance
        
        plt.figure(figsize=(10,2))
        
        plt.plot(x_values, y_values, color='#f1bc31') # yellow
        plt.fill_between(x_values,
                         y_lower_bound,
                         y_upper_bound,
                         color='#f1bc31',
                         alpha=0.3)
        
        if compare_performance_values is not None:
            # Compute mean performance across tasks
            mean_task_compare_performance = np.mean(compare_performance_values, axis=0)
            #mean_task_compare_performance = np.median(compare_performance_values, axis=0)
        
            # Compute mean and std performance across reruns
            mean_compare_performance = np.mean(mean_task_compare_performance, axis=0)
            std_compare_performance = np.std(mean_task_compare_performance, axis=0)
        
            x_values = np.arange(0, len(mean_performance), 1)
            y_values = mean_compare_performance
            y_upper_bound = mean_compare_performance + std_compare_performance
            y_lower_bound = mean_compare_performance - std_compare_performance
        
            plt.plot(x_values, y_values, color='#4fb0bf')  # blue
            plt.fill_between(x_values,
                             y_lower_bound,
                             y_upper_bound,
                             color='#4fb0bf',
                             alpha=0.3)
        
        plt.grid()
        plt.xticks(np.arange(0, len(mean_performance), 1))
        plt.title(title)
        plt.xlim(0,31)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        current_loc = __file__.rsplit("/",3)[0]
        plt.savefig(current_loc +
                    f'/{dir}/plots/{experiment_name}/'
                    f'{metric}_mean_performance_across_reruns.png',
                    bbox_inches='tight',pad_inches=0.1)
        plt.close()
         
    def create__performance_table(performance_values:np.array,
                                    metric:['AUC', 'ΔAUC-PR'],
                                    experiment_name=str,
                                    dir = ""
        ):
        """
        This function creates a table with the relative gain in performance for each
        support set size.
        """
        if dir == "":
            dir = "results"

        # Compute mean performance across tasks
        mean_task_performance = np.mean(performance_values, axis=0)
        
        # Compute mean and std across reruns
        mean_performance = np.mean(mean_task_performance, axis=0)
        std_performance = np.std(mean_task_performance, axis=0)
        
        # Compute relative gain
        reference = np.broadcast_to(
            mean_task_performance[:,0].reshape(-1,1),
            (mean_task_performance.shape[0],32))
        relative_gain = mean_task_performance - reference
        mean_relative_gain = np.mean(relative_gain, axis=0)
        std_relative_gain = np.std(relative_gain, axis=0)
        
        # Create table
        df = pd.DataFrame({'autoreg-iteration':list(range(32)),
                           'autoreg-inf-mean': mean_performance,
                           'autoreg-inf-std': std_performance,
                           'mean-gain': mean_relative_gain,
                           'std-gain': std_relative_gain})
        
        current_loc = __file__.rsplit("/",3)[0]
        #current_loc = os.getcwd().rsplit("/",2)[0] 
        df.to_csv(
            current_loc +
            f'/{dir}/performance_tables/{experiment_name}_{metric}.csv')
    
    def create_paired_Wilcoxon_rank_sum_test(performance_values:np.array, 
                                            compare_performance_values:np.array,
                                            metric:['AUC', 'ΔAUC-PR'],
                                            dir = ""
        ):
        """
        This function creates the paired Wilcoxon statistics of the performances
        which should be compared with each other.
        """
        if dir == "":
            dir = "results"

        # Compute mean performance across tasks
        mean_task_performance = np.mean(performance_values, axis=0)
        mean_task_compare_performance = np.mean(compare_performance_values, axis=0)
        statisticsL = []
        pvaluesL = []
        statisticsD = []
        pvaluesD = []
        statisticsG = []
        pvaluesG = []
        results = []
        for iter in range(mean_task_performance.shape[1]):
            statistic, pvalue = ranksums(mean_task_compare_performance[:, iter], mean_task_performance[:, iter], alternative='less')
            statisticsL.append(statistic)
            pvaluesL.append(pvalue)
            statistic, pvalue = ranksums(mean_task_compare_performance[:, iter], mean_task_performance[:, iter], alternative='two-sided')
            statisticsD.append(statistic)
            pvaluesD.append(pvalue)
            statistic, pvalue = ranksums(mean_task_compare_performance[:, iter], mean_task_performance[:, iter], alternative='greater')
            statisticsG.append(statistic)
            pvaluesG.append(pvalue)
            result = "no difference"
            if pvaluesD[iter] < 0.05 and pvaluesL[iter] < 0.05:
                result = "greater"
            elif pvaluesD[iter] < 0.05 and pvaluesG[iter] < 0.05:
                result = "lower"
            elif ((pvaluesD[iter] < 0.05) ^ (pvaluesL[iter] < 0.05) ^ (pvaluesG[iter] < 0.05)) or (pvaluesL[iter] < 0.05 and pvaluesG[iter] < 0.05):
                result = "error"
            results.append(result)
        

        # Create table
        df = pd.DataFrame({'autoreg-iteration':list(range(32)),
                           'p-value-Difference': pvaluesD,
                           'p-value-Base-Less': pvaluesL,
                           'p-value-Base-Greater': pvaluesG,
                           'significant-result-for-tested-model': results})
        
        current_loc = __file__.rsplit("/",3)[0]
        df.to_csv(
            current_loc +
            f'/{dir}/performance_tables/{experiment_name}_{metric}_Wilcoxon_results.csv')

    def plot_task_comparison(performance_values:np.array,                                   
            metric:['AUC', 'ΔAUC-PR'],
            experiment_name='str',
            dir = ""
        ):
        """
        This function creates a plot with individual task performances (mean across reruns).
        """
        if dir == "":
            dir = "results"

        # Compute mean performance across reruns
        mean_rerun_performance = np.mean(performance_values, axis=1)
        std_rerun_performance = np.std(performance_values, axis=1)
        
        # Create plot
        title = f"MHNfs performance [{metric}] with increasing support set"
        x_label = 'Nbr of both actives and inactives in the support set'
        y_label = f'{metric} (mean across reruns)'
        plt.figure(figsize=(15,7))

        for i in range(mean_rerun_performance.shape[0]):
            task_mean_performance = mean_rerun_performance[i, :]
            task_std_performance = std_rerun_performance[i, :]
            x_values = np.arange(1, len(task_mean_performance)+1, 1)
            y_values = task_mean_performance
            y_upper_bound = task_mean_performance + task_std_performance
            y_lower_bound = task_mean_performance - task_std_performance
            
            plt.plot(x_values, y_values, label='task'+str(i))
            """plt.fill_between(x_values,
                            y_lower_bound,
                            y_upper_bound,
                            alpha=0.1)"""

        plt.grid()
        plt.xticks(np.arange(1, len(mean_rerun_performance[0, :])+1, 1))
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        current_loc = __file__.rsplit("/",3)[0]
        plt.savefig(current_loc +
                    f'/{dir}/plots/{experiment_name}/'
                    f'{metric}_task_comparison_mean_performance_across_reruns.pdf',
                    bbox_inches='tight',pad_inches=0.1)
        plt.close()
            
    #-----------------------------------------------------------------------------------
    # Main function
    
    # Load results
    if compare == "":
        aucs, dauc_prs = load_results(experiment_name=experiment_name, dir=dir)
        aucsBase = None
        dauc_prsBase = None
    else:
        aucs, dauc_prs, aucsBase, dauc_prsBase = load_results(experiment_name=experiment_name, compare=compare, dir=dir)
        # create performance arrays of compare data
        aucsBase = create_performance_array(aucsBase)
        dauc_prsBase = create_performance_array(dauc_prsBase)

    # create performance arrays
    aucs = create_performance_array(aucs)
    dauc_prs = create_performance_array(dauc_prs)

    # Evaluate how the mean performance of MHNfs changes with increasing support set
    plot_mean_performance_across_tasks(aucs, 'AUC', experiment_name, compare_performance_values=aucsBase, dir=dir)
    plot_mean_performance_across_tasks(dauc_prs, 'ΔAUC-PR', experiment_name, compare_performance_values=dauc_prsBase, dir=dir)
    
    plot_mean_performance_across__reruns(aucs, 'AUC', experiment_name, compare_performance_values=aucsBase, dir=dir)
    plot_mean_performance_across__reruns(dauc_prs, 'ΔAUC-PR', experiment_name, compare_performance_values=dauc_prsBase, dir=dir)

    # Create performance tables
    create__performance_table(aucs, 'AUC', experiment_name, dir=dir)
    create__performance_table(dauc_prs, 'ΔAUC-PR', experiment_name, dir=dir)

    # Create Wilcoxon table
    if compare != "":
        create_paired_Wilcoxon_rank_sum_test(aucs, aucsBase, 'AUC', dir=dir)
        create_paired_Wilcoxon_rank_sum_test(dauc_prs, dauc_prsBase, 'ΔAUC-PR', dir=dir)

    # Compare tasks:
    plot_task_comparison(aucs, 'AUC', experiment_name, dir=dir)

def compare_along_experiments(experiment_names):
    def load_results(experiment_name):
        current_loc = __file__.rsplit("/",3)[0]
        with open(current_loc + '/' + experiment_name + '/autoreg_inf_fsmol.pkl', 'rb') as f:
            aucs, dauc_prs = pickle.load(f)

        return aucs, dauc_prs

    def create_performance_array(performance_dict:dict) -> np.array:
        """
        The raw results are stored in a nested dictionary where the first level of keys
        corresponds to the task id and the second level to the reruns with different
        seeds. For each (task-id, rerun) pair, the performance values are stored in a
        list.
        This function creates an array with the performance values:
        - The first dimensions corresponds to the task id
        - The second dimension corresponds to the reruns
        - The third dimension corresponds to the different support set sizes
        """
        
        performance_values = np.zeros((157,10,32)) # test 157 validation 40 #TODO
        # Compute mean performances across reruns
        for task_nbr, task_id in enumerate(performance_dict.keys()):
            # Filter relevant values
            rerun_dict = performance_dict[task_id]
            
            for rerun_nbr,rerun_key in enumerate(rerun_dict.keys()):
                values = rerun_dict[rerun_key]
                
                performance_values[task_nbr,rerun_nbr,:] = values
        
        return performance_values

    def plot_comparison_for_each_task(performance_values_list,                                   
            metric:['AUC', 'ΔAUC-PR'],
            experiment_names
            ):
        # Compute mean performance across rerun
        mean_rerun_performance_list = []
        std_rerun_performance_list = []
        for performance_values in performance_values_list:
            mean_rerun_performance = np.mean(performance_values, axis=1)
            mean_rerun_performance_list.append(mean_rerun_performance)
            std_rerun_performance = np.std(performance_values, axis=1)
            std_rerun_performance_list.append(std_rerun_performance)

        # Compute mean performance across tasks
        mean_task_performance_list = []
        for performance_values in performance_values_list:
            mean_task_performance = np.mean(performance_values, axis=0)
            #mean_task_performance = performance_values[13, :, :] #TODO just for individual task
            mean_task_performance_list.append(mean_task_performance)

        
        # Create plots for each task
        """for i in range(mean_rerun_performance_list[0].shape[0]): # loop over tasks
            title = f"MHNfs performance [{metric}] with increasing support set for different methods (task{i})"
            x_label = 'Nbr of autoregressive inference interations(1+/1- added in each round)'
            y_label = f'{metric}'
            plt.figure(figsize=(15,7))

            for j in range(len(experiment_names)): # loop over experiments
                task_mean_performance = mean_rerun_performance_list[j][i, :]
                task_std_performance = std_rerun_performance_list[j][i, :]
                x_values = np.arange(0, len(task_mean_performance), 1)
                y_values = task_mean_performance
                y_upper_bound = task_mean_performance + task_std_performance
                y_lower_bound = task_mean_performance - task_std_performance
                
                plt.plot(x_values, y_values, label=experiment_names[j])
                plt.fill_between(x_values,
                                y_lower_bound,
                                y_upper_bound,
                                alpha=0.1)

            plt.grid()
            plt.xticks(np.arange(0, len(mean_rerun_performance_list[0][0, :]), 1))
            plt.xlim(0,31)
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.legend()
            current_loc = __file__.rsplit("/",3)[0]
            #current_loc = os.getcwd().rsplit("/",2)[0] 
            plt.savefig(current_loc +
                        f'/results_test_task_comparison_for_bestexperiments_startwith8/' #TODO: set path
                        f'{metric}_task{i}_comparison_mean_performance_across_reruns_between_experiments.png',
                        bbox_inches='tight',pad_inches=0.1)
            plt.close()"""

        # Create plot for mean across tasks
        title = f"MHNfs performance [{metric}] with increasing support set for different methods"
        x_label = 'Nbr of autoregressive inference interations(1+/1- added in each round)'
        y_label = f'{metric} (mean across tasks)'
        plt.figure(figsize=(15,7))

        for j in range(len(experiment_names)): # loop over experiments
            mean_performance = np.mean(mean_rerun_performance_list[j], axis=0)
            #mean_performance = mean_rerun_performance_list[j][13, :] #TODO just for individual task
            std_performance = np.std(mean_rerun_performance_list[j], axis=0)
            #std_performance = mean_rerun_performance_list[j][13, :] * 0  #TODO just for individual task
            x_values = np.arange(0, len(mean_performance), 1)
            y_values = mean_performance
            y_upper_bound = mean_performance + std_performance
            y_lower_bound = mean_performance - std_performance
            
            plt.plot(x_values, y_values, label=experiment_names[j])
            plt.fill_between(x_values,
                            y_lower_bound,
                            y_upper_bound,
                            alpha=0.1)

        plt.grid()
        plt.xticks(np.arange(0, len(mean_rerun_performance_list[0][0, :]), 1))
        plt.xlim(0,31)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        current_loc = __file__.rsplit("/",3)[0]
        plt.savefig(current_loc +
                    f'/results_eval_comparison_for_experiments_startwith8/' #TODO: set path
                    f'{metric}_comparison_mean_performance_across_tasks_between_experiments.png',
                    bbox_inches='tight',pad_inches=0.1)
        plt.close()

        # Create plot for mean across reruns
        title = f"MHNfs performance [{metric}] with increasing support set for different methods"
        x_label = 'Nbr of autoregressive inference iterations (1+/1- added in each round)'
        y_label = f'{metric}'
        plt.figure(figsize=(15,7))

        for j in range(len(experiment_names)): # loop over experiments
            mean_performance = np.mean(mean_task_performance_list[j], axis=0)
            #mean_performance = np.mean(mean_task_performance_list[j], axis=0) #TODO just for individual task
            std_performance = np.std(mean_task_performance_list[j], axis=0)
            #std_performance = np.std(mean_task_performance_list[j], axis=0) #TODO just for individual task
            x_values = np.arange(0, len(mean_performance), 1)
            y_values = mean_performance
            y_upper_bound = mean_performance + std_performance
            y_lower_bound = mean_performance - std_performance
            
            plt.plot(x_values, y_values, label=experiment_names[j])
            plt.fill_between(x_values,
                            y_lower_bound,
                            y_upper_bound,
                            alpha=0.1)

        plt.grid()
        plt.xticks(np.arange(0, len(mean_task_performance_list[0][0, :]), 1))
        plt.xlim(0,31)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        current_loc = __file__.rsplit("/",3)[0]
        plt.savefig(current_loc +
                    f'/results_eval_comparison_for_experiments_startwith8/' #TODO: set path
                    f'{metric}_comparison_mean_performance_across_reruns_between_experiments.png',
                    bbox_inches='tight',pad_inches=0.1)
        plt.close()

    #-----------------------------------------------------------------------------------
    # Main function
    
    # Load results
    experiments_aucs = []
    experiments_daucs = []
    for experiment_name in experiment_names:
        aucs, daucs = load_results(experiment_name=experiment_name)
        # create performance arrays
        aucs = create_performance_array(aucs)
        daucs = create_performance_array(daucs)
        experiments_aucs.append(aucs)
        experiments_daucs.append(daucs)

    # Compare tasks:
    plot_comparison_for_each_task(experiments_aucs, 'AUC', experiment_names)
    plot_comparison_for_each_task(experiments_daucs, 'ΔAUC-PR', experiment_names)


#---------------------------------------------------------------------------------------
if __name__ == '__main__':
    evaluate_experiment(experiment_name='semi_supervised', dir='results_test_startwith8_VanillaMHNfs')
    #evaluate_experiment(experiment_name='semi_supervised', compare='results_test_startwith8_VanillaMHNfs', dir='results_test_startwith8_LabelCleaningMHNfs')

    # compare_along_experiments(experiment_names=['results_eval_startwith1_VanillaMHNfs', 'results_eval_startwith1_LabelCleaningMHNfs', 'results_eval_startwith8_VanillaMHNfs', 'results_eval_startwith8_LabelCleaningMHNfs'])
