import os
import argparse
import pandas as pd
import pickle
import yaml
import torch
from datetime import datetime
import numpy as np
import wandb
import re
from shutil import rmtree
import logging
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args_config(test=False, ensemble=False):
    parser = argparse.ArgumentParser(description='Introduce the location of a YAML configuration file.')
    if(test):
        if(ensemble):
            default_path = "./config_test_ensemble.yaml"
        else:
            default_path = "./config_test.yaml"
    else:
        if(ensemble):
            default_path = "./config_train_ensemble.yaml"
        else:
            default_path = "./config_train.yaml"
    parser.add_argument('--config', '-cfg', type=str, required=False, default=default_path, help='Path to YAML configuration file')
    args = parser.parse_args()
    return args.config


def maybe_make_dir(path):
    if(not(os.path.isdir(path))):
        os.makedirs(path)


def load_configs(PATH):
    with open(PATH) as file:
        configs = yaml.safe_load(file)

    return configs


def load_input_data(configs, test=False):
    '''
    Loads input data
    Inputs:
        configs: dict
    Outputs:
        df: Pandas DataFrame (dataset info)
        data: dict with images
        
    NOTE: We take into account different fields from CSV file (DataFrame) to load the data. The most important ones:
            - patient_id (used to load images from dict)
            - fold_cv (if cross_validation enabled)
          Warning! If cross validation (CV) is enabled, please make sure that "fold_cv" column is properly included.
    '''
    
    # 1. Read DF
    try:
        df = pd.read_csv(configs['data_in']['csv'])
        if 'Unnamed: 0' in df.columns:
            df = pd.read_csv(configs['data_in']['csv'], index_col=0)
    except:
        raise Exception("CSV unable to be read")
    
    # 2. Select view
    view = configs['experimentEnv']['view']
    if(view in ['AP', 'LAT']):
        df = df[df[f"filepath_{view}"]!='-1']  # take out all cases where filepath_{view} is -1 (null)
    elif(view=='AP-LAT'):
        df = df[(df["filepath_AP"]!='-1') & (df["filepath_LAT"]!='-1')]  # take out all cases where filepath_{view} is -1 (null)
    
    # 2. Check fold_cv column and folds
    if((configs['experimentEnv']['cross_validation']['enabled']) and (not(test) or (test and configs['experimentEnv']['shap']))):
        assert 'fold_cv' in df.columns, "fold_cv column not included in CSV!"
        folds = configs['experimentEnv']['cross_validation']['k']
        folds_csv = np.max(df['fold_cv'])+1
        assert folds == folds_csv, "Folds do not match with CSV"

    # 3. Read PKL
    with open(os.path.join(configs['data_in']['pkl']), 'rb') as handle:
        data = pickle.load(handle)

    return df, data


def init_log(configs, test=False):
    if(test):
        log_dir = os.path.join(os.path.join(configs['out_dir']))
    else:
        log_dir = os.path.join(configs['out_training_dir'],configs['experimentDescription']['experiment_name'])
    maybe_make_dir(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, "log.txt"), filemode='w', level=logging.DEBUG, format='%(message)s')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.info("###### LOGGER INITIALIZED ######")
    configs['logger'] = logger
    
    return configs
    

def init(configs, test=False):
    '''
    Function with some initialization steps
    '''
    # Init WandB
    if(not(test)):
        if(configs['wandb']['enable_tracking']):
            wandb_init()
    # Configurations + Logging
    if(test):
        if configs['experimentDescription']['experiment_name'] == "" or configs['data_in']['csv'] == "" or configs['data_in']['pkl'] == "" or configs['data_in']['models_ckpts_dir'] == "" or configs['out_dir'] == "":
            raise Exception("Please check you have entered all the information required in the corresponding config YAML file.")
        configs['out_dir'] = os.path.join(configs['out_dir'],configs['experimentDescription']['experiment_name'])
    else:
        if configs['out_training_dir'] == "":
            raise Exception("Please check you have entered all the information required in the corresponding config YAML file.")
        if(not(configs['experimentDescription']['experiment_name']) and not(test)):
            now = datetime.now() # current date and time
            d = now.strftime("%d%m%Y_%H%M%S")
            configs['experimentDescription']['experiment_name'] = f"exp_{d}"
    if(configs['log_output']):    
        configs = init_log(configs, test)
    print_cite(configs)
    # Initialize configurations/seeds/device
    torch.manual_seed(configs['random_state'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Print and log configs
    print_and_log("### CONFIGURATIONS ###\n", configs)
    print_and_log(configs, configs)
    print_and_log("\n######################\n", configs)
    
    # Save configs
    if(not(test)):
        save_configs(configs)

    return configs, device


def wandb_init():
    wandb.login()


def get_best_epoch(history,metric,trim_epochs_ratio=1):
    '''
    trim_epochs_ratio: above which epoch we must search for this epoch. 
    Example: 0.75. That means that, for example, if the number of epochs used for training were 400, the best epoch that we are looking for will among the last 300 epochs (75% of 400).
    
    Default value: 1. That is, all epochs are considered
    
    best_epoch: Best epoch knowing that epochs begin by 0
    '''
    a = [[epoch,value] for epoch,value in zip(range(0,len(history[metric])),history[metric]) if epoch in range(int((1-trim_epochs_ratio)*len(history[metric])),len(history[metric]))]  # Considering trim_epochs_ratio
    # a = [[epoch,value] for epoch,value in zip(range(0,len(history[metric])),history[metric])]  # Without considering trim_epochs_ratio
    a = np.array(a) # To numpy
    if(metric=='val_loss'):
        a = a[a[:,1].argsort()]  # Order by second column in ascendent order (best at the top)
        a = a[a[:,1]==a[0,1]]  # Get all cases where the metric value is the highest (could be more than one)
        best_epoch_values = a[np.argmax(a[:,0])]  # Among the ones with the best metric, take the highest epoch
        best_epoch = int(best_epoch_values[0])  # Best epoch to int
    elif(metric in ['val_acc','val_auc','val_aupr']):
        a = a[a[:,1].argsort()][::-1]  # Order by second column in descendent order (best at the top)
        a = a[a[:,1]==a[0,1]]  # Get all cases where the metric value is the highest (could be more than one)
        best_epoch_values = a[np.argmax(a[:,0])]  # Among the ones with the best metric, take the highest epoch
        best_epoch = int(best_epoch_values[0])  # Best epoch to int
    return best_epoch

def get_best_model_checkpoint(history,configs,model_name,optimizer_name,lossFn_name,fold,metric='val_acc',trim_epochs_ratio=1):
    best_epoch = get_best_epoch(history,metric,trim_epochs_ratio)
    best_epoch_metrics = {
        "val_loss": history["val_loss"][best_epoch],
        "val_acc": history["val_acc"][best_epoch],
        "val_f1" : history["val_f1"][best_epoch],
        "val_precision" : history["val_precision"][best_epoch],
        "val_recall" : history["val_recall"][best_epoch],
        "val_auc" : history["val_auc"][best_epoch],
        "val_aupr" : history["val_aupr"][best_epoch],
        "val_sn" : history["val_sn"][best_epoch],
        "val_sp" : history["val_sp"][best_epoch]
    }
    if(configs['wandb']['enable_tracking']):
        wandb.log({
            "best_epoch": best_epoch,  # Epoch starting with 0
            "best_val_loss": best_epoch_metrics["val_loss"],
            "best_val_acc": best_epoch_metrics["val_acc"],
            "best_val_f1" : best_epoch_metrics["val_f1"],
            "best_val_precision" : best_epoch_metrics["val_precision"],
            "best_val_recall" : best_epoch_metrics["val_recall"],
            "best_val_auc" : best_epoch_metrics["val_auc"],
            "best_val_aupr" : best_epoch_metrics["val_aupr"],
            "best_val_sn" : best_epoch_metrics["val_sn"],
            "best_val_sp" : best_epoch_metrics["val_sp"]
        })
    # Get best model
    best_model_path = os.path.join(configs['out_training_dir'],configs['experimentDescription']['experiment_name'],model_name,optimizer_name,lossFn_name,f"fold_{str(fold)}",f"epoch_{str(best_epoch).zfill(len(str(configs['experimentEnv']['epochs'])))}","model.pth")

    return best_epoch, best_epoch_metrics, best_model_path


def log_wandb(wandb_dict):
    '''
    Log info in WandB
    '''
    wandb.log(wandb_dict)


def remove_checkpoints(PATH, configs):
    '''
    Removes subfolders inside PATH that match regex pattern "epoch_\d+"
    '''
    pattern = re.compile("epoch_\d+")
    for root, subdirs, files in os.walk(PATH, True):
        for subdir in subdirs:
            if(re.match(r"epoch_(\d+)", subdir)):
                print_and_log(f"Deleting... {os.path.join(root,subdir)}", configs)
                rmtree(os.path.join(root,subdir), ignore_errors=True)


# def print_cite(configs):
#     print_and_log("\nPlease cite the following papers when using LightTBNet:\n\nCapellán-Martín, D., Gómez-Valverde, J.J., Bermejo-Peláez, D. et al. "
#       "\"A lightweight, rapid and efficient deep convolutional network for chest X-ray tuberculosis detection\" "
#       "2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI 2023). https://doi.org/10.1109/ISBI53787.2023.10230500\n", configs)
#     print_and_log("If you have questions or suggestions, feel free to open an issue at https://github.com/dani-capellan/LightTBNet\n", configs)
def print_cite(configs):
    print_and_log("\nPlease cite the following papers when using pTBLightNet:\n", configs)
    print_and_log("- Capellán-Martín, D., Gómez-Valverde, J.J., Bermejo-Peláez, D. et al. "
      "\"A lightweight, rapid and efficient deep convolutional network for chest X-ray tuberculosis detection\" "
      "2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI 2023). https://doi.org/10.1109/ISBI53787.2023.10230500\n", configs)
    print_and_log("- Capellán-Martín, D., Gómez-Valverde, J. J., Sánchez-Jacob, R., et al. (2025). "
      "\"Multi-View Deep Learning Framework for the Detection of Chest X-Rays Compatible with Pediatric Pulmonary Tuberculosis.\" Under Review.\n", configs)
    print_and_log("If you have questions or suggestions, feel free to open an issue at https://github.com/dani-capellan/pTBLightNet\n", configs)
    

def print_and_log(s: str, configs):
    print(s)
    if(configs['log_output']):
        configs['logger'].info(s)
        
        
def finish_log(configs):
    if(configs['log_output']):
        now = datetime.now()
        now_string = now.strftime("%d/%m/%Y %H:%M:%S")
        configs['logger'].info(f"\nEnd date and time: {now_string}")
        

# def plot_step(H, epoch, configs, model_name, optimizer_name, lossFn_name):
#     fig = plt.figure(figsize=(30, 24))
#     ax1 = fig.gca()
#     ax2 = ax1.twinx()
#     ax1.plot(list(range(epoch+1)), H['train_loss'], 'b-', label='Training loss')
#     ax1.plot(list(range(epoch+1)), H['val_loss'], 'r-', label='Validation loss')
#     ax2.plot(list(range(epoch+1)), H['val_auc'], 'g-', label='Validation AUC')
#     ax1.legend()
#     ax2.legend(loc='upper right', bbox_to_anchor=(1, -0.15, 0, 0), ncol=2)
#     ax1.set_xlabel("Epoch")
#     ax1.set_ylabel("Loss")
#     ax2.set_ylabel("Evaluation Metric")
#     out_dir = os.path.join(configs['out_training_dir'], configs['experimentDescription']['experiment_name'], model_name, optimizer_name, lossFn_name)
#     maybe_make_dir(out_dir)
#     fig.savefig(os.path.join(out_dir, 'progress.svg'))
#     plt.close(fig)


def plot_step(H, epoch, configs, model_name, optimizer_name, lossFn_name, fold):
    fig = plt.figure(figsize=(30, 24))
    ax1 = fig.gca()
    ax2 = ax1.twinx()
    ax1.plot(list(range(epoch+1)), H['train_loss'], 'b-', label='Training loss')
    ax1.plot(list(range(epoch+1)), H['val_loss'], 'r-', label='Validation loss')
    ax2.plot(list(range(epoch+1)), H['val_auc'], 'g-', label='Validation AUC')
    ax2.plot(list(range(epoch+1)), H['val_aupr'], 'c--', label='Validation AUPR')

    lns1 = ax1.get_lines()
    lns2 = ax2.get_lines()
    labs1 = [l.get_label() for l in lns1]
    labs2 = [l.get_label() for l in lns2]

    lns = lns1 + lns2
    labs = labs1 + labs2
    ax1.legend(lns, labs, loc='upper right')

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Evaluation Metric")
    out_dir = os.path.join(configs['out_training_dir'], configs['experimentDescription']['experiment_name'], model_name, optimizer_name, lossFn_name, f"fold_{str(fold)}")
    maybe_make_dir(out_dir)
    fig.savefig(os.path.join(out_dir, 'progress.svg'))
    fig.savefig(os.path.join(out_dir, 'progress.pdf'))
    plt.close(fig)
        
        
def generate_boxplot_preds(configs, preds_logits_all, y_pred_logits, out_dir):
    '''
    pred_logits_all: (n_folds, n_samples, n_classes)
    y_pred_logits: (n_samples, n_classes)

    Generates three plots:
    i) Boxplot with predictions from each of the folds (class 1), each fold one color
    ii) Boxplot with predictions from final predictions (class 1) (y_pred_logits[:,1])
    iii) Boxplot of predictions including each of the folds and final predictions
    '''
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    
    # 1. First Boxplot
    ## Create a figure and axis for the first boxplot
    fig, ax = plt.subplots(figsize=(8, 8))  # Adjust the figsize to make it more vertical

    ## Rearrange the data into a dataframe where the first column is the fold and the second all the values
    preds_logits_all_df = pd.DataFrame(columns=['Fold', 'Value'])
    for i in range(len(preds_logits_all)):
        preds_logits_all_df = preds_logits_all_df.append(pd.DataFrame({'Fold': [i]*len(preds_logits_all[i, :, 1]), 'Value': preds_logits_all[i, :, 1]}))
    
    ## Plot each fold's values as a boxplot
    sns.boxplot(data=preds_logits_all_df, x='Fold', y='Value', palette="Set2", ax=ax)
    
    ## Add a stripplot to show individual predictions as points
    sns.stripplot(data=preds_logits_all_df, x='Fold', y='Value', jitter=True, color="black", size=3, ax=ax)

    ## Style the plot
    ax.set_xticks(range(len(preds_logits_all)))
    ax.set_xticklabels(['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'])
    ax.set_title(f"Boxplot of predictions - Folds")
    ax.set(xlabel=None)
    ax.set_ylabel("Probabilities")

    ## Save the plot
    for ext in ['svg', 'pdf']:
        fig.savefig(os.path.join(out_dir, f"boxplot_preds_folds.{ext}"), bbox_inches="tight")  # Use bbox_inches to make the output SVG more vertical
    plt.close(fig)
    
    # 2. Second Boxplot
    ## Create a figure and axis for the second boxplot
    fig, ax = plt.subplots(figsize=(5, 8))  # Adjust the figsize to make it more vertical
    
    ## Plot each fold's values as a boxplot
    sns.boxplot(data=y_pred_logits[:, 1], palette="Set2", ax=ax)
    
    ## Add a stripplot to show individual predictions as points
    sns.stripplot(data=y_pred_logits[:, 1], jitter=True, color="black", size=3, ax=ax)
    
    ## Style the plot
    ax.set_title(f"Boxplot of predictions - Ensemble")
    ax.set_xticklabels(['Ensembled predictions'])
    ax.set(xlabel=None)
    ax.set_ylabel("Probabilities")
    
    ## Save the plot
    for ext in ['svg', 'pdf']:
        fig.savefig(os.path.join(out_dir, f"boxplot_preds_avg.{ext}"), bbox_inches="tight")  # Use bbox_inches to make the output SVG more vertical
    plt.close(fig)
    
    # 3. Third Boxplot
    ## Create a figure and axis for the third boxplot
    fig, ax = plt.subplots(figsize=(7, 6))
    
    ## Rearrange the data into a dataframe where the first column is the fold and the second all the values
    ### Use preds_logits_all_df generated before and just append (vertically) the final predictions
    preds_logits_all_df = pd.concat([preds_logits_all_df, pd.DataFrame({'Fold': ['Ensemble']*len(y_pred_logits[:, 1]), 'Value': y_pred_logits[:, 1]})])
        
    preds_logits_all_df.to_csv(os.path.join(out_dir, "boxplot_preds_all.csv"), index=False)
    
    ## Plot each fold's values as a boxplot
    sns.boxplot(data=preds_logits_all_df, x='Fold', y='Value', palette="Set2", ax=ax)
    
    ## Add a stripplot to show individual predictions as points
    sns.stripplot(data=preds_logits_all_df, x='Fold', y='Value', jitter=True, color="black", size=3, ax=ax)

    ## Style the plot
    ax.set_xticks(range(len(preds_logits_all)+1))
    ax.set_xticklabels(['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Ensemble'])
    ax.set_title(f"Boxplot of predictions - Folds & Ensemble")
    ax.set(xlabel=None)
    ax.set_ylabel("Probabilities")

    ## Save the plot
    for ext in ['svg', 'pdf']:
        fig.savefig(os.path.join(out_dir, f"boxplot_preds_all.{ext}"), bbox_inches="tight")  # Use bbox_inches to make the output SVG more vertical
    plt.close(fig)


def save_configs(configs):
    out_path = os.path.join(configs['out_training_dir'],configs['experimentDescription']['experiment_name'],"params.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(configs, f)
        
        
def process_clv_ensemble(configs, clv_i, test=False):
    if(configs['clinical_vars']['enabled'] and configs['clinical_vars']['only_clv']):
        clv_value = configs['clinical_vars']['values'][clv_i]
        if(not(test)):
            configs['experimentDescription']['experiment_description'] = configs['experimentDescription']['experiment_description'].replace("__clinical_variables__",f"ONLY clinical variables")
    elif(configs['clinical_vars']['enabled'] and not(configs['clinical_vars']['only_clv']) and clv_i==0):
        clv_value = configs['clinical_vars']['values'][clv_i]
        if(not(test)):
            configs['experimentDescription']['experiment_description'] = configs['experimentDescription']['experiment_description'].replace("__clinical_variables__",f"clinical variables (-{clv_value},{clv_value})")
    elif(configs['clinical_vars']['enabled'] and not(configs['clinical_vars']['only_clv']) and clv_i>0):
        clv_value = configs['clinical_vars']['values'][clv_i]
        if(not(test)):
            configs['experimentDescription']['experiment_description'] = configs['experimentDescription']['experiment_description'].replace(f"clinical variables (-{configs['clinical_vars']['values'][clv_i-1]},{configs['clinical_vars']['values'][clv_i-1]})",f"clinical variables (-{clv_value},{clv_value})")
    else:
        clv_value = 0
        if(not(test)):
            configs['experimentDescription']['experiment_description'] = configs['experimentDescription']['experiment_description'].replace("__clinical_variables__",f"no clinical variables")
        
    return configs, clv_value


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__