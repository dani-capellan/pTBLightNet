import shap
import matplotlib.pyplot as plt
import os
from utils import maybe_make_dir
import torch
import numpy as np
import copy

def get_and_export_shap(configs, model, df_train, df_test, out_dir, fold, mode="aggregate"):
    """
    Compute and export SHAP values for a given model and input data.

    Args:
        configs (dict): A dictionary containing configurations.
        model: PyTorch model.
        df (pandas.DataFrame): A pandas dataframe containing the data to be explained.
        out_dir (str): A directory where the output files will be saved.
        fold (int): An integer indicating the fold number for cross-validation.
        mode (str): Can be either "aggregate" or "average", depending on what we want to do with SHAP values (aggregate values (flatten), or average them).

    Returns:
        shap_values: The computed SHAP values.
    """
    
    # Get train (reference) and test data
    train_data = get_feats_from_df(configs, df_train)
    test_data = get_feats_from_df(configs, df_test)
    
    # Explainer definition
    explainer = shap.DeepExplainer(model, train_data)
    
    # Get SHAP values
    shap_values = explainer.shap_values(test_data)
    
    # Process SHAP values
    if(mode=="average"):
        shap_values_mean, shap_values_std, feat_values_mean, feat_values_std, feature_names = process_shap_results(configs, shap_values, df_test, mode)
        # Explanation
        explanation = shap.Explanation(values=shap_values_mean[1], data=feat_values_mean, feature_names=feature_names)
    elif(mode=="aggregate"):
        shap_values_final, _, feat_values_final, _, feature_names = process_shap_results(configs, shap_values, df_test, mode)
        # Explanation
        explanation = shap.Explanation(values=shap_values_final[1], data=feat_values_final, feature_names=feature_names)
    
    # Prepare export
    out_dir = os.path.join(out_dir,'shap')
    maybe_make_dir(out_dir)

    # Plot shap values
    plot_and_save_beeswarm_shap(configs, explanation, out_dir, fold)
    
    if(mode=="aggregate"):
        return shap_values_final, None, feat_values_final, None, feature_names
    else:     
        return shap_values_mean, shap_values_std, feat_values_mean, feat_values_std, feature_names
    
    
def plot_and_save_beeswarm_shap(configs, explanation, out_dir, fold):

    # Figure creation
    fig = plt.figure(figsize=(20,15))
    ax1 = fig.gca()
    
    # SHAP plot
    shap.plots.beeswarm(explanation)
    
    if(fold=="all"):
        ax1.set_title("SHAP values - All models")
        fig.savefig(os.path.join(out_dir, f"shap_beeswarm_all.svg"), bbox_inches='tight', dpi=600)
        fig.savefig(os.path.join(out_dir, f"shap_beeswarm_all.pdf"), bbox_inches='tight', dpi=600)
    else:
        ax1.set_title(f"SHAP values - Fold {fold} model")
        fig.savefig(os.path.join(out_dir, f"shap_beeswarm_fold{fold}.svg"), bbox_inches='tight', dpi=600)
        fig.savefig(os.path.join(out_dir, f"shap_beeswarm_fold{fold}.pdf"), bbox_inches='tight', dpi=600)
    
    # Close fig
    plt.close(fig) 
    

def process_and_plot_shap_results_all(configs, shap_results_all, out_dir, mode="aggregate"):

    # Initialize variables
    shap_values_v = copy.deepcopy(shap_results_all[0][0])
    shap_values_std = copy.deepcopy(shap_results_all[0][1])
    feat_values_v = copy.deepcopy(shap_results_all[0][2])
    feat_values_std = copy.deepcopy(shap_results_all[0][3])
    feature_names = copy.deepcopy(shap_results_all[0][4])

    # Adapt variables
    if(mode=="aggregate"):
        ## Shap values
        for cl in range(len(shap_values_v)):
            shap_values_v[cl] = np.concatenate([shap_results_all[fold][0][cl] for fold in range(len(shap_results_all))], axis=0)
            if(not(shap_values_std is None)):
                shap_values_std = np.concatenate([shap_results_all[fold][1][cl] for fold in range(len(shap_results_all))], axis=0)
                
        ## Feat values
        feat_values_v = np.concatenate([shap_results_all[fold][2] for fold in range(len(shap_results_all))], axis=0)
        if(not(feat_values_std is None)):
                feat_values_std = np.concatenate([shap_results_all[fold][3] for fold in range(len(shap_results_all))], axis=0)
    elif(mode=="average"):
        ## Shap values
        for cl in range(len(shap_values_v)):
            shap_values_v[cl] = np.mean([shap_results_all[fold][0][cl] for fold in range(len(shap_results_all))], axis=0)
            if(not(shap_values_std is None)):
                shap_values_std = np.mean([shap_results_all[fold][0][cl] for fold in range(len(shap_results_all))], axis=0)
                
        ## Feat values
        feat_values_v = np.concatenate([shap_results_all[fold][2] for fold in range(len(shap_results_all))], axis=0)
        if(not(feat_values_std is None)):
                feat_values_std = np.concatenate([shap_results_all[fold][3] for fold in range(len(shap_results_all))], axis=0)
    
    # Explanation
    explanation = shap.Explanation(values=shap_values_v[1], data=feat_values_v, feature_names=feature_names)
    
    # Prepare export
    out_dir = os.path.join(out_dir,'shap')
    maybe_make_dir(out_dir)

    # Plot shap values
    plot_and_save_beeswarm_shap(configs, explanation, out_dir, fold="all")
    
    return shap_values_v, shap_values_std, feat_values_v, feat_values_std, feature_names


def process_shap_results(configs, shap_values, df, mode="aggregate"):
    '''
    mode: ["average", "aggregate"]
    '''
    
    if(configs['clinical_vars']['enabled'] and configs['clinical_vars']['only_clv']):
        feature_names = df.columns.values
        feature_values = df.values
        return shap_values, None, feature_values, None, feature_names
    else:
        # Step 1. (Maybe) Average feature values across AP and LAT values
        cols_df = df.columns.values
        if (configs['clinical_vars']['enabled']):
            # Get subDFs
            cols_AP = [c for c in cols_df if "APfeat" in c]
            cols_LAT = [c for c in cols_df if "LATfeat" in c]
            cols_clv = [c for c in cols_df if "LATfeat" not in c and "APfeat" not in c]
            df_AP_values = df[cols_AP].values
            df_LAT_values = df[cols_LAT].values
            df_clv_values = df[cols_clv].values
            # Perform operations
            if(mode=="average"):
                feats_AP_mean = np.expand_dims(np.mean(df_AP_values,axis=1),axis=-1)
                feats_LAT_mean = np.expand_dims(np.mean(df_LAT_values,axis=1),axis=-1)
                feats_clv_mean = df_clv_values
                feats_AP_std = np.expand_dims(np.std(df_AP_values,axis=1),axis=-1)
                feats_LAT_std = np.expand_dims(np.std(df_LAT_values,axis=1),axis=-1)
                feats_clv_std = df_clv_values*0
                # Concatenate
                feat_values_mean = np.concatenate([feats_AP_mean, feats_LAT_mean, feats_clv_mean], axis=1)
                feat_values_std = np.concatenate([feats_AP_std, feats_LAT_std, feats_clv_std], axis=1)
            elif(mode=="aggregate"):
                # Flatten/Repeat - Aggregate NumPy arrays
                feats_AP_agr = np.expand_dims(df_AP_values.flatten(),axis=-1)
                feats_LAT_agr = np.expand_dims(df_LAT_values.flatten(),axis=-1)
                reps = np.ceil(feats_AP_agr.shape[0]/df_clv_values.shape[0])
                feats_clv_agr = np.repeat(df_clv_values, reps, axis=0)
                # Concatenate
                feat_values_agr = np.concatenate([feats_AP_agr, feats_LAT_agr, feats_clv_agr], axis=1)
            else:
                feat_values_mean = np.concatenate([df_AP_values, df_LAT_values, df_clv_values], axis=1)
                feat_values_std = feat_values_mean*0
        else:
            # Get subDFs
            cols_AP = [c for c in cols_df if "APfeat" in c]
            cols_LAT = [c for c in cols_df if "LATfeat" in c]
            df_AP_values = df[cols_AP].values
            df_LAT_values = df[cols_LAT].values
            # Perform operations
            if(mode=="average"):
                feats_AP_mean = np.expand_dims(np.mean(df_AP_values,axis=1),axis=-1)
                feats_LAT_mean = np.expand_dims(np.mean(df_LAT_values,axis=1),axis=-1)
                feats_AP_std = np.expand_dims(np.std(df_AP_values,axis=1),axis=-1)
                feats_LAT_std = np.expand_dims(np.std(df_LAT_values,axis=1),axis=-1)
                # Concatenate
                feat_values_mean = np.concatenate([feats_AP_mean, feats_LAT_mean], axis=1)
                feat_values_std = np.concatenate([feats_AP_std, feats_LAT_std], axis=1)
            elif(mode=="aggregate"):
                # Flatten/Repeat - Aggregate NumPy arrays
                feats_AP_agr = np.expand_dims(df_AP_values.flatten(),axis=-1)
                feats_LAT_agr = np.expand_dims(df_LAT_values.flatten(),axis=-1)
                # Concatenate
                feat_values_agr = np.concatenate([feats_AP_agr, feats_LAT_agr], axis=1)
            else:
                feat_values_mean = np.concatenate([df_AP_values, df_LAT_values], axis=1)
                feat_values_std = feat_values_mean*0
                
        # Step 2. (Maybe) Average shap values across AP and LAT values
        if(mode=="average"):
            shap_values_mean = copy.deepcopy(shap_values)
            shap_values_std = copy.deepcopy(shap_values)
            for i in range(len(shap_values)):
                for j in range(len(shap_values[i])):
                    if(configs['clinical_vars']['only_clv'] or j>=2) or not(mode=="average"):
                        shap_values_std[i][j] = 0*shap_values[i][j]
                        continue  # Do not apply to clv
                    else:
                        shap_values_mean[i][j] = np.expand_dims(np.mean(shap_values[i][j], axis=1), axis=-1)
                        shap_values_std[i][j] = np.expand_dims(np.std(shap_values[i][j], axis=1), axis=-1)
        elif(mode=="aggregate"):
            shap_values_agr = copy.deepcopy(shap_values)
            for i in range(len(shap_values)):
                for j in range(len(shap_values[i])):
                    if(j in [0,1]):
                        shap_values_agr[i][j] = np.expand_dims(shap_values[i][j].flatten(), axis=-1)
                    else:
                        shap_values_agr[i][j] = np.repeat(shap_values[i][j], reps, axis=0)
        
        # Step 3. Stack shap values - concatenate corresponding AP, LAT and CLV output SHAP values
        if(mode=="average"):
            for i in range(len(shap_values_mean)):
                shap_values_mean[i] = np.concatenate([shap_values_mean[i][j] for j in range(len(shap_values_mean[i]))], axis=1)
                shap_values_std[i] = np.concatenate([shap_values_std[i][j] for j in range(len(shap_values_std[i]))], axis=1)
        elif(mode=="aggregate"):
            for i in range(len(shap_values_agr)):
                shap_values_agr[i] = np.concatenate([shap_values_agr[i][j] for j in range(len(shap_values_agr[i]))], axis=1)
            
        # Step 4. Adapt DF columns
        cols_df = df.columns.values
        if(mode in ["average", "aggregate"]):
            if(configs['clinical_vars']['enabled']):
                cols_clv = [c for c in cols_df if "LATfeat" not in c and "APfeat" not in c]
            else:
                cols_clv = []
            feature_names = ['feats_AP', 'feats_LAT'] + cols_clv
        else:
            feature_names = cols_df
         
        if(mode=="aggregate"):
            return shap_values_agr, None, feat_values_agr, None, feature_names
        else:     
            return shap_values_mean, shap_values_std, feat_values_mean, feat_values_std, feature_names


def get_feats_from_df(configs, df, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    
    cols = df.columns.values
    if(configs['clinical_vars']['enabled']):
        if(configs['clinical_vars']['only_clv']):
            data = torch.from_numpy(df.values).to(device)
        else:
            cols_AP = [c for c in cols if "APfeat" in c]
            cols_LAT = [c for c in cols if "LATfeat" in c]
            cols_clv = [c for c in cols if "LATfeat" not in c and "APfeat" not in c]
            df_AP = df[cols_AP]
            df_LAT = df[cols_LAT]
            df_clv = df[cols_clv]
            data = [
                torch.from_numpy(df_AP.values).to(device),
                torch.from_numpy(df_LAT.values).to(device),
                torch.from_numpy(df_clv.values).to(device)
            ]
    else:
        cols_AP = [c for c in cols if "APfeat" in c]
        cols_LAT = [c for c in cols if "LATfeat" in c]
        df_AP = df[cols_AP]
        df_LAT = df[cols_LAT]
        data = [
            torch.from_numpy(df_AP.values).to(device),
            torch.from_numpy(df_LAT.values).to(device)
        ]

    return data