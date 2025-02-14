import os
import torch
import numpy as np
from utils import parse_args_config, load_input_data, load_configs, init, print_and_log
from evaluate import evaluate_model_ensemble, compute_test_metrics
from models.models_utils import getModel, getOptimizer, getLossFn, maybe_load_models_for_ensemble, get_AP_LAT_models_for_ensemble
from dataloader import load_train_dataloader, load_test_dataloader
from utils import summary, process_clv_ensemble, get_and_export_shap, process_and_plot_shap_results_all, generate_boxplot_preds, maybe_make_dir


def test(configs, df, data, device, return_test_metrics=False):
    # Define folds
    if configs['experimentEnv']['cross_validation']['enabled']:
        folds = configs['experimentEnv']['cross_validation']['k']
    else:
        folds = 1
    # Define clv values
    if(configs['clinical_vars']['enabled'] and not(configs['clinical_vars']['only_clv'])):
        clv_values_loop = range(len(configs['clinical_vars']['values']))
    else:
        clv_values_loop = range(1)
    # Initialize test_metrics dict
    test_metrics = {model_name: {optimizer_name: {lossFn_name: {clv_i: {} for clv_i in clv_values_loop} for lossFn_name in configs['experimentEnv']['losses']} for optimizer_name in configs['experimentEnv']['optimizers']} for model_name in configs['experimentEnv']['models']}
    # Training
    for model_name in configs['experimentEnv']['models']:
        for optimizer_name in configs['experimentEnv']['optimizers']:
            for lossFn_name in configs['experimentEnv']['losses']:  
                for clv_i in clv_values_loop:
                    # Process clv value for ensemble    
                    configs, clv_value = process_clv_ensemble(configs, clv_i, test=True)
                    # Models paths
                    models_paths = []
                    if(configs['clinical_vars']['enabled'] and not(configs['clinical_vars']['only_clv'])):
                        MODEL_DIR = os.path.join(configs['data_in']['models_ckpts_dir'],configs['experimentDescription']['experiment_name'],model_name,optimizer_name,lossFn_name,f"clv_{clv_value}")
                    else:
                        MODEL_DIR = os.path.join(configs['data_in']['models_ckpts_dir'],configs['experimentDescription']['experiment_name'],model_name,optimizer_name,lossFn_name)
                    for root, dirs, files in os.walk(MODEL_DIR):
                        if(configs['experimentEnv']['ckpt_filename'] in files):
                            models_paths.append(os.path.join(root,configs['experimentEnv']['ckpt_filename']))
                    # Evaluation
                    preds_logits_all = []
                    shap_results_all = []
                    for fold in range(folds):
                        # Initialize model and configs
                        print_and_log("[INFO] initializing the model...", configs)
                        print_and_log(f"\t Model: {model_name}", configs)
                        print_and_log(f"\t Optimizer: {optimizer_name}", configs)
                        print_and_log(f"\t Loss function: {lossFn_name}", configs)
                        
                        if(not(configs['experimentEnv']['cross_validation']['enabled']) and not(configs['experimentEnv']['use_validation_split'])):  # no CV, no validation split
                            fold="all"  # Define fold = "all"

                        # Print and log fold and clv_value
                        print_and_log(f"\t Fold: {fold}", configs)
                        if(configs['clinical_vars']['enabled'] and not(configs['clinical_vars']['only_clv'])):
                            print(f"\t CLV value: {clv_value}")
                        
                        # Maybe load train data (SHAP)
                        if(configs['experimentEnv']['shap']):
                            trainDataLoader, trainSteps = load_train_dataloader(configs, df, data, fold, clv_value, test=True)
                        else:
                            trainDataLoader, trainSteps = None, None
                        
                        # Load test data
                        testDataLoader, testSteps = load_test_dataloader(configs, df, data, clv_value)
                                               
                        # Get and load AP & LAT models
                        models_AP_LAT = get_AP_LAT_models_for_ensemble(configs, device, fold, test=True)

                        # Model definition - MLP
                        model = getModel(model_name, configs, device, test=True)
                        
                        # Summary of the model
                        try:
                            if(configs['clinical_vars']['enabled']):
                                if(configs['clinical_vars']['only_clv']):
                                    summary(model,[(6,)])
                                else:
                                    summary(model,[(4096,),(4096,),(6,)])
                            else:
                                summary(model,[(4096,),(4096,)])
                        except:
                            print_and_log("Pytorch-Summary failed. Please check if anything is wrong.", configs)

                        # Initialize optimizer and loss function
                        opt = getOptimizer(optimizer_name, model_name, model, configs, test=True)
                        lossFn = getLossFn(lossFn_name)
                        
                        # Load model, opt
                        model_path = [p for p in models_paths if f"fold_{fold}" in p][0]
                        checkpoint = torch.load(model_path)
                        epoch = checkpoint['epoch']
                        model.load_state_dict(checkpoint['model_state_dict'])
                        opt.load_state_dict(checkpoint['optimizer_state_dict'])

                        # Test process
                        print_and_log("[INFO] testing network...\n", configs)
                        eval_output = evaluate_model_ensemble(
                            configs,
                            model,
                            models_AP_LAT,
                            lossFn,
                            device,
                            testDataLoader,
                            testSteps,
                            classes=configs['experimentEnv']['classes'], 
                            lossFn_name=lossFn_name,
                            return_preds = True, 
                            apply_softmax=configs['experimentEnv']['apply_softmax'], 
                            threshold=configs['experimentEnv']['pred_thresh']
                        )

                        if(configs['experimentEnv']['shap']):
                            tr_eval_output = evaluate_model_ensemble(
                                configs,
                                model,
                                models_AP_LAT,
                                lossFn,
                                device,
                                trainDataLoader,
                                trainSteps,
                                classes=configs['experimentEnv']['classes'], 
                                lossFn_name=lossFn_name,
                                return_preds = True, 
                                apply_softmax=configs['experimentEnv']['apply_softmax'], 
                                threshold=configs['experimentEnv']['pred_thresh']
                            )
                            _, _, _, _, df_train_feats_shap = tr_eval_output  # training info for shap reference
                            gt, preds_logits, preds, test_loss, df_feats_shap = eval_output
                        else:
                            gt, preds_logits, preds, test_loss = eval_output
                            
                        # Maybe export shap
                        if(configs['experimentEnv']['shap']):
                            out_dir = os.path.join(configs['out_dir'], model_name, optimizer_name, lossFn_name, f"clv_{clv_value}")
                            shap_results = get_and_export_shap(configs, model, df_train_feats_shap, df_feats_shap, out_dir, fold, mode="aggregate")
                            if configs['experimentEnv']['cross_validation']['enabled']:
                                shap_results_all.append(get_and_export_shap(configs, model, df_train_feats_shap, df_feats_shap, out_dir, fold))
                        
                        # Process outputs
                        preds_logits_processed = np.vstack(preds_logits)
                        preds_logits_all.append(preds_logits_processed)

                    # Adapt outputs (logits)
                    preds_logits_all = np.stack(preds_logits_all)
                    y_pred_logits = np.mean(preds_logits_all, axis=0)
                    y_pred = y_pred_logits[:,1]>=configs['experimentEnv']['pred_thresh']
                    
                    # Out dir
                    out_dir = os.path.join(configs['out_dir'], model_name, optimizer_name, lossFn_name, f"clv_{clv_value}")
                    maybe_make_dir(out_dir)
                    
                    # Boxplot of predictions
                    generate_boxplot_preds(configs, preds_logits_all, y_pred_logits, out_dir=out_dir)
                    
                    # Process shap results
                    if(configs['experimentEnv']['shap'] and configs['experimentEnv']['cross_validation']['enabled']):
                        shap_results_final = process_and_plot_shap_results_all(configs, shap_results_all, out_dir)
                    else:
                        shap_results_final = shap_results_all
                    
                    # Metrics
                    info = {
                        'model_name': model_name,
                        'optimizer_name': optimizer_name,
                        'lossFn_name': lossFn_name,
                        'OUT_DIR': out_dir
                    }
                    test_metrics[model_name][optimizer_name][lossFn_name][clv_value] = compute_test_metrics(gt, y_pred, y_pred_logits, info, configs)
        
        if(return_test_metrics):
            return test_metrics
    

if __name__ == "__main__":
    # 0. Parse args for config file
    config_path = parse_args_config(test=True, ensemble=True)
    # 1. Load configs
    configs = load_configs(config_path)
    configs = maybe_load_models_for_ensemble(configs)
    # 2. Load input data
    df, data = load_input_data(configs, test=True)
    # 3. Define device and init
    configs, device = init(configs, test=True)
    # 4. Test
    test(configs, df, data, device)