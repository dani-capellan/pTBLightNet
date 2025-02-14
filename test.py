import os
from utils import parse_args_config, load_input_data, load_configs, init, print_and_log, compute_and_export_gradcam, generate_boxplot_preds, maybe_make_dir
from evaluate import evaluate_model, compute_test_metrics
from models.models_utils import getModel, getOptimizer, getLossFn
from dataloader import load_test_dataloader
import torch
import numpy as np


def test(configs, df, data, device):
    # Define folds
    if configs['experimentEnv']['cross_validation']['enabled']:
        folds = configs['experimentEnv']['cross_validation']['k']
    else:
        folds = 1
    # Load testing dataset and DataLoader
    testDataLoader, testSteps, df = load_test_dataloader(configs, df, data, return_df=True)
    # Initialize test_metrics dict
    test_metrics = {model_name: {optimizer_name: {lossFn_name: {} for lossFn_name in configs['experimentEnv']['losses']} for optimizer_name in configs['experimentEnv']['optimizers']} for model_name in configs['experimentEnv']['models']}
    # Testing
    for model_name in configs['experimentEnv']['models']:
        for optimizer_name in configs['experimentEnv']['optimizers']:
            for lossFn_name in configs['experimentEnv']['losses']:
                # Models paths
                models_paths = []
                MODEL_DIR = os.path.join(configs['data_in']['models_ckpts_dir'],configs['experimentDescription']['experiment_name'],model_name,optimizer_name,lossFn_name)
                for root, dirs, files in os.walk(MODEL_DIR):
                    if(configs['experimentEnv']['ckpt_filename'] in files):
                        models_paths.append(os.path.join(root,configs['experimentEnv']['ckpt_filename']))
                # Evaluation
                preds_logits_all = []
                gradcam_output_aggr = []
                input_aggr_aggr = []
                gt_aggr = []
                for fold in range(folds):
                    # Initialize model and configs
                    print_and_log("[INFO] initializing the model...", configs)
                    print_and_log(f"\t Model: {model_name}", configs)
                    print_and_log(f"\t Optimizer: {optimizer_name}", configs)
                    print_and_log(f"\t Loss function: {lossFn_name}", configs)
                    if(configs['experimentEnv']['cross_validation']['enabled']):
                        print_and_log(f"\t Fold: {fold}", configs)
                    # Get model, opt and LossFn
                    model = getModel(model_name, configs, device, test=True)
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
                    gt, preds_logits, preds, test_loss, input_aggr, gradcam_output = evaluate_model(
                        configs, 
                        model, 
                        lossFn, 
                        device, 
                        testDataLoader,
                        testSteps, 
                        classes=configs['experimentEnv']['classes'],
                        model_name=model_name,
                        lossFn_name=lossFn_name, 
                        return_preds = True, 
                        apply_softmax=configs['experimentEnv']['apply_softmax'], 
                        threshold=configs['experimentEnv']['pred_thresh']
                    )
                    # Process outputs
                    preds_logits_processed = np.vstack(preds_logits)
                    preds_logits_all.append(preds_logits_processed)
                    if configs["experimentEnv"]["gradcam"]:
                        gradcam_output_aggr.append(gradcam_output)
                        input_aggr_aggr.append(input_aggr)
                        gt_aggr.append(gt)

                # Adapt outputs (logits)
                # if configs["experimentEnv"]["gradcam"]:
                #     preds_logits_all_aggr = preds_logits_all.copy()
                preds_logits_all = np.stack(preds_logits_all)
                y_pred_logits = np.mean(preds_logits_all, axis=0)
                y_pred = y_pred_logits[:,1]>=configs['experimentEnv']['pred_thresh']
                
                # Out dir
                out_dir = os.path.join(configs['out_dir'], model_name, optimizer_name, lossFn_name)
                maybe_make_dir(out_dir)
                
                # Boxplot of predictions
                generate_boxplot_preds(configs, preds_logits_all, y_pred_logits, out_dir=out_dir)
                
                # Metrics
                info = {
                    'model_name': model_name,
                    'optimizer_name': optimizer_name,
                    'lossFn_name': lossFn_name,
                    'OUT_DIR': out_dir
                }
                if configs["experimentEnv"]["gradcam"]:
                    compute_and_export_gradcam(configs, df, gradcam_output_aggr, info)
                test_metrics[model_name][optimizer_name][lossFn_name] = compute_test_metrics(gt, y_pred, y_pred_logits, info, configs)
                
    return test_metrics


if __name__ == "__main__":
    # 0. Parse args for config file
    config_path = parse_args_config(test=True)
    # 1. Load configs
    configs = load_configs(config_path)
    # 2. Load input data
    df, data = load_input_data(configs, test=True)
    # 3. Define device and init
    configs, device = init(configs, test=True)
    # 4. Test
    test_metrics = test(configs, df, data, device)