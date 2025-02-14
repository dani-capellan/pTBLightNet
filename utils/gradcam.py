
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM
from skimage.color import gray2rgb
from utils import maybe_make_dir
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import cv2
import os


def get_gradcam(model, input_tensor, gt, pred_logits, pred, model_name, device,use_cuda=True, imshow=False):
    '''
    input_tensor: 1 x dim x dim torch Tensor (in CPU)
    source: https://github.com/jacobgil/pytorch-grad-cam
    
    '''
    model.eval()  # Model in evaluation mode
    if(model_name == 'pTBResNetv2'):
        target_layers = [model.layer3[-1]]
    elif((model_name == 'pTBResNetv3') or ("pTBLightNet" in model_name)):
        target_layers = [model.features[-1][0]]
    elif(model_name == 'DenseNet121'):
        target_layers = [model.features.denseblock4.denselayer16.conv2]
    else:
        raise NotImplementedError(f"Model {model_name} not implemented in GradCAMs.")
    # Convert input_tensor to RGB image
    rgb_img = gray2rgb(input_tensor[0].cpu().numpy())
    # Normalize between 0 and 1
    rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))
    # Adapt input tensor
    input_tensor = input_tensor[None,:].to(device)  # From (1,dim,dim) to (1,1,dim,dim)
    # GradCAM
    if(imshow):
        plt.figure()
    gradcams_out = {
        "rgb_img": (rgb_img*255).astype(np.uint8),
        "gradcam": {tgt: None for tgt in [0,1]},
        "overlay": {tgt: None for tgt in [0,1]},
        "gt": gt[1],
        "pred_logits": pred_logits,
        "pred": pred,
        "pred_type": get_pred_type(gt[1], pred),
    }
    for tgt in [0,1]:  # Obtain GradCAM for each of the classes
        targets = [ClassifierOutputTarget(tgt)]
        with GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  # No smoothing. You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = grayscale_cam[0, :]  # (1,dim,dim)
            gradcams_out["gradcam"][tgt] = grayscale_cam.copy()  # (dim, dim)
            visualization = show_cam_on_image(rgb_img, grayscale_cam)
            gradcams_out["overlay"][tgt] = visualization.copy() # (dim, dim, 3)
            if(imshow):
                plt.subplot(1,2,tgt+1)
                plt.imshow(visualization)
                plt.title(f"Target: {tgt}")
                plt.show()
    
    return gradcams_out


def get_gradcam_batch(model, input_tensor_batch, gts, preds_logits, preds, model_name, device,use_cuda=True, imshow=False):
    '''
    input_tensor: batch x dim x dim torch Tensor (in CPU)
    '''
    gradcams_all = []
    for input_tensor, gt, pred_logits, pred in zip(input_tensor_batch, gts, preds_logits, preds):
        gradcams_all.append(get_gradcam(model, input_tensor, gt, pred_logits, pred, model_name, device, use_cuda, imshow))
        
    return gradcams_all
    
    
def get_pred_type(gt, pred):
    if gt==1 and pred==1:
        return "TP"
    elif gt==0 and pred==0:
        return "TN"
    elif gt==1 and pred==0:
        return "FN"
    elif gt==0 and pred==1:
        return "FP"
    

def compute_mean_gradcams(configs, gradcam_output_aggr):
    '''
    gradcam_output_mean: (kfold, batch):
        inside: dict len n_classes:
            inside: (dim,dim)
        
    desired ouput: (batch, n_classes, dim, dim)
    '''
    folds = len(gradcam_output_aggr)
    n_cases = len(gradcam_output_aggr[0])
    n_classes = len(gradcam_output_aggr[0][0]["gradcam"])
    gradcam_output_mean = [gradcam_output_aggr[0].copy()]
    for idx in range(n_cases):
        gradcam_output_mean[0][idx]["pred_logits"] = np.mean([gradcam_output_aggr[fold][idx]["pred_logits"] for fold in range(folds)],axis=0)
        gradcam_output_mean[0][idx]["pred"] = int(gradcam_output_mean[0][idx]["pred_logits"][1]>=configs['experimentEnv']['pred_thresh'])
        gradcam_output_mean[0][idx]["pred_type"] = get_pred_type(gradcam_output_mean[0][idx]["gt"], gradcam_output_mean[0][idx]["pred"])
        for tgt in range(n_classes):
            gradcam_output_mean[0][idx]["gradcam"][tgt] = np.mean(np.stack([gradcam_output_aggr[fold][idx]["gradcam"][tgt] for fold in range(folds)]),axis=0)
            gradcam_output_mean[0][idx]["overlay"][tgt] = np.mean(np.stack([gradcam_output_aggr[fold][idx]["overlay"][tgt] for fold in range(folds)]),axis=0)
    
    return gradcam_output_mean

def write_text_on_image(img, text1, text2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    color = (255, 0, 0)
    (text1_width, text1_height), _ = cv2.getTextSize(text1, font, font_scale, thickness)
    (text2_width, text2_height), _ = cv2.getTextSize(text2, font, font_scale, thickness)
    text1_org = (10, img.shape[0] - text1_height - 10)
    text2_org = (10, img.shape[0] - text1_height - text2_height - 20)
    cv2.putText(img, text1, text1_org, font, font_scale, color, thickness)
    cv2.putText(img, text2, text2_org, font, font_scale, color, thickness)

def compute_and_export_gradcam(configs, df, gradcam_output_aggr, info, print_text=True):
    # Out dir
    out_gradcams = os.path.join(info['OUT_DIR'], 'gradcams')
    maybe_make_dir(out_gradcams)
    # 1. Export gradcam per fold
    for fold in range(len(gradcam_output_aggr)):
        for idx in range(len(gradcam_output_aggr[fold])):
            # Get info
            gt, pred, pred_score, pred_type = [gradcam_output_aggr[fold][idx][key] for key in ["gt", "pred", "pred_logits", "pred_type"]]
            pid = df.iloc[idx]['patient_id']
            # Collage
            # out_img = np.hstack([gradcam_output_aggr[fold][idx]["rgb_img"], gradcam_output_aggr[fold][idx]["overlay"][pred]]).astype(np.uint8)  # Gradcam of the predicted class
            out_img = np.hstack([gradcam_output_aggr[fold][idx]["rgb_img"], gradcam_output_aggr[fold][idx]["overlay"][1]]).astype(np.uint8)  # Gradcam of the target class (1)
            # Put text on the image (info)
            if print_text:
                score = str(np.round(pred_score[1],2))
                text1 = f"GT: {gt}, PRED: {pred} ({score}) - {pred_type}"
                text2 = f"PID: {df.iloc[idx]['patient_id']}"
                write_text_on_image(out_img, text1, text2)
            # Get image name
            image_name = f"gradcam_idx_{pid}_gt_{gt}_pred_{pred}_type_{pred_type}_fold_{fold}.png"
            # Save collage
            cv2.imwrite(os.path.join(out_gradcams, image_name), out_img)

    # Export averaged gradcam
    gradcam_output_mean = compute_mean_gradcams(configs, gradcam_output_aggr)
    for idx in range(len(gradcam_output_mean[0])):
        # Get info
        gt, pred, pred_score, pred_type = [gradcam_output_mean[0][idx][key] for key in ["gt", "pred", "pred_logits", "pred_type"]]
        pid = df.iloc[idx]['patient_id']
        # Collage
        out_img = np.hstack([gradcam_output_mean[0][idx]["rgb_img"], gradcam_output_mean[0][idx]["overlay"][pred]]).astype(np.uint8)
        # Put text on the image (info)
        if print_text:
            score = str(np.round(pred_score[1],2))
            text1 = f"GT: {gt}, PRED: {pred} ({score}) - {pred_type}"
            text2 = f"PID: {pid}"
            write_text_on_image(out_img, text1, text2)
        # Get image name
        image_name = f"gradcam_idx_{pid}_gt_{gt}_pred_{pred}_type_{pred_type}_fold_avg.png"
        # Save collage
        cv2.imwrite(os.path.join(out_gradcams, image_name), out_img)