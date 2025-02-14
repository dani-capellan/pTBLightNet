import os
import torch
import losses
from timm.scheduler import create_scheduler
from utils import dotdict, print_and_log

def getModel(model_name, configs, device, in_channels=1, outputs=2, test=False):
    '''
    Function that returns a model given its input configs.
    Input:
        model_name: str
        configs: dict - with parameters & info about experiment
        in_channels: int - image channels
        outputs: int - output classes
    Output: 
        model: PyTorch model
    '''
    
    if (model_name=='pTBLightNetv5_BN_BN'):
        from models import pTBLightNetv5_BN_BN, CustomResBlockv3
        model = pTBLightNetv5_BN_BN(in_channels=in_channels, CustomResBlock=CustomResBlockv3, outputs=outputs)
    #### ResNets and torchvision models
    elif (model_name=='ResNet18'):
        from models import ResNet, ResBlock
        model = ResNet(in_channels=in_channels, resblock=ResBlock, repeat=[2, 2, 2, 2], useBottleneck=False, outputs=outputs)
    elif (model_name=='ResNet34'):
        from models import ResNet, ResBlock
        model = ResNet(in_channels=in_channels, resblock=ResBlock, repeat=[3, 4, 6, 3], useBottleneck=False, outputs=outputs)
    elif (model_name=='ResNet50'):
        from models import ResNet, ResBlock, ResBottleneckBlock
        model = ResNet(in_channels=in_channels, resblock=ResBottleneckBlock, repeat=[3, 4, 6, 3], useBottleneck=True, outputs=outputs)
    elif (model_name=='ResNet101'):
        from models import ResNet, ResBlock, ResBottleneckBlock
        model = ResNet(in_channels=in_channels, resblock=ResBottleneckBlock, repeat=[3, 4, 23, 3], useBottleneck=True, outputs=outputs)
    elif (model_name=='ResNet152'):
        from models import ResNet, ResBlock, ResBottleneckBlock
        model = ResNet(in_channels=in_channels, resblock=ResBottleneckBlock, repeat=[3, 8, 36, 3], useBottleneck=True, outputs=outputs)
    elif (model_name=='DenseNet121'):
        from torchvision import models
        model = models.densenet121()
        model.features.conv0 = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.classifier = torch.nn.Linear(in_features=1024, out_features=outputs, bias=True)
    elif (model_name=='EfficientNetB0'):
        from torchvision import models
        model = models.efficientnet_b0()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=outputs, bias=True)
    elif (model_name=='EfficientNetB1'):
        from torchvision import models
        model = models.efficientnet_b1()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=outputs, bias=True)
    elif (model_name=='EfficientNetB2'):
        from torchvision import models
        model = models.efficientnet_b2()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = torch.nn.Linear(in_features=1408, out_features=outputs, bias=True)
    elif (model_name=='EfficientNetB3'):
        from torchvision import models
        model = models.efficientnet_b3()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = torch.nn.Linear(in_features=1536, out_features=outputs, bias=True)
    elif (model_name=='EfficientNetB4'):
        from torchvision import models
        model = models.efficientnet_b4()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = torch.nn.Linear(in_features=1792, out_features=outputs, bias=True)
    elif (model_name=='EfficientNetB5'):
        from torchvision import models
        model = models.efficientnet_b5()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = torch.nn.Linear(in_features=2048, out_features=outputs, bias=True)
    elif (model_name=='EfficientNetB6'):
        from torchvision import models
        model = models.efficientnet_b6()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 56, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = torch.nn.Linear(in_features=2304, out_features=outputs, bias=True)
    elif (model_name=='EfficientNetB7'):
        from torchvision import models
        model = models.efficientnet_b7()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = torch.nn.Linear(in_features=2560, out_features=outputs, bias=True)
    elif (model_name=='EfficientNetv2_s'):
        from torchvision import models
        model = models.efficientnet_v2_s()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=outputs, bias=True)
    elif (model_name=='EfficientNetv2_m'):
        from torchvision import models
        model = models.efficientnet_v2_m()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=outputs, bias=True)
    elif (model_name=='EfficientNetv2_l'):
        from torchvision import models
        model = models.efficientnet_v2_l()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=outputs, bias=True)
    elif (model_name=='MobileNetv3_small'):
        from torchvision import models
        model = models.mobilenet_v3_small()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[3] = torch.nn.Linear(in_features=1024, out_features=outputs, bias=True)
    elif (model_name=='MobileNetv3_large'):
        from torchvision import models
        model = models.mobilenet_v3_large()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[3] = torch.nn.Linear(in_features=1280, out_features=outputs, bias=True)
    #### MLPS FOR ENSEMBLES
    elif (model_name=="MLP"):
        from models import MLP
        model = MLP(in_dim=8192, outputs=outputs)
    elif (model_name=="MLPv2"):
        from models import MLPv2
        model = MLPv2(in_dim_x1=4096,in_dim_x2=4096, outputs=outputs)
    elif (model_name=="MLPv2_clv"):
        from models import MLPv2_clv
        model = MLPv2_clv(in_dim_x1=4096,in_dim_x2=4096, in_dim_clv=len(configs['clinical_vars']['list']), outputs=outputs)
    elif (model_name=="MLPv3"):
        from models import MLPv3
        model = MLPv3(in_dim_x1=4096,in_dim_x2=4096, outputs=outputs)
    elif (model_name=="MLPv3_clv"):
        from models import MLPv3_clv
        model = MLPv3_clv(in_dim_x1=4096,in_dim_x2=4096, in_dim_clv=len(configs['clinical_vars']['list']), outputs=outputs)
    elif (model_name=="MLPv4"):
        from models import MLPv4
        model = MLPv4(in_dim_x1=4096,in_dim_x2=4096, outputs=outputs)
    elif (model_name=="MLPv4_clv"):
        from models import MLPv4_clv
        model = MLPv4_clv(in_dim_x1=4096,in_dim_x2=4096, in_dim_clv=len(configs['clinical_vars']['list']), outputs=outputs)
    elif (model_name=="MLPv2_onlyclv"):
        from models import MLPv2_onlyclv
        model = MLPv2_onlyclv(in_dim_clv=len(configs['clinical_vars']['list']), outputs=outputs)
    elif (model_name=="MLPv4_clv_wo_bn"):
        from models import MLPv4_clv_wo_bn
        model = MLPv4_clv_wo_bn(in_dim_x1=4096,in_dim_x2=4096, in_dim_clv=len(configs['clinical_vars']['list']), outputs=outputs)
    else:
        model = None

    # Pretrained - load weights
    if(not(test)):
        if(configs['pretrained']['enabled']):
            if(configs['pretrained']['model_ckpt']['path'] in ["", " "]):
                ckpt_path = os.path.join("./results",configs['pretrained']['model_ckpt']['experiment_name'],
                                        model_name,
                                        configs['pretrained']['model_ckpt']['optimizer_name'],
                                        configs['pretrained']['model_ckpt']['lossFn_name'],
                                        f"fold_{str(configs['pretrained']['model_ckpt']['fold'][model_name])}", "model_best.pth")
            else:
                ckpt_path = configs['pretrained']['model_ckpt']['path']
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = maybe_freeze_model(configs, model)
        
    model.to(device)

    return model


def maybe_freeze_model(configs, model):
    if(configs['pretrained']['enabled'] and configs['pretrained']['freeze']['enabled']):
        print("Freezing model layers")
        for name, para in model.named_parameters():
            if((name.split(".")[0]=="features") and (name.split(".")[1].startswith("layer")) and (name.split(".")[1][-1] in str(configs['pretrained']['freeze']['resblocks']))):
                para.requires_grad = False
    else:
        print("No model freezing")
            
    return model


def getOptimizer(optimizer_name, model_name, model, configs, test=False):
    '''
    Function that returns an optimizer given its input configs.
    Input:
        optimizer_name: str
        model: PyTorch model
        configs: dict - with parameters & info about experiment
    Output: 
        optimizer: PyTorch optimizer
    '''
    
    # if(configs['pretrained']['enabled'] and configs['pretrained']['freeze']['enabled']):
    #     parameters = [p for p in net.parameters() if p.requires_grad]
    
    if (optimizer_name=="Adam"):
        optimizer = torch.optim.Adam(
            params = model.parameters(),
            lr = configs['experimentEnv']['optim_args'][optimizer_name]['learning_rate'],
            betas = (configs['experimentEnv']['optim_args'][optimizer_name]['beta_1'], configs['experimentEnv']['optim_args'][optimizer_name]['beta_2']),
            weight_decay = configs['experimentEnv']['optim_args'][optimizer_name]['weight_decay']
        )
        
    elif (optimizer_name=="SGD"):
        optimizer = torch.optim.SGD(
            params = model.parameters(), 
            lr = configs['experimentEnv']['optim_args'][optimizer_name]['learning_rate'], 
            weight_decay = configs['experimentEnv']['optim_args'][optimizer_name]['weight_decay']
        )
    elif (optimizer_name=="SGD_momentum"):
        optimizer = torch.optim.SGD(
            params = model.parameters(), 
            lr = configs['experimentEnv']['optim_args'][optimizer_name]['learning_rate'], 
            momentum=0.9,
            weight_decay = configs['experimentEnv']['optim_args'][optimizer_name]['weight_decay']
        )
    elif (optimizer_name=="Nesterov"):
        optimizer = torch.optim.SGD(
            params = model.parameters(), 
            lr = configs['experimentEnv']['optim_args'][optimizer_name]['learning_rate'], 
            weight_decay = configs['experimentEnv']['optim_args'][optimizer_name]['weight_decay']
        )
    elif (optimizer_name=="Nesterov_momentum"):
        optimizer = torch.optim.SGD(
            params = model.parameters(), 
            lr = configs['experimentEnv']['optim_args'][optimizer_name]['learning_rate'], 
            momentum=0.9,
            weight_decay = configs['experimentEnv']['optim_args'][optimizer_name]['weight_decay'],
            nesterov = True
        )
    elif (optimizer_name=="RMSprop"):
        optimizer = torch.optim.RMSprop(
            params = model.parameters(), 
            lr = configs['experimentEnv']['optim_args'][optimizer_name]['learning_rate'], 
            weight_decay = configs['experimentEnv']['optim_args'][optimizer_name]['weight_decay']
        )
    else:
        optimizer = None
        
    # Pretrained - load weights
    if(not(test)):
        if(configs['pretrained']['enabled']):
            if(configs['pretrained']['model_ckpt']['path'] in ["", " "]):
                ckpt_path = os.path.join("./results",configs['pretrained']['model_ckpt']['experiment_name'],
                                        model_name,
                                        configs['pretrained']['model_ckpt']['optimizer_name'],
                                        configs['pretrained']['model_ckpt']['lossFn_name'],
                                        f"fold_{str(configs['pretrained']['model_ckpt']['fold'][model_name])}", "model_best.pth")
            else:
                ckpt_path = configs['pretrained']['model_ckpt']['path']
            checkpoint = torch.load(ckpt_path)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
    return optimizer


def getScheduler(configs, optimizer, optimizer_name):
    """
    timm.scheduler.sheduler_factory.scheduler_kwargs
    """
    eval_metric = "top1"
    plateau_mode = 'min' if 'loss' in eval_metric else 'max'
    args_sched = dict(
        optimizer=optimizer,
        sched=configs['experimentEnv']['optim_args']['scheduler']['name'],
        epochs=configs['experimentEnv']['epochs'],
        decay_epochs=30,
        warmup_epochs=5,
        cooldown_epochs=10,
        patience_epochs=10,
        decay_rate=0.1,
        lr=configs['experimentEnv']['optim_args'][optimizer_name]['learning_rate'],
        lr_noise=None,
        lr_noise_pct=0.67,
        lr_noise_std=1.0,
        warmup_lr=1e-6,
        min_lr=1e-5, 
        eval_metric=eval_metric,
        decay_milestones=[30, 60],
        warmup_prefix=False,
        seed=42,
        lr_cycle_mul=1.,
        lr_cycle_decay=0.1,
        lr_cycle_limit=1,
        lr_k_decay=1.0,
        plateau_mode=plateau_mode,
        sched_on_updates=False,
    )
    args_sched = dotdict(args_sched)
    lr_scheduler, _ = create_scheduler(args_sched, optimizer)
    return lr_scheduler


def getLossFn(lossFn_name):
    '''
    Function that returns a loss function given its input configs.
    Inputs:
        lossFn_name: str
    Outputs:
        lossFn: Loss function
    '''
    if(lossFn_name=='CrossEntropy'):
        lossFn = torch.nn.CrossEntropyLoss()
    elif(lossFn_name=='FocalLoss'):
        lossFn = losses.FocalLoss()
    return lossFn


def maybe_load_models_for_ensemble(configs):
    if 'ensemble' in configs:
        if 'models_ckpts_dir' not in configs['data_in']:
            model_ckpts_dir = configs['out_training_dir']
        else:
            model_ckpts_dir = configs['data_in']['models_ckpts_dir']
        try:
            for v in ['AP','LAT']:
                models_paths = []
                MODEL_DIR = os.path.join(model_ckpts_dir,configs['ensemble'][v]['experiment_name'],configs['ensemble'][v]['model_name'],configs['ensemble'][v]['optimizer_name'],configs['ensemble'][v]['lossFn_name'])
                for root, dirs, files in os.walk(MODEL_DIR):
                    if('model_best.pth' in files):
                        models_paths.append(os.path.join(root,'model_best.pth'))
                configs['ensemble'][v]['models_paths'] = sorted(models_paths)
        except:
            raise Exception("Please check models for ensemble in configs YAML file")
        
    return configs


def get_AP_LAT_models_for_ensemble(configs, device, fold, test=False):
    # Check
    if('ensemble' not in configs):
        raise Exception("No \'ensemble\' key in configs.")
    
    # AP & LAT models
    model_AP = getModel(configs['ensemble']['AP']['model_name'], configs, device, test=test)
    model_LAT = getModel(configs['ensemble']['LAT']['model_name'], configs, device, test=test)
    opt_AP = getOptimizer(configs['ensemble']['AP']['optimizer_name'], configs['ensemble']['AP']['model_name'], model_AP, configs, test=test)
    opt_LAT = getOptimizer(configs['ensemble']['LAT']['optimizer_name'], configs['ensemble']['LAT']['model_name'], model_LAT, configs, test=test)
    lossFn_AP = getLossFn(configs['ensemble']['AP']['lossFn_name'])
    lossFn_LAT = getLossFn(configs['ensemble']['LAT']['lossFn_name'])

    # Load model checkpoints
    ## AP
    model_path = [p for p in configs['ensemble']['AP']['models_paths'] if f"fold_{fold}" in p][0]
    checkpoint = torch.load(model_path)
    epoch = checkpoint['epoch']
    model_AP.load_state_dict(checkpoint['model_state_dict'])
    opt_AP.load_state_dict(checkpoint['optimizer_state_dict'])
    ap_ok = True
    ## LAT
    try:
        model_path = [p for p in configs['ensemble']['LAT']['models_paths'] if f"fold_{fold}" in p][0]
        checkpoint = torch.load(model_path)
        epoch = checkpoint['epoch']
        model_LAT.load_state_dict(checkpoint['model_state_dict'])
        opt_LAT.load_state_dict(checkpoint['optimizer_state_dict'])
        lat_ok = True
    except Exception as err:
        if ap_ok and configs['ensemble']['LAT']['experiment_name'] in ["", None]:
            model_LAT = None
            # print and log
            print_and_log("LAT model not found. Using only AP model", configs)
            lat_ok = False
        else:
            print(Exception, err)
            raise Exception("An error occurred while loading LAT model. Please check. See above for details.")    
    
    # Switch to eval mode
    model_AP.eval()
    if lat_ok:
        model_LAT.eval()
    
    return model_AP, model_LAT, opt_AP, opt_LAT, lossFn_AP, lossFn_LAT
