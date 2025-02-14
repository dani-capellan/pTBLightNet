import torch
from dataloader.dataloader import pTBredCISMDataset
from utils import print_and_log


def load_train_dataloader(configs, df, data, fold, clv_value=0, test=False):
    
    if(fold=='all'):
        # DataFrame
        df_train = df[(df['split']=='train')].reset_index()
    else:
        # DataFrame
        df_train = df[(df['split']=='train') & (df['fold_cv']!=fold)].reset_index()
    
    if('ensemble' in configs):
        # Adapt
        if (configs["ensemble"]["LAT"]["experiment_name"] in ["", None]):
            df_train = df_train[(df_train['filepath_AP']!="-1")]  # both views
        else:
            df_train = df_train[(df_train['filepath_AP']!="-1") & (df_train['filepath_LAT']!="-1")]  # both views
        # Dict
        train_data = pTBredCISMDataset(configs, df_train, data, do_transform=True if not(test) else False, one_hot_encoding=True, clinical_vars_value=clv_value)
    else:
        # Dict
        train_data = pTBredCISMDataset(configs, df_train, data, do_transform=True if not(test) else False, one_hot_encoding=True)
        
    # DataLoader and Steps
    if(test and configs['experimentEnv']['shap']):
        # DataLoader
        trainDataLoader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=True, num_workers=0)
        # Check no batch is just one sample
        for batch in trainDataLoader:
            if(len(batch[0]) == 1):
                print_and_log("WARNING: dropping last batch from trainDataLoader due to only having 1 sample in one of the batches", configs)
                trainDataLoader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=True, num_workers=0, drop_last=True)
                break
        # Steps
        trainSteps = len(trainDataLoader.dataset) // len(train_data)
        if trainSteps == 0:
            print_and_log("WARNING: trainSteps defined to 1 due to batch_size>len(train_data)", configs)
            trainSteps = 1
    else:
        # DataLoader
        trainDataLoader = torch.utils.data.DataLoader(train_data, batch_size=configs['experimentEnv']['batch_size'], shuffle=True, num_workers=0)
        # Check no batch is just one sample
        for batch in trainDataLoader:
            if(len(batch[0]) == 1):
                print_and_log("WARNING: dropping last batch from trainDataLoader due to only having 1 sample in one of the batches", configs)
                trainDataLoader = torch.utils.data.DataLoader(train_data, batch_size=configs['experimentEnv']['batch_size'], shuffle=True, num_workers=0, drop_last=True)
                break
        # Steps
        trainSteps = len(trainDataLoader.dataset) // configs['experimentEnv']['batch_size']
        if trainSteps == 0:
            print_and_log("WARNING: trainSteps defined to 1 due to batch_size>len(train_data)", configs)
            trainSteps = 1
        
    print_and_log(f"\t Total training images: {len(train_data)}", configs)
    
    return trainDataLoader, trainSteps
    
    
def load_val_dataloader(configs, df, data, fold, clv_value=0):
    
    if(fold=='all'):
        return None, None
    
    # DataFrame
    df_val = df[(df['split']=='train') & (df['fold_cv']==fold)].reset_index()
    
    if('ensemble' in configs):
        # Adapt
        if (configs["ensemble"]["LAT"]["experiment_name"] in ["", None]):
            df_val = df_val[(df_val['filepath_AP']!="-1")]  # both views
        else:
            df_val = df_val[(df_val['filepath_AP']!="-1") & (df_val['filepath_LAT']!="-1")]  # both views
        # Dict
        val_data = pTBredCISMDataset(configs, df_val, data, do_transform=False, one_hot_encoding=True, clinical_vars_value=clv_value)
        # DataLoader
        valDataLoader = torch.utils.data.DataLoader(val_data, batch_size=configs['experimentEnv']['batch_size'], shuffle=False, num_workers=0)
        # Check no batch is just one sample
        for batch in valDataLoader:
            if(len(batch[0]) == 1):
                print_and_log("WARNING: dropping last batch from valDataLoader due to only having 1 sample in one of the batches", configs)
                valDataLoader = torch.utils.data.DataLoader(val_data, batch_size=configs['experimentEnv']['batch_size'], shuffle=False, num_workers=0, drop_last=True)
                break
        # Steps
        valSteps = len(valDataLoader.dataset) // configs['experimentEnv']['batch_size']
        if valSteps == 0:
            print_and_log("WARNING: valSteps defined to 1 due to batch_size>len(train_data)", configs)
            valSteps = 1
    else:
        # Dict
        val_data = pTBredCISMDataset(configs, df_val, data, do_transform=False, one_hot_encoding=True)
        # DataLoader
        valDataLoader = torch.utils.data.DataLoader(val_data, batch_size=configs['experimentEnv']['batch_size'], shuffle=False, num_workers=0)
        # Check no batch is just one sample
        for batch in valDataLoader:
            if(len(batch[0]) == 1):
                print_and_log("WARNING: dropping last batch from valDataLoader due to only having 1 sample in one of the batches", configs)
                valDataLoader = torch.utils.data.DataLoader(val_data, batch_size=configs['experimentEnv']['batch_size'], shuffle=False, num_workers=0, drop_last=True)
                break
        # Steps
        valSteps = len(valDataLoader.dataset) // configs['experimentEnv']['batch_size']
        if valSteps == 0:
            print_and_log("WARNING: valSteps defined to 1 due to batch_size>len(train_data)", configs)
            valSteps = 1
        
    print_and_log(f"\t Total validation images: {len(val_data)}", configs)
    
    return valDataLoader, valSteps


def load_test_dataloader(configs, df, data, clv_value=0, return_df=False):
    # DataFrame
    df_test = df[(df['split']=='test')].reset_index()
    
    if('ensemble' in configs):
        # Adapt
        if (configs["ensemble"]["LAT"]["experiment_name"] in ["", None]):
            df_test = df_test[(df_test['filepath_AP']!="-1")]  # both views
        else:
            df_test = df_test[(df_test['filepath_AP']!="-1") & (df_test['filepath_LAT']!="-1")]  # both views
        # Dict
        test_data = pTBredCISMDataset(configs, df_test, data, do_transform=False, one_hot_encoding=True, clinical_vars_value=clv_value)
        # DataLoader
        testDataLoader = torch.utils.data.DataLoader(test_data, batch_size=configs['experimentEnv']['test_batch_size'], shuffle=False, num_workers=0)
        # Check no batch is just one sample
        for batch in testDataLoader:
            if(len(batch[0]) == 1):
                print_and_log("WARNING: dropping last batch from testDataLoader due to only having 1 sample in one of the batches", configs)
                testDataLoader = torch.utils.data.DataLoader(test_data, batch_size=configs['experimentEnv']['test_batch_size'], shuffle=False, num_workers=0, drop_last=True)
                break
        # Steps
        testSteps = len(testDataLoader.dataset) // configs['experimentEnv']['test_batch_size']
    else:
        # Dict
        test_data = pTBredCISMDataset(configs, df_test, data, do_transform=False, one_hot_encoding=True)
        # DataLoader
        testDataLoader = torch.utils.data.DataLoader(test_data, batch_size=configs['experimentEnv']['test_batch_size'], shuffle=False, num_workers=0)
        # Check no batch is just one sample
        for batch in testDataLoader:
            if(len(batch[0]) == 1):
                print_and_log("WARNING: dropping last batch from testDataLoader due to only having 1 sample in one of the batches", configs)
                testDataLoader = torch.utils.data.DataLoader(test_data, batch_size=configs['experimentEnv']['test_batch_size'], shuffle=False, num_workers=0, drop_last=True)
                break
        # Steps
        testSteps = len(testDataLoader.dataset) // configs['experimentEnv']['test_batch_size']
        
    print_and_log(f"\t Total testing images: {len(test_data)}", configs)
    
    if return_df:
        return testDataLoader, testSteps, df_test
    else:
        return testDataLoader, testSteps