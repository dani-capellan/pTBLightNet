import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
    
class GenericCXRDataset(Dataset):
    """
    Generic Dataset class
    Info taken from dict (.pkl)
    """

    def __init__(self, df, data, configs, do_transform=True, one_hot_encoding=True, apply_clahe=True, mean_norm=0.5, std_norm=0.5):
        self.df = df
        self.data = data
        self.configs = configs
        self.do_transform = do_transform
        self.one_hot_encoding = one_hot_encoding
        self.apply_clahe = apply_clahe
        self.random_state = configs['random_state']
        self.mean_norm=mean_norm
        self.std_norm=std_norm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # Get Image - ALL AP & without CLAHE (CLAHE applied in preprocess before CNN)
        key = "filename"
        img_path = self.df[key][idx]
        img = self.data[img_path]
        
        # Get Label
        label = self.df['pathologic'][idx]

        # Get highest value of img
        max_value = np.max(img)

        # Optional one hot encoding
        if self.one_hot_encoding:
            label = torch.nn.functional.one_hot(torch.from_numpy(np.array(label)),num_classes=self.configs['experimentEnv']['num_classes']) 
        else:
            label = torch.LongTensor([label])

        transform = A.Compose([
            A.CLAHE(clip_limit=(0.01*max_value,0.01*max_value), p=1) if self.apply_clahe else None,
            A.HorizontalFlip(p=0.5) if self.do_transform else None,  # Only during training
            A.ShiftScaleRotate(shift_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5) if self.do_transform else None,
            A.Resize(height=self.configs['img_dim_ai'],width=self.configs['img_dim_ai']),  # Always applied
            A.Normalize(mean=self.mean_norm,std=self.std_norm),  # Always applied. Normalization is applied by the formula: img = (img - mean * max_pixel_value) / (std * max_pixel_value).
            ToTensorV2(),  # Always applied
        ])

        # Apply final transformation
        img_tr = transform(image=img)

        return img_tr['image'], label
    
class pTBredCISMDataset(Dataset):
    """pTB - pTBred & CISM dataset from dict"""

    def __init__(self, configs, df, data, do_transform=True, one_hot_encoding=True, apply_clahe=True, tta=False, clinical_vars_value=0):
        self.configs = configs
        self.df = df
        self.view = self.configs['experimentEnv']['view']
        self.data = data
        self.do_transform = do_transform
        self.one_hot_encoding = one_hot_encoding
        self.apply_clahe = apply_clahe
        self.tta = tta
        self.random_state = self.configs['random_state']
        self.return_clinical_vars = self.configs['clinical_vars']['enabled']
        if self.configs['clinical_vars']['enabled']:
            self.clinical_vars = self.configs['clinical_vars']['list']
        else:
            self.clinical_vars = None
        self.clinical_vars_value = clinical_vars_value

        if(self.tta):
            assert self.do_transform == True, 'If TTA is enabled, do_transform should be enabled as well!'

    def __len__(self):
        return len(self.df)

    def process_clinical_vars(self, idx):

        clinical_vars_array = []
        for clv in self.clinical_vars:
            if(self.df.iloc[idx][clv] == 'yes'):
                clinical_vars_array.append(self.clinical_vars_value)
            elif(self.df.iloc[idx][clv] == 'no'):
                clinical_vars_array.append(-self.clinical_vars_value)
            else:
                # clinical_vars_array.append(np.mean(self.configs['clinical_vars']['values']).astype(int))
                clinical_vars_array.append(0)

        return np.array(clinical_vars_array)

    def __getitem__(self, idx):

        if self.view == 'AP-LAT':  # AP & LAT

            imgs = {}  # Initialize

            for v in ['AP', 'LAT']:

                # Get image
                pid = self.df.loc[idx,'patient_id']
                img = self.data[pid][v]
                # Get Label
                label = self.df['TB_class'][idx]

                # Get highest value of img
                max_value = np.max(img)

                # Optional one hot encoding
                if self.one_hot_encoding:
                    label = torch.nn.functional.one_hot(torch.from_numpy(
                        np.array(label)), num_classes=self.configs['experimentEnv']['num_classes'])
                else:
                    label = torch.LongTensor([label])

                if(self.tta):
                    # Warning! With this, all images have same transformation!
                    random.seed(self.random_state)

                transform = A.Compose([
                    A.CLAHE(clip_limit=(0.01*max_value, 0.01*max_value),
                            p=1) if self.apply_clahe else None,
                    # Only during training
                    A.HorizontalFlip(
                        p=0.5) if self.do_transform and self.tta == False else None,
                    A.ShiftScaleRotate(shift_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT,
                                       value=0, p=0.5) if self.do_transform else None,
                    # Always applied
                    A.Resize(height=self.configs['img_dim_ai'],
                             width=self.configs['img_dim_ai']),
                    A.Normalize(mean=self.configs['experimentEnv']['norm'][v]['mean'],std=self.configs['experimentEnv']['norm'][v]['std']),  # Always applied
                    ToTensorV2(),  # Always applied
                ])

                # Apply final transformation
                img_tr = transform(image=img)

                # Store in mini dict
                imgs[v] = img_tr['image']

            if(self.return_clinical_vars):
                return imgs['AP'], imgs['LAT'], self.process_clinical_vars(idx), label
            else:
                return imgs['AP'], imgs['LAT'], label

        else:  # AP OR LAT

            # Get image
            pid = self.df.loc[idx,'patient_id']
            img = self.data[pid][self.view]
            # Get Label
            label = self.df['TB_class'][idx]

            # Get highest value of img
            max_value = np.max(img)

            # Optional one hot encoding
            if self.one_hot_encoding:
                label = torch.nn.functional.one_hot(torch.from_numpy(
                    np.array(label)), num_classes=self.configs['experimentEnv']['num_classes'])
            else:
                label = torch.LongTensor([label])

            if(self.tta):
                # Warning! With this, all images have same transformation!
                random.seed(self.random_state)

            transform = A.Compose([
                A.CLAHE(clip_limit=(0.01*max_value, 0.01*max_value),
                        p=1) if self.apply_clahe else None,
                # Only during training
                A.HorizontalFlip(
                    p=0.5) if self.do_transform and self.tta == False else None,
                A.ShiftScaleRotate(shift_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT,
                                   value=0, p=0.5) if self.do_transform else None,
                # Always applied
                A.Resize(height=self.configs['img_dim_ai'],
                         width=self.configs['img_dim_ai']),
                A.Normalize(mean=self.configs['experimentEnv']['norm'][self.view]['mean'],std=self.configs['experimentEnv']['norm'][self.view]['std']),  # Always applied
                ToTensorV2(),  # Always applied
            ])

            # Apply final transformation
            img_tr = transform(image=img)

            if(self.return_clinical_vars):
                return img_tr['image'], self.process_clinical_vars(idx), label
            else:
                return img_tr['image'], label