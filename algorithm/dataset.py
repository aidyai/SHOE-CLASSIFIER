import os
import json
from PIL import Image
import pandas as pd
from sklearn import model_selection, metrics
import torch
#from torch.utils.data import Dataset 

#from utils import CFG, FOLD_CFG

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split  #, KFold


from libraries import *


# Load configuration file
# Load the YAML file
with open('/notebooks/algorithm/config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

    test_size = config['config']['test_size']


### READ IN CSV FILE ### DEFINE DIRECTORIES
##########-----------##################
df = pd.read_csv("/notebooks/dataset/SHOE_NET.csv")
###########______________###############
df['Label'] = df['Label'].apply(lambda x: x[0:4]).astype('category').cat.codes
########
DF = df
### SPLIT FUNCTION
train_df, valid_df = train_test_split(DF, test_size=test_size, shuffle=True)


#test_df = pd.read_csv("/notebooks/pixels-CLS/RICE/DATA/TEST.csv", encoding='unicode_escape')
#sample_submission = pd.read_csv("/notebooks/pixels-CLS/RICE/DATA/SampleSubmission.csv", encoding='unicode_escape')
######################
#kf = KFold(n_splits=FOLD_CFG.SPLITS, shuffle=True, random_state=FOLD_CFG.SEED)
##############


####
TRAIN_PATH = "/notebooks/dataset/DATASET"

########
##TEST_PATH = "/notebooks/pixels-CLS/RICE/DATA/TEST-IMAGES"




# ====================================================
# Dataset
# ====================================================

   


class ShoeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['IMAGE_ID'].values
        self.labels = df['Label'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df) 

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(TRAIN_PATH, file_name)
        
        # Load the image using PIL.Image.open()
        with Image.open(file_path) as image:
            # Convert the image to RGB
            image = image.convert('RGB')
        
        
        # Convert the PIL image to a PyTorch tensor
        tensor_image = transforms.ToTensor()(image)

        # Apply the transformation (if any) to the tensor
        if self.transform:
            tensor_image = self.transform(tensor_image)
                        
        # Convert the label to a PyTorch tensor
        label = torch.tensor(self.labels[idx]).long()

        return tensor_image, label
    
    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

