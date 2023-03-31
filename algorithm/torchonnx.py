# Load the required packages and the ONNX model
import onnx
import onnxruntime
import torch
from PIL import Image
import torchvision.transforms as transforms



import libraries
import wandb
import argparse
import yaml

wandb.login()



#   # Define model
#SHOENET = timm.create_model("vit_base_patch16_384", pretrained=True)
#SHOENET.fc = nn.Sequential(
#    nn.Dropout(0.1),
#    nn.Linear(124, 5))    
#weights = torch.load('/notebooks/ONNX/model_.pth', map_location=torch.device('cpu'))
#SHOENET.load_state_dict(weights)
#


# Define model
#SHOENET = timm.create_model("vit_base_patch16_384", pretrained=True)
#SHOENET.fc = nn.Sequential(
#    nn.Dropout(0.1),
#    nn.Linear(124, 5))    

#checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
#SHOENET.load_state_dict(checkpoint['model_state_dict'])
#SHOENET.eval()

# Define model
SHOENET = models.convnext_base(pretrained=True)
SHOENET.head = nn.Sequential(
                 nn.Linear(64, 5))

checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
SHOENET.load_state_dict(checkpoint['model_state_dict'])
SHOENET.eval()




# Set the input shape of the model
input_shape = (1, 3, 512, 512)

# Create a dummy input tensor of the input shape
dummy_input = torch.randn(input_shape)

input_names = [ "actual_input_1" ]
output_names = [ "output1" ]

torch.onnx.export(SHOENET, dummy_input, "/notebooks/ONNX/oonx/oNeXt.onnx", verbose=True, input_names=input_names, output_names=output_names,export_params=True)