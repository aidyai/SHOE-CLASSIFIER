import pathlib
import uvicorn

from typing import Optional
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Load the required packages and the ONNX model
import io
from PIL import Image
import onnx
import onnxruntime
import numpy as np
import torch
import torchvision.transforms as transforms



BASE_DIR = pathlib.Path(__file__).parent
app = FastAPI()



app.mount("/static", StaticFiles(directory= BASE_DIR/"static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR/"templates"))
onnx_pth = "./src/_store/oNeXt.onnx"



ort_session = onnxruntime.InferenceSession(onnx_pth)

## Preprocess the input image data
preprocess = transforms.Compose([
   transforms.Resize((512, 512)),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# Add your FastAPI routes and functions below
@app.get("/", response_class=HTMLResponse)
async def home(request:Request):
    return templates.TemplateResponse("index.html", {"request":request})



# Defining the endpoint that will receive the image file and perform predictions


@app.post("/results", response_class=HTMLResponse)
async def results(request:Request, file: UploadFile = File(...) ):
    # Read the image file into memory
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Apply the image transformations
    input_data = preprocess(image).unsqueeze(0)

    # Make a prediction using the model
    # Run the inference
    ort_inputs = {ort_session.get_inputs()[0].name: input_data.detach().numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    # Convert the output to a probability score
    # Get class predictions and probability scores
    class_names = ["Boot", "Dressing Shoe", "Heels", "Sandals", "Sneakers"]
    class_predictions = np.argmax(ort_outputs[0], axis=1)
    probability_scores = np.exp(ort_outputs[0]) / np.sum(np.exp(ort_outputs[0]), axis=1, keepdims=True)
    probs = probability_scores[0, class_predictions.item()]
    percentage = round(probs * 100, 2) 
    #percentage = str(percnt + "%"
    aboveText = "Prediction"
    belowText = class_names[class_predictions.item()]
    return templates.TemplateResponse("results.html", {"request": request,"percentage": str(percentage) +"%", "aboveText":aboveText, "belowText":belowText})


