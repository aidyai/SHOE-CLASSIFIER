## DESCRIPTION
________
The Shoe Classification App is a machine learning web application built using PyTorch and the model uses ONNX for inferencing. The app allows users to upload an image of a shoe and get a prediction on the shoe's class, such as "sneaker", "boot", "heels", "dressing shoe", "sandals". 

The app uses a machine learning model that has been trained on a dataset of shoe images to make its predictions. The app provides a simple and intuitive user interface, making it easy for anyone to use. 

The Shoe Classification App is a useful tool for anyone interested in shoe fashion or working in the shoe industry, as it can quickly and accurately classify different types of shoes.

## TESTING
____
To test the app locally, follow these steps:

1.  Clone the repository: `git clone https://github.com/aidyai/SHOE-CLASSIFIER.git`
2.  Install the required dependencies: `pip install -r requirements.txt`
3.  Start the server: `python server.py`
4.  Open your web browser and go to `http://localhost:8000`
5.  Upload an image of a shoe and get a prediction on the shoe's class!

```python

import uvicorn  

if __name__ == "__main__":

  uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True) 

```

