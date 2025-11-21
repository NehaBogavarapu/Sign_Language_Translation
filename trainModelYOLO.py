from tkinter import Image
from ultralytics import YOLO
from IPython import display

import gdown
import os
import zipfile

# Download a sample dataset
url = 'https://drive.google.com/FFFFFFFFFFFFFFFFF'
file_id = url.split('id=')[1]

gdown.download(id=file_id, output='sample_dataset.zip', quiet=False)

# Unzip the dataset
with zipfile.ZipFile('sample_dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('sample_dataset')
os.remove('sample_dataset.zip')


model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data="sample_dataset/data.yaml",  # This will be changed!!!!!!
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="cpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

# Visualize the results
display.display(Image("/content/rus/runs/detect/train/confusion_matrix.png", width=600))



