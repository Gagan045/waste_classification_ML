Waste Classification using Streamlit and ResNet50
This repository contains the code for a waste classification system developed using Streamlit and a machine learning model based on the ResNet50 architecture.
The test_3.ipynb is the resnet50 model for classifing waste in to organic,recycable and non recycable/

The test_10.ipynb is the resnet50 model for classifying waste into 10 categories:
1.Battery
2.Biological
3.Cardboard
4.Clothes
5.Glass
6.Metal
7.Paper
8.Plastic
9.Shoes
10.Trash

The model_3.h5 file is the weights stored after training the resnet50 model on the dataset.
The model_10.pt file is the weights stored after traing the resnet50 model on the dataset

Datasets:
3 waste classification dataset: https://www.kaggle.com/datasets/shubhamdivakar/waste-classification-dataset
12 Waste Classification dataset: https://www.kaggle.com/datasets/mostafaabla/garbage-classification

app_3.py file is for deploying in streamlit for 3 classification dataset
app_10.py file is for deploying in streamlit for 10 classification dataset
