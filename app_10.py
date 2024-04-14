import streamlit as st
import torch
from torchvision.transforms import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models  
from torchvision.datasets import ImageFolder  
from pathlib import Path

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = models.resnet50(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


model = torch.load('model_10.pt', map_location=torch.device('cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

data_dir = 'D:\waste2\Garbage_Classification'


dataset = ImageFolder(data_dir, transform=transform)

def predict_image(image):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    _, predicted_class = torch.max(output, 1)

    return predicted_class.item()
st.title("Garbage Classification Prediction")
st.write("Upload an image to classify it into one of the garbage categories.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    
    if st.button("Predict"):
        predicted_class = predict_image(image)
        classes = dataset.classes  
        result = classes[predicted_class]
        st.success(f"The image belongs to the category: {result}")
