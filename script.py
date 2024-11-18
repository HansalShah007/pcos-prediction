import streamlit as st
import torch
import os
import gdown
import joblib
import pandas as pd
import torch.nn.functional as F

from PIL import Image
from torch import nn
from torchvision import transforms
from xgboost import XGBClassifier

TARGET_WIDTH = 1024
TARGET_HEIGHT = 1024

# Making the directory for storing models if one does not exists
os.makedirs("models", exist_ok=True)

# ----------------- PyTorch CNN Model Definition -----------------
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1)

        self.final_output_channels = 64
        self.final_conv_width = (((((TARGET_WIDTH - 4) // 2) - 4) // 2) - 4) // 2
        self.final_conv_height = (((((TARGET_HEIGHT - 4) // 2) - 4) // 2) - 4) // 2

        self.fc1 = nn.Linear(self.final_output_channels * self.final_conv_width * self.final_conv_height, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.final_output_channels * self.final_conv_width * self.final_conv_height)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
cnn_model_path = 'models/pcos_cnn_model.pth'
if not os.path.exists("models/pcos_cnn_model.pth"):
    # URL to the shared Google Drive file
    url = 'https://drive.google.com/uc?id=1198wvqymPeJ8VLw5U3cVpHVEduATVq1B'

    # Download the model
    gdown.download(url, cnn_model_path, quiet=False)

# Load PyTorch model
cnn_model = CNN()
cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=torch.device('cpu')))
print("CNN Model Loaded")
cnn_model.eval()

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((TARGET_WIDTH, TARGET_HEIGHT)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  
])

# ----------------- XGBoost Model Loading -----------------

# xgb_model_path = 'models/pcos_xgb_model.json'
# if not os.path.exists("models/pcos_xgb_model.json"):
#     # URL to the shared Google Drive file
#     url = 'https://drive.google.com/uc?id=1lPOWayhWkJEbxdH8P3JehLThm5rillQg'

#     # Download the model
#     gdown.download(url, xgb_model_path, quiet=False)

# xgboost_model = XGBClassifier()
# xgboost_model.load_model(xgb_model_path)
# print("XGB Model Loaded")

# ----------------- Streamlit UI -----------------
st.title("PCOS Prediction Tool")
# st.write("Choose your input method for PCOS prediction:")

# # User input method
# input_method = st.radio(
#     "Select input method:",
#     ("Upload Image (Sonogram)", "Manual Input Features")
# )

# if input_method == "Upload Image (Sonogram)":
st.write("Upload an image to classify!")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    input_tensor = preprocess(image).unsqueeze(0)

    # Perform inference
    with st.spinner('Classifying...'):
        output = cnn_model(input_tensor)
        _, predicted_class = torch.max(output, 1)
        st.write(f"Predicted Class: {'Infected' if predicted_class.item()==0 else 'Not Infected'}")
        st.write(f"Raw Output: {output.tolist()}")

# elif input_method == "Manual Input Features":
#     st.write("Enter the following features for manual prediction:")

#     # List of features required by the XGBoost model
#     feature_names = xgboost_model.get_booster().feature_names

#     # Collect user input for each feature
#     input_data = {}
#     for feature in feature_names:
#         if "(Y/N)" in feature:
#             input_data[feature] = st.radio(feature, options=[0, 1])
#         else:
#             input_data[feature] = st.number_input(feature, value=0.0)

#     # Predict button
#     if st.button("Predict"):
#         # Convert input data to a DataFrame
#         input_df = pd.DataFrame([input_data])

#         # Perform inference
#         with st.spinner('Classifying...'):
#             prediction = xgboost_model.predict(input_df)
#             prediction_proba = xgboost_model.predict_proba(input_df)
#             st.write(f"Predicted Class: {'Infected' if prediction[0]==1 else 'Not Infected'}")
#             st.write(f"Prediction Probabilities: {prediction_proba[0]}")
