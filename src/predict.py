import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from PIL import Image
from torchvision import transforms
from src.model import get_model

def predict_image(image_path, model_path, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = get_model(len(classes), device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Transform image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)[0]
        top5_probs, top5_indices = torch.topk(probs, 5)
        top5_results = [(classes[idx], prob.item()) for idx, prob in zip(top5_indices, top5_probs)]
    
    return top5_results  # Returns [(class, confidence), ...]