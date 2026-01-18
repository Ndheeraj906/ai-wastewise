"""
Model utilities for AI WasteWise.

- Provides a `load_model` function that loads a pretrained model (MobileNetV2)
  and a `predict_image` function that returns top-k predictions and confidences.

Note: For quick demo the code falls back to a dummy random predictor if torch or
weights are not available.
"""
import os
from PIL import Image
import numpy as np

LABELS = ["plastic", "paper", "metal", "glass", "organic", "ewaste"]

def preprocess_image(img: Image.Image, target_size=(224, 224)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    # reshape to CHW
    arr = np.transpose(arr, (2, 0, 1))
    arr = arr[np.newaxis, ...]
    return arr

try:
    import torch
    import torchvision.transforms as T
    from torchvision import models

    def load_model(weights_path=None, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = models.mobilenet_v2(pretrained=True)
        # replace classifier
        num_classes = len(LABELS)
        model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
        if weights_path and os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device

    def predict_image(model_tuple, pil_image, topk=3):
        model, device = model_tuple
        transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        x = transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
        topk_idx = probs.argsort()[-topk:][::-1]
        return [(LABELS[i], float(probs[i])) for i in topk_idx]

except Exception as e:
    # Fallback dummy predictor if torch not installed
    import random
    def load_model(weights_path=None, device=None):
        return ("dummy-model", "cpu")

    def predict_image(model_tuple, pil_image, topk=3):
        probs = np.random.dirichlet(np.ones(len(LABELS)), size=1)[0]
        topk_idx = probs.argsort()[-topk:][::-1]
        return [(LABELS[i], float(probs[i])) for i in topk_idx]