import torch
from torchvision import models, transforms
from PIL import Image
# Load pretrained ResNet50
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.eval()
# ImageNet labels
labels = weights.meta["categories"]
# Find goldfish index
goldfish_index = labels.index("goldfish")
# Image preprocessing
preprocess = weights.transforms()
# Input image
image_path = input("Enter image path: ")
image = Image.open(image_path).convert("RGB")
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)
# Prediction
with torch.no_grad():
    output = model(input_batch)
probabilities = torch.nn.functional.softmax(output[0], dim=0)
goldfish_probability = probabilities[goldfish_index].item()
print(f"\nProbability of finding a goldfish:")
print(f"{goldfish_probability*100:.2f}%")
top5_prob, top5_catid = torch.topk(probabilities, 5)
print("\nTop 5 Predictions:")
for i in range(5):
    print(f"{labels[top5_catid[i]]:<20} {top5_prob[i].item()*100:.2f}%")