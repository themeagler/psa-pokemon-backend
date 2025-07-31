import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

# Load your image
img_path = "my_card.jpg"
image = Image.open(img_path).convert("RGB")

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0)

# Load pretrained model (placeholder)
model = models.resnet18(pretrained=True)
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

# Load your image
img_path = "my_card.jpg"
image = Image.open(img_path).convert("RGB")

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0)


# Load pretrained model (placeholder)
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 3)
model.eval()

with torch.no_grad():
    output = model(input_tensor)
    pred = torch.argmax(output, dim=1).item()

grade_map = {0: "PSA 8", 1: "PSA 9", 2: "PSA 10"}
print(f"🧠 Predicted Grade: {grade_map.get(pred, 'Unknown')}")
