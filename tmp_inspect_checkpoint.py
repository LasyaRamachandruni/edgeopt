import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

CKPT = 'mobilenetv2_cifar10.pth'
print('Checking checkpoint path:', CKPT, 'exists=', os.path.isfile(CKPT))
if not os.path.isfile(CKPT):
    raise SystemExit('Checkpoint not found')

ckpt = torch.load(CKPT, map_location='cpu')
print('Loaded checkpoint type:', type(ckpt))

if isinstance(ckpt, dict):
    keys = list(ckpt.keys())
    print('Top-level keys:', keys)
    # Print shapes for tensor-like entries
    for k in keys[:50]:
        v = ckpt[k]
        if hasattr(v, 'shape'):
            try:
                print(f"  {k}: tensor shape {v.shape}")
            except Exception:
                print(f"  {k}: (tensor-like, could not get shape)")

# Build model for CIFAR-10
model = torchvision.models.mobilenet_v2(weights=None)
in_feats = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_feats, 10)

loaded_model = None
# Try to load state dicts or full models
try:
    if isinstance(ckpt, dict) and ('state_dict' in ckpt or any(k.startswith('module.') or k in model.state_dict() for k in ckpt.keys())):
        state = ckpt.get('state_dict', ckpt)
        # Remove module. prefix if present
        new_state = {}
        for k, v in state.items():
            new_key = k.replace('module.', '')
            new_state[new_key] = v
        # Attempt strict load; capture missing/unexpected
        res = model.load_state_dict(new_state, strict=False)
        print('load_state_dict result:', res)
        loaded_model = model
    elif isinstance(ckpt, nn.Module):
        print('Checkpoint is a full nn.Module instance')
        loaded_model = ckpt
    else:
        print('Unknown checkpoint structure; attempting to load into model with strict=False')
        try:
            model.load_state_dict(ckpt, strict=False)
            loaded_model = model
        except Exception as e:
            print('Failed to load state_dict into model:', e)
except Exception as e:
    print('Exception while loading checkpoint into model:', e)

if loaded_model is None:
    raise SystemExit('Failed to create a PyTorch model from checkpoint')

# Run a quick CIFAR-10 evaluation in PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded_model = loaded_model.to(device)
loaded_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

correct = 0
total = 0
if __name__ == '__main__':
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = loaded_model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f'PyTorch eval accuracy: {100.0 * correct / total:.2f}%')
