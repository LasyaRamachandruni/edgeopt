import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms

def prune_mobilenet(model, amount=0.1, min_layer=5):
    for idx, block in enumerate(model.features):
        if idx >= min_layer:
            for submodule in block.modules():
                if isinstance(submodule, nn.Conv2d):
                    prune.ln_structured(submodule, name="weight", amount=amount, n=2, dim=0)
                    prune.remove(submodule, 'weight')
    return model

def finetune(model, trainloader, testloader, epochs=3, lr=0.001, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    print(f"Test accuracy after prune+retrain: {100*correct/total:.2f}%")
    return model

def main(weights_path='mobilenetv2_cifar10.pth', prune_amount=0.1, min_prune_layer=5, output_onnx='mobilenetv2_pruned_finetuned.onnx', epochs=3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    model = torchvision.models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))

    model = prune_mobilenet(model, amount=prune_amount, min_layer=min_prune_layer)
    model = finetune(model, trainloader, testloader, epochs=epochs, lr=0.001, device=device)

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Exported pruned and fine-tuned model to {output_onnx}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='mobilenetv2_cifar10.pth', help='Path to fine-tuned weights')
    parser.add_argument('--prune-amount', type=float, default=0.1, help='Fraction of channels to prune')
    parser.add_argument('--min-prune-layer', type=int, default=5, help='Starting features index to prune')
    parser.add_argument('--output', default='mobilenetv2_pruned_finetuned.onnx', help='ONNX export path')
    parser.add_argument('--epochs', type=int, default=3, help='Epochs to fine-tune after pruning')
    args = parser.parse_args()
    main(args.weights, args.prune_amount, args.min_prune_layer, args.output, args.epochs)

