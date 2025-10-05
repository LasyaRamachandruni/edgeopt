import torch
import onnxruntime
import numpy as np
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

def eval_onnx_classifier(model_path, batch_size=64, max_batches=None):
    # Prepare CIFAR-10 test data loader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to fit MobileNetV2
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    ort_session = onnxruntime.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name

    total, correct = 0, 0
    for i, (images, labels) in enumerate(tqdm(testloader)):
        if max_batches and i >= max_batches:
            break
        # ONNX expects numpy
        outputs = ort_session.run(None, {input_name: images.numpy()})[0]
        preds = np.argmax(outputs, axis=1)
        labels_np = labels.numpy()
        correct += (preds == labels_np).sum()
        total += len(labels)

    acc = correct / total * 100
    print(f"Top-1 Accuracy on CIFAR-10 test set: {acc:.2f}%")
    return acc

if __name__ == "__main__":
    import torch
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help="Path to ONNX model file")
    parser.add_argument('--batch-size', type=int, default=64, help="Evaluation batch size")
    parser.add_argument('--max-batches', type=int, default=None, help="Max eval batches (for quick check)")
    args = parser.parse_args()

    eval_onnx_classifier(args.model, batch_size=args.batch_size, max_batches=args.max_batches)
