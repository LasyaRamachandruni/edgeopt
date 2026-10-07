import onnxruntime

from edgeopt.data import fixed_subset, load_cifar10, numpy_batches


def accuracy_onnx(model_path, dataset, batch_size=64):
    sess = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    correct = total = 0
    for images, labels in numpy_batches(dataset, batch_size):
        logits = sess.run(None, {input_name: images})[0]
        correct += int((logits.argmax(axis=1) == labels).sum())
        total += len(labels)
    return 100.0 * correct / total


def eval_onnx_classifier(model_path, batch_size=64, subset=None, img_size=224, data_root="./data"):
    """Top-1 accuracy on the CIFAR-10 test set, or a fixed seeded subset of it."""
    testset = fixed_subset(load_cifar10(train=False, img_size=img_size, root=data_root), subset)
    acc = accuracy_onnx(model_path, testset, batch_size)
    print(f"Top-1 accuracy on {len(testset)} CIFAR-10 test images: {acc:.2f}%")
    return acc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to ONNX model file")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--subset", type=int, default=None, help="Evaluate on a fixed subset of N test images")
    args = parser.parse_args()
    eval_onnx_classifier(args.model, batch_size=args.batch_size, subset=args.subset)
