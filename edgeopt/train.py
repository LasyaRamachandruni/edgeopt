import time

import torch
import torch.nn as nn


def finetune(model, trainset, epochs=1, lr=1e-3, batch_size=64, log_every=50):
    """Short recovery fine-tune after pruning (Adam, cross-entropy, CPU)."""
    loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        running, start = 0.0, time.time()
        for step, (x, y) in enumerate(loader, 1):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item()
            if step % log_every == 0:
                print(f"epoch {epoch + 1} step {step}/{len(loader)} loss {running / step:.4f} ({time.time() - start:.0f}s)")
        print(f"epoch {epoch + 1} done, mean loss {running / len(loader):.4f}")
    return model.eval()


@torch.no_grad()
def accuracy_torch(model, dataset, batch_size=128):
    model.eval()
    correct = total = 0
    for x, y in torch.utils.data.DataLoader(dataset, batch_size=batch_size):
        correct += (model(x).argmax(1) == y).sum().item()
        total += len(y)
    return 100.0 * correct / total
