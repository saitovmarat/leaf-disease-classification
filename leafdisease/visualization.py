import random 
import matplotlib.pyplot as plt 
import torch


def show_samples(dataset, label_names, n=6, cols=2):
    indices = random.sample(range(len(dataset)), n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 4, rows * 4),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, index in zip(axes.flat, indices):
        image, label = dataset[index]
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().permute(1, 2, 0)

        ax.imshow(image)
        ax.set_title(label_names[str(label)], fontsize=9)
    plt.tight_layout(); plt.show()
    

def plot_history(result, title):
    history = result["history"]
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, train_key, val_key, ylabel in [
        (axes[0], "train_loss", "val_loss", "Loss (Cross-Entropy)"),
        (axes[1], "train_accuracy", "val_accuracy", "Accuracy"),
    ]:
        ax.plot(epochs, [row[train_key] for row in history], label="train")
        ax.plot(epochs, [row[val_key] for row in history], label="val")
        ax.axvline(result["best_epoch"], linestyle="--", color="gray", alpha=0.6, label="best epoch by loss")
        ax.set(xlabel="Epoch", ylabel=ylabel, title=f"{title}: {ylabel}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()