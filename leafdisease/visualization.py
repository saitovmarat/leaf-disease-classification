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
        sample = dataset[index]
        image = sample["image"]
        
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().permute(1, 2, 0)

        ax.imshow(image)
        ax.set_title(
            label_names[str(sample["label"])],
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()
    
    
def plot_history(result, title):
    '''
    Вспомогательная функция построения графиков обучения.
    '''
    h = result["history"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for ax, train_key, val_key, ylabel in [
        (axes[0], "train_loss", "val_loss", 'Кросс-энтропия'),
        (axes[1], "train_acc", "val_acc", 'Доля верных ответов')
    ]:
        ax.plot([r["epoch"] for r in h], [r[train_key] for r in h], label='обучение (в течение эпохи)')
        ax.plot([r["epoch"] for r in h], [r[val_key] for r in h], label='валидация')
        ax.axvline(result["best_epoch"], ls="--", color="gray", alpha=0.6, label='выбранная эпоха')
        ax.set(xlabel='Эпоха', ylabel=ylabel, title=f"{title}: {ylabel}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()