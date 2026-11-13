import torch
import torch.nn.functional as F
from pathlib import Path


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss, total_accuracy = 0.0, 0.0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)

        loss = F.cross_entropy(logits, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * len(y)
        total_accuracy += (logits.argmax(-1) == y).sum().item()

    num_samples = len(train_loader.dataset)
    return {
        "train_loss": total_loss / num_samples, 
        "train_accuracy": total_accuracy / num_samples
    }

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss, total_accuracy = 0.0, 0.0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        total_accuracy += (logits.argmax(-1) == y).sum().item()

    num_samples = len(val_loader.dataset)
    return {
        "val_loss": total_loss / num_samples, 
        "val_accuracy": total_accuracy / num_samples
    }

def fit_model(
    model, epochs, train_loader, val_loader,
    optimizer, checkpoint_path, device, scheduler=None
):
    history = []
    best_loss, best_state, best_epoch = float("inf"), None, None
    
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer)
        val_metrics = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch, **train_metrics, **val_metrics
        })
        if val_metrics["val_loss"] < best_loss:
            best_loss, best_epoch = val_metrics["val_loss"], epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if scheduler is not None:
            scheduler.step()
        
        print(
            f"epoch {epoch:2d}/{epochs} | "
            f"train loss {train_metrics["train_loss"]}, train accuracy {train_metrics["train_accuracy"]} | "
            f"val loss {val_metrics["val_loss"]}, val accuracy {val_metrics["val_accuracy"]}"
        )

    model.load_state_dict(best_state)
    model.eval()

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": best_state, "model_class": type(model).__name__,
            "best_epoch": best_epoch, "best_loss": best_loss
        }, checkpoint_path
    )
    print(f"Best epoch: {best_epoch}; best loss: {best_loss:.5f}")

    return {"model": model, "history": history, "best_epoch": best_epoch, "best_loss": best_loss}
