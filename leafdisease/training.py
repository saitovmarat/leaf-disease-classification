import time
from pathlib import Path
import torch
import torch.nn.functional as F

from leafdisease.utils import synchronize


def train_one_epoch(
    model, train_loader, optimizer, device,
    loss_fn, metric_fn    
):
    model.train()
    total_loss = 0.0
    num_samples = 0
    all_gt, all_preds = [], []
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        logits = model(x)
        loss = loss_fn(logits, y)
        
        loss.backward()
        optimizer.step()

        num_samples += len(y)
        total_loss += loss.item() * len(y)
        
        all_gt.append(y.detach())
        all_preds.append(logits.detach().argmax(dim=1))   

    gts = torch.cat(all_gt).cpu().numpy()
    preds = torch.cat(all_preds).cpu().numpy()
    
    score = metric_fn(gts, preds)
    return {
        "train_loss": total_loss / num_samples, 
        "train_metric": float(score)
    }


@torch.no_grad()
def evaluate(
    model, val_loader, device, 
    loss_fn, metric_fn
):
    model.eval()
    
    total_loss = 0.0
    num_samples = 0
    all_gt, all_preds = [], []
    
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = loss_fn(logits, y)
        
        num_samples += len(y)        
        total_loss += loss.item() * len(y)
        
        all_gt.append(y)
        all_preds.append(logits.argmax(dim=1))

    gts = torch.cat(all_gt).cpu().numpy()
    preds = torch.cat(all_preds).cpu().numpy()
    
    score = metric_fn(gts, preds)
    return {
        "val_loss": total_loss / num_samples, 
        "val_metric": float(score)
    }


def fit_model(
    model, epochs, train_loader, val_loader,
    optimizer, checkpoint_path, device, scheduler=None,
    *,
    loss_fn,
    metric_fn,
    monitor="val_metric",
    mode="max",
    run_config=None,
):
    if epochs < 1:
        raise ValueError("epochs должно быть положительным")

    if monitor not in {"val_loss", "val_metric"}:
        raise ValueError("monitor: 'val_loss' или 'val_metric'")

    if mode not in {"min", "max"}:
        raise ValueError("mode: 'min' или 'max'")

    device = torch.device(device)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    run_config = dict(run_config or {})

    history = []
    best_score = float("inf") if mode == "min" else -float("inf")
    best_loss = None
    best_state = None
    best_epoch = None
    total_train_seconds = 0.0

    num_parameters = sum(p.numel() for p in model.parameters())

    baseline = evaluate(
        model, val_loader, device,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
    )

    print(
        f"parameters={num_parameters:,} | "
        f"val before training: "
        f"loss={baseline['val_loss']:.3f}, "
        f"metric={baseline['val_metric']:.3f}"
    )

    for epoch in range(1, epochs + 1):
        learning_rates = [
            float(group["lr"])
            for group in optimizer.param_groups
        ]

        synchronize(device)
        started = time.perf_counter()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            loss_fn=loss_fn,
            metric_fn=metric_fn,
        )

        synchronize(device)
        train_seconds = time.perf_counter() - started
        total_train_seconds += train_seconds

        val_metrics = evaluate(
            model, val_loader, device,
            loss_fn=loss_fn,
            metric_fn=metric_fn,
        )

        history.append({
            "epoch": epoch,
            "learning_rates": learning_rates,
            "train_seconds": train_seconds,
            **train_metrics,
            **val_metrics,
        })

        score = float(val_metrics[monitor])

        if not math.isfinite(score):
            raise FloatingPointError(
                f"Некорректное значение {monitor} "
                f"на эпохе {epoch}: {score}"
            )

        improved = (
            score < best_score
            if mode == "min"
            else score > best_score
        )

        if improved:
            best_score = score
            best_loss = float(val_metrics["val_loss"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            torch.save({
                "state_dict": best_state,
                "model_class": type(model).__name__,
                "num_parameters": num_parameters,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "best_loss": best_loss,
                "monitor": monitor,
                "mode": mode,
                "run_config": run_config,
            }, checkpoint_path)

        print(
            f"epoch {epoch:2d}/{epochs} | "
            f"lr={learning_rates} | "
            f"train loss={train_metrics['train_loss']:.3f}, "
            f"metric={train_metrics['train_metric']:.3f} | "
            f"val loss={val_metrics['val_loss']:.3f}, "
            f"metric={val_metrics['val_metric']:.3f}"
        )

        # Для scheduler с шагом раз в эпоху.
        if scheduler is not None:
            scheduler.step()

    model.load_state_dict(best_state)
    model.eval()

    print(
        f"Best epoch: {best_epoch}; "
        f"{monitor}={best_score:.5f}; "
        f"training time: {total_train_seconds:.1f}s"
    )

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_loss": best_loss,
        "monitor": monitor,
        "mode": mode,
        "num_parameters": num_parameters,
        "train_seconds": total_train_seconds,
        "run_config": run_config,
    }