import torch
import time
from torch import nn 
import torch.nn.functional as F
from pathlib import Path


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += len(y)
    return {"loss": total_loss / total, "accuracy": correct / total}

def fit_model(model, train_loader, val_loader):
    seed_everything(SEED)

    # Настраиваем оптимизатор
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Оцениваем модель до обучения
    baseline = evaluate(model, val_loader)
    print(f"P={model.patch_size} | параметров={sum(p.numel() for p in model.parameters()):,} | "
          f"валидация до обучения: потери={baseline['loss']:.3f}, доля верных ответов={baseline['accuracy']:.1%}")

    # Инициализируем логгеры метрик.
    history = []
    best_loss, best_state, best_epoch = float("inf"), None, None
    train_seconds = 0.0
    synchronize()
    wall_start = time.perf_counter()

    # Обучение.
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        synchronize()
        start = time.perf_counter()

        for batch in train_loader:
            x, y = batch["image"].to(device), batch["label"].to(device)

            logits = model(x)

            loss = F.cross_entropy(logits, y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # логгируем метрики.
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(-1) == y).sum().item()
            total += len(y)

        # Вывод метрик и оценка модели после эпохи обучения.
        synchronize()
        epoch_seconds = time.perf_counter() - start
        train_seconds += epoch_seconds
        validation = evaluate(model, val_loader)
        history.append(dict(epoch=epoch, train_loss=total_loss / total, train_acc=correct / total,
                            val_loss=validation["loss"], val_acc=validation["accuracy"]))

        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step()
        print(f"эпоха {epoch:2d}/{EPOCHS} | потери на обучении {total_loss/total:.3f}, доля верных {correct/total:.1%} | "
              f"потери на валидации {validation['loss']:.3f}, доля верных {validation['accuracy']:.1%} | обучение {epoch_seconds:.1f}с")

    synchronize()
    wall_seconds = time.perf_counter() - wall_start
    model.load_state_dict(best_state)
    model.eval()
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "config": model_config(model.patch_size),
                "seed": SEED, "best_epoch": best_epoch, "model_class": type(model).__name__},
               f"checkpoints/{type(model).__name__}_p{model.patch_size}.pt")
    best = history[best_epoch - 1]
    print(f"Лучшая эпоха: {best_epoch}; обучение {train_seconds:.1f}с; полное время {wall_seconds:.1f}с")
    return dict(model=model, history=history, baseline=baseline, best_epoch=best_epoch,
                val_loss=best["val_loss"], val_acc=best["val_acc"], train_seconds=train_seconds,
                wall_seconds=wall_seconds, parameters=sum(p.numel() for p in model.parameters()),
                updates=EPOCHS * len(train_loader))

