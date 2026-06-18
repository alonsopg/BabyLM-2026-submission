from __future__ import annotations

import torch


def evaluate_mlm_loss(model, loader, device, max_batches: int) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            batch.pop("mask_metadata", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            out = model(**batch)
            losses.append(float(out.loss.detach().cpu()))
            if step + 1 >= max_batches:
                break
    model.train()
    return sum(losses) / max(1, len(losses))
