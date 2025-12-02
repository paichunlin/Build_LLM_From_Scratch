from __future__ import annotations

import torch
import numpy as np
from nn_utils import cross_entropy
from tqdm import tqdm

def get_batch(
    input_path: str, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.memmap(input_path, dtype=np.int32, mode='r')
    starting_idxs = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([
            torch.from_numpy((data[i : i + context_length]).astype(np.int32))
            for i in starting_idxs
    ])
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + context_length]).astype(np.int32))
            for i in starting_idxs
        ]
    )
    if "cuda" in device:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss(model, 
                  eval_iters,
                  test_input_path,
                  batch_size,
                  context_length,
                  device):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in tqdm(range(eval_iters), desc="running eval"):
        input_ids, labels = get_batch(test_input_path, batch_size, context_length, device)
        logits = model(input_ids)
        loss = cross_entropy(logits, labels)
        losses[k] = loss.item()

    mean_loss = losses.mean()    
    model.train()

    return mean_loss