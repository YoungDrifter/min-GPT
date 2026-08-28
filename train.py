"""DialoGPT-small 微调"""

import argparse
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    ACCUMULATION_STEPS,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    GRAD_CLIP,
    LEARNING_RATE,
    MAX_STEPS,
    MIN_LEARNING_RATE,
    SEED,
    VALIDATION_BATCHES,
    VALIDATION_INTERVAL,
    WANDB_PROJECT,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from data import DialogueDataset, collate_examples
from model import GPT, get_device
from tokenizer import GPT2BPETokenizer


def build_parser():
    parser = argparse.ArgumentParser(description="微调 DialoGPT-small")
    parser.add_argument("--wandb", action="store_true", help="记录训练指标到 W&B")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        metavar="N",
        help="本次训练的 optimizer steps（默认 config.MAX_STEPS）",
    )
    return parser


def get_learning_rate(step, max_steps):
    """30 steps warmup，然后 cosine decay 到最小学习率。"""
    warmup_steps = min(WARMUP_STEPS, max_steps)
    if step <= warmup_steps:
        return LEARNING_RATE * step / warmup_steps

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LEARNING_RATE + cosine * (LEARNING_RATE - MIN_LEARNING_RATE)


def start_wandb(enabled, max_steps):
    if not enabled:
        return None

    import wandb

    return wandb.init(
        project=WANDB_PROJECT,
        config={
            "batch_size": BATCH_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "max_steps": max_steps,
            "peak_learning_rate": LEARNING_RATE,
            "min_learning_rate": MIN_LEARNING_RATE,
            "warmup_steps": min(WARMUP_STEPS, max_steps),
            "weight_decay": WEIGHT_DECAY,
            "validation_interval": VALIDATION_INTERVAL,
        },
    )


def compute_loss(model, input_ids, labels):
    logits = model(input_ids)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def evaluate(model, loader, device, max_batches=VALIDATION_BATCHES):
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    try:
        with torch.no_grad():
            for batch_index, (input_ids, labels) in enumerate(loader):
                if batch_index == max_batches:
                    break
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                logits = model(input_ids)
                targets = labels[:, 1:]
                total_loss += F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                ).item()
                total_tokens += (targets != -100).sum().item()
    finally:
        if was_training:
            model.train()

    if total_tokens == 0:
        raise ValueError("validation batch 中没有 reply token")
    return total_loss / total_tokens


def save_checkpoint(
    model,
    path,
    step,
    val_loss,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )


def main():
    args = build_parser().parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps 必须大于 0")

    torch.manual_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = GPT2BPETokenizer.from_pretrained()
    dataset = DialogueDataset("train", tokenizer)
    validation_dataset = DialogueDataset("validation", tokenizer)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_examples,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_examples,
    )

    model = GPT.from_pretrained().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    max_steps = args.max_steps if args.max_steps is not None else MAX_STEPS
    wandb_run = start_wandb(args.wandb, max_steps)

    best_val_loss = evaluate(model, validation_loader, device)
    save_checkpoint(model, CHECKPOINT_DIR / "latest.pt", 0, best_val_loss)
    save_checkpoint(model, CHECKPOINT_DIR / "best.pt", 0, best_val_loss)
    print(f"validation step {0:4d} | loss {best_val_loss:.4f}")
    if wandb_run:
        wandb_run.log(
            {
                "validation/loss": best_val_loss,
                "validation/perplexity": math.exp(best_val_loss),
            },
            step=0,
        )

    model.train()
    batches = iter(loader)
    for step in range(1, max_steps + 1):
        learning_rate = get_learning_rate(step, max_steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for _ in range(ACCUMULATION_STEPS):
            try:
                input_ids, labels = next(batches)
            except StopIteration:
                batches = iter(loader)
                input_ids, labels = next(batches)

            input_ids = input_ids.to(device)
            labels = labels.to(device)
            loss = compute_loss(model, input_ids, labels)
            (loss / ACCUMULATION_STEPS).backward()
            step_loss += loss.item()

        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        train_loss = step_loss / ACCUMULATION_STEPS
        print(f"step {step:4d} | loss {train_loss:.4f}")
        if wandb_run:
            wandb_run.log(
                {
                    "train/loss": train_loss,
                    "train/learning_rate": learning_rate,
                    "train/gradient_norm": gradient_norm.item(),
                },
                step=step,
            )

        if step % VALIDATION_INTERVAL == 0 or step == max_steps:
            val_loss = evaluate(model, validation_loader, device)
            save_checkpoint(
                model, CHECKPOINT_DIR / "latest.pt", step, val_loss
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, CHECKPOINT_DIR / "best.pt", step, val_loss
                )
            print(f"validation step {step:4d} | loss {val_loss:.4f}")
            if wandb_run:
                wandb_run.log(
                    {
                        "validation/loss": val_loss,
                        "validation/perplexity": math.exp(val_loss),
                    },
                    step=step,
                )

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
