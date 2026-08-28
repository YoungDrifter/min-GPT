"""Evaluate the pretrained baseline or a fine-tuned checkpoint."""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import BATCH_SIZE, CONTEXT_LENGTH, SEED, WANDB_PROJECT
from data import DialogueDataset, collate_examples
from model import GPT, get_device
from tokenizer import GPT2BPETokenizer


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate min-GPT on the full test split")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Fine-tuned checkpoint; omit it to evaluate pretrained DialoGPT-small",
    )
    parser.add_argument("--wandb", action="store_true", help="Log evaluation to W&B")
    parser.add_argument("--run-name", help="W&B run name")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        metavar="N",
        help="Log cumulative metrics every N evaluation batches",
    )
    return parser


def current_metrics(total_loss, total_tokens, total_samples):
    if total_tokens == 0:
        raise ValueError("test split contains no reply token")
    loss = total_loss / total_tokens
    return {
        "loss": loss,
        "perplexity": math.exp(loss),
        "samples": total_samples,
        "tokens": total_tokens,
    }


def evaluation_log_payload(batch_index, metrics):
    return {
        "evaluation/batch": batch_index,
        "evaluation/samples": metrics["samples"],
        "evaluation/tokens": metrics["tokens"],
        "test/loss": metrics["loss"],
        "test/perplexity": metrics["perplexity"],
    }


def update_evaluation_summary(summary, metrics):
    summary["test/final_loss"] = metrics["loss"]
    summary["test/final_perplexity"] = metrics["perplexity"]
    summary["evaluation/final_samples"] = metrics["samples"]
    summary["evaluation/final_tokens"] = metrics["tokens"]


def evaluate_model(
    model,
    loader,
    device,
    log_interval=50,
    progress_callback=None,
):
    if log_interval <= 0:
        raise ValueError("log_interval must be greater than 0")

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_samples = 0
    last_batch = 0

    try:
        with torch.no_grad():
            for batch_index, (input_ids, labels) in enumerate(loader, start=1):
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
                total_samples += input_ids.size(0)
                last_batch = batch_index

                if progress_callback and batch_index % log_interval == 0:
                    progress_callback(
                        batch_index,
                        current_metrics(total_loss, total_tokens, total_samples),
                    )

        result = current_metrics(total_loss, total_tokens, total_samples)
        if progress_callback and last_batch % log_interval != 0:
            progress_callback(last_batch, result)
        return result
    finally:
        if was_training:
            model.train()


def load_evaluation_model(checkpoint, device):
    model = GPT.from_pretrained()
    metadata = {"kind": "baseline", "step": None, "validation_loss": None}

    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "model" not in state:
            raise ValueError(f"checkpoint has no model state: {checkpoint}")
        model.load_state_dict(state["model"])
        metadata = {
            "kind": "fine-tuned",
            "step": state.get("step"),
            "validation_loss": state.get("val_loss"),
        }

    return model.to(device), metadata


def start_wandb(enabled, run_name, checkpoint, metadata, sample_count):
    if not enabled:
        return None

    import wandb

    run = wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        group="full-test-evaluation",
        job_type="evaluation",
        config={
            "model_kind": metadata["kind"],
            "checkpoint": str(checkpoint) if checkpoint else "pretrained",
            "checkpoint_step": metadata["step"],
            "checkpoint_validation_loss": metadata["validation_loss"],
            "split": "test",
            "samples": sample_count,
            "batch_size": BATCH_SIZE,
            "context_length": CONTEXT_LENGTH,
            "seed": SEED,
        },
    )
    run.define_metric("evaluation/batch")
    run.define_metric("evaluation/samples")
    run.define_metric("evaluation/tokens")
    run.define_metric("test/*", step_metric="evaluation/samples")
    return run


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.log_interval <= 0:
        parser.error("--log-interval must be greater than 0")

    torch.manual_seed(SEED)
    device = get_device()
    tokenizer = GPT2BPETokenizer.from_pretrained()
    dataset = DialogueDataset("test", tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_examples,
    )

    try:
        model, metadata = load_evaluation_model(args.checkpoint, device)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    run_name = args.run_name or (
        "best-test" if args.checkpoint is not None else "baseline-test"
    )
    wandb_run = start_wandb(
        args.wandb,
        run_name,
        args.checkpoint,
        metadata,
        len(dataset),
    )

    print(f"Using device: {device}")
    print(f"Evaluating {metadata['kind']} on {len(dataset):,} test samples")

    def report_progress(batch_index, metrics):
        print(
            f"evaluation batch {batch_index:4d} | "
            f"samples {metrics['samples']:4d} | "
            f"loss {metrics['loss']:.4f} | "
            f"perplexity {metrics['perplexity']:.4f}"
        )
        if wandb_run:
            wandb_run.log(evaluation_log_payload(batch_index, metrics))

    try:
        result = evaluate_model(
            model,
            loader,
            device,
            log_interval=args.log_interval,
            progress_callback=report_progress,
        )
        print(
            f"final test | samples {result['samples']:,} | "
            f"tokens {result['tokens']:,} | loss {result['loss']:.6f} | "
            f"perplexity {result['perplexity']:.6f}"
        )
        if wandb_run:
            update_evaluation_summary(wandb_run.summary, result)
    finally:
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    main()
