"""文本生成与终端对话"""

import argparse
from pathlib import Path

import torch

from config import (
    CONTEXT_LENGTH,
    EOS_ID,
    MAX_NEW_TOKENS,
    REPETITION_PENALTY,
    SEED,
    TEMPERATURE,
    TOP_K,
)
from model import GPT, get_device
from tokenizer import GPT2BPETokenizer


def validate_sampling(max_new_tokens, temperature, top_k, repetition_penalty):
    if max_new_tokens <= 0:
        raise ValueError("max-new-tokens 必须大于 0")
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    if top_k <= 0:
        raise ValueError("top-k 必须大于 0")
    if repetition_penalty <= 0:
        raise ValueError("repetition-penalty 必须大于 0")


def encode_turn(tokenizer, text):
    if not text.strip():
        raise ValueError("prompt 不能为空")
    token_ids = tokenizer.encode(text)[-(CONTEXT_LENGTH - 1) :]
    return token_ids + [EOS_ID]


def trim_history(turns, max_tokens=CONTEXT_LENGTH):
    turns = [list(turn) for turn in turns]
    while len(turns) > 1 and sum(map(len, turns)) > max_tokens:
        turns.pop(0)
    if turns and len(turns[0]) > max_tokens:
        turns[0] = turns[0][-max_tokens:]
    return turns


def generate_ids(
    model,
    input_ids,
    eos_token_id=EOS_ID,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    repetition_penalty=REPETITION_PENALTY,
    device=torch.device("cpu"),
    generator=None,
):
    validate_sampling(max_new_tokens, temperature, top_k, repetition_penalty)
    if not input_ids:
        raise ValueError("输入 token 不能为空")

    tokens = list(input_ids)
    generated = []
    generator = generator or torch.Generator().manual_seed(SEED)

    for _ in range(max_new_tokens):
        window = tokens[-CONTEXT_LENGTH:]
        x = torch.tensor([window], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(x)[0, -1].float().cpu()

        for token_id in set(tokens):
            if logits[token_id] < 0:
                logits[token_id] *= repetition_penalty
            else:
                logits[token_id] /= repetition_penalty

        logits /= temperature
        k = min(top_k, logits.numel())
        top_logits, top_indices = torch.topk(logits, k)
        probabilities = torch.softmax(top_logits, dim=-1)
        sampled = torch.multinomial(probabilities, 1, generator=generator)
        next_id = top_indices[sampled].item()

        if next_id == eos_token_id:
            break
        generated.append(next_id)
        tokens.append(next_id)

    return generated


def load_model(checkpoint, device):
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"找不到 checkpoint：{checkpoint}")

    model = GPT.from_pretrained()
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state)
    return model.to(device).eval()


def reply_ids(model, tokenizer, prompt, args, device, generator):
    input_ids = encode_turn(tokenizer, prompt)
    return generate_ids(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        device=device,
        generator=generator,
    )


def run_generate(args, tokenizer, device):
    model = load_model(args.checkpoint, device)
    generator = torch.Generator().manual_seed(args.seed)
    generated = reply_ids(model, tokenizer, args.prompt, args, device, generator)
    print(tokenizer.decode(generated, skip_special_tokens=True).strip())


def run_chat(args, tokenizer, device):
    model = load_model(args.checkpoint, device)
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    print("输入 exit 或 quit 结束对话。")

    while True:
        try:
            prompt = input("You: ")
        except EOFError:
            print()
            break
        if prompt.strip().lower() in {"exit", "quit"}:
            break
        if not prompt.strip():
            print("Bot: prompt 不能为空")
            continue

        history.append(encode_turn(tokenizer, prompt))
        history = trim_history(history)
        context = [token_id for turn in history for token_id in turn]
        generated = generate_ids(
            model,
            context,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            device=device,
            generator=generator,
        )
        reply = tokenizer.decode(generated, skip_special_tokens=True).strip()
        print(f"Bot: {reply}")
        history.append(generated + [EOS_ID])
        history = trim_history(history)


def run_compare(args, tokenizer, device):
    input_ids = encode_turn(tokenizer, args.prompt)

    baseline = load_model(None, device)
    baseline_ids = generate_ids(
        baseline,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        device=device,
        generator=torch.Generator().manual_seed(args.seed),
    )
    del baseline
    if device.type == "mps":
        torch.mps.empty_cache()

    fine_tuned = load_model(args.checkpoint, device)
    fine_tuned_ids = generate_ids(
        fine_tuned,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        device=device,
        generator=torch.Generator().manual_seed(args.seed),
    )

    print("Baseline:", tokenizer.decode(baseline_ids, True).strip())
    print("Fine-tuned:", tokenizer.decode(fine_tuned_ids, True).strip())


def add_sampling_arguments(parser):
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--repetition-penalty", type=float, default=REPETITION_PENALTY)


def build_parser():
    parser = argparse.ArgumentParser(description="min-GPT 对话生成")
    commands = parser.add_subparsers(dest="command", required=True)

    generate_parser = commands.add_parser("generate", help="生成单轮回复")
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--checkpoint", type=Path)
    add_sampling_arguments(generate_parser)

    chat_parser = commands.add_parser("chat", help="开始多轮终端对话")
    chat_parser.add_argument("--checkpoint", type=Path)
    add_sampling_arguments(chat_parser)

    compare_parser = commands.add_parser("compare", help="比较原始与微调模型")
    compare_parser.add_argument("--prompt", required=True)
    compare_parser.add_argument("--checkpoint", type=Path, required=True)
    add_sampling_arguments(compare_parser)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_sampling(
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.repetition_penalty,
        )
        if hasattr(args, "prompt") and not args.prompt.strip():
            raise ValueError("prompt 不能为空")
        if args.checkpoint is not None and not args.checkpoint.is_file():
            raise FileNotFoundError(f"找不到 checkpoint：{args.checkpoint}")
        tokenizer = GPT2BPETokenizer.from_pretrained()
        device = get_device()
        print(f"Using device: {device}")
        if args.command == "generate":
            run_generate(args, tokenizer, device)
        elif args.command == "chat":
            run_chat(args, tokenizer, device)
        else:
            run_compare(args, tokenizer, device)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
