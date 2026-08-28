"""GPT-2 byte-level BPE tokenizer"""

import json
import re
import unicodedata
from pathlib import Path

from config import EOS_ID, MODEL_DIR

EOS_TOKEN = "<|endoftext|>"


def unicode_ranges(category):
    """生成类似正则 \p{L} 和 \p{N} 的 Unicode 区间。"""
    ranges = []
    start = None

    for codepoint in range(0x110000):
        matches = unicodedata.category(chr(codepoint)).startswith(category)
        if matches and start is None:
            start = codepoint
        elif not matches and start is not None:
            ranges.append((start, codepoint - 1))
            start = None

    def escape(codepoint):
        if codepoint <= 0xFFFF:
            return f"\\u{codepoint:04x}"
        return f"\\U{codepoint:08x}"

    return "".join(
        escape(start) if start == end else f"{escape(start)}-{escape(end)}"
        for start, end in ranges
    )


LETTERS = unicode_ranges("L")
NUMBERS = unicode_ranges("N")
TOKEN_PATTERN = re.compile(
    rf"'s|'t|'re|'ve|'m|'ll|'d| ?[{LETTERS}]+| ?[{NUMBERS}]+|"
    rf" ?[^\s{LETTERS}{NUMBERS}]+|\s+(?!\S)|\s+"
)


def bytes_to_unicode():
    """把 0-255 的 bytes 映射成可用于 BPE 的字符。"""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    unicode_values = byte_values.copy()
    extra = 0

    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            unicode_values.append(256 + extra)
            extra += 1

    return dict(zip(byte_values, map(chr, unicode_values)))


class GPT2BPETokenizer:
    def __init__(self, vocab, merges):
        self.encoder = vocab
        self.decoder = {token_id: token for token, token_id in vocab.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {char: byte for byte, char in self.byte_encoder.items()}
        self.bpe_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.cache = {}
        self.eos_token_id = EOS_ID

    @classmethod
    def from_pretrained(cls, model_dir=MODEL_DIR):
        model_dir = Path(model_dir)
        with (model_dir / "vocab.json").open(encoding="utf-8") as file:
            vocab = json.load(file)
        with (model_dir / "merges.txt").open(encoding="utf-8") as file:
            merges = [
                tuple(line.split())
                for line in file
                if line.strip() and not line.startswith("#")
            ]
        return cls(vocab, merges)

    @property
    def vocab_size(self):
        return len(self.encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = list(token)
        while len(word) > 1:
            pairs = set(zip(word, word[1:]))
            pair = min(pairs, key=lambda item: self.bpe_ranks.get(item, float("inf")))
            if pair not in self.bpe_ranks:
                break

            merged = []
            index = 0
            while index < len(word):
                if index < len(word) - 1 and (word[index], word[index + 1]) == pair:
                    merged.append(word[index] + word[index + 1])
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = merged

        self.cache[token] = word
        return word

    def tokenize(self, text):
        if text == EOS_TOKEN:
            return [EOS_TOKEN]

        tokens = []
        for word in TOKEN_PATTERN.findall(text):
            byte_text = "".join(self.byte_encoder[byte] for byte in word.encode("utf-8"))
            tokens.extend(self.bpe(byte_text))
        return tokens

    def encode(self, text):
        return [self.encoder[token] for token in self.tokenize(text)]

    def decode(self, ids, skip_special_tokens=False):
        tokens = []
        for token_id in ids:
            if skip_special_tokens and token_id == EOS_ID:
                continue
            token = self.decoder.get(token_id)
            if token is not None:
                tokens.append(token)

        byte_text = "".join(tokens)
        byte_values = bytes(self.byte_decoder[char] for char in byte_text)
        return byte_values.decode("utf-8", errors="replace")
