import numpy as np
import torch

from .impl_0 import PlainSentences, TokenManager
from .impl_1 import run_base_training
from argparse import ArgumentParser
from torch import nn


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()

        assert (hidden_dim % num_heads) == 0

        self._num_heads = num_heads
        self._query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._projection = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        batch_size, input_length, last_dim = x.shape

        assert (last_dim % self._num_heads) == 0
        attention_dim = last_dim // self._num_heads

        q = self._query(x).view(batch_size, input_length, self._num_heads, attention_dim).transpose(1, 2)
        k = self._key(x).view(batch_size, input_length, self._num_heads, attention_dim).transpose(1, 2)
        v = self._value(x).view(batch_size, input_length, self._num_heads, attention_dim).transpose(1, 2)

        pre_normalization_weights = q @ k.transpose(2, 3) / np.sqrt(attention_dim)

        # i文字目について、i文字目以降の情報を利用しないように-infで埋めておく。
        mask = torch.tril(torch.ones((1, 1, input_length, input_length))).to(pre_normalization_weights.device)
        mask = mask.to('cuda' if mask.is_cuda else 'cpu')
        masked_weights = pre_normalization_weights.masked_fill(mask == 0, float('-inf'))

        # softmaxをかける。-infで埋めたところは0になる。
        normalized_weights = nn.functional.softmax(masked_weights, dim=-1)

        attention = normalized_weights @ v
        attention = attention.transpose(1, 2).contiguous().view(batch_size, input_length, last_dim)

        out = self._projection(attention)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()

        self._attention = MaskedMultiHeadAttention(hidden_dim, num_heads)
        self._mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self._norm_0 = nn.LayerNorm(hidden_dim)
        self._norm_1 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self._attention(x)
        x = self._norm_0(x)
        x = x + self._mlp(x)
        x = self._norm_1(x)
        
        return x


class PlainGpt(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, num_heads: int, num_transformer_blocks) -> None:
        super().__init__()

        print("hidden dim: %d" % (hidden_dim,))
        print("num heads: %d" % (num_heads,))
        print("num blocks: %d" % (num_transformer_blocks,))

        self._word_embedding = nn.Embedding(vocab_size, hidden_dim)
        self._norm = nn.LayerNorm(hidden_dim)
        self._last_layer = nn.Linear(hidden_dim, vocab_size)
        blocks = [TransformerBlock(hidden_dim, num_heads) for _ in range(num_transformer_blocks)]
        self._transformer = nn.Sequential(*blocks)
        self._norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self._word_embedding(x)
        x = self._transformer(x)
        x = self._norm(x)
        x = self._last_layer(x)
        return x


def run():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data-path', type=str)
    arg_parser.add_argument('--model-output-path', type=str)
    arg_parser.add_argument('--batch-size', type=int, default=32)
    arg_parser.add_argument('--hidden-dim', type=int, default=64)
    arg_parser.add_argument('--num-heads', type=int, default=4)
    arg_parser.add_argument('--num-transformer-blocks', type=int, default=6)
    arg_parser.add_argument('--input-length', type=int, default=128)
    arg_parser.add_argument('--learning-rate', type=float, default=3e-4)
    arg_parser.add_argument('--device', type=str, default='cpu')
    args = arg_parser.parse_args()
    print(args)

    plain_sentences = PlainSentences.load(args.data_path)
    token_manager = TokenManager.create(plain_sentences)
    plain_gpt = PlainGpt(
        hidden_dim=args.hidden_dim, vocab_size=token_manager.get_vocab_size(),
        num_heads=args.num_heads, num_transformer_blocks=args.num_transformer_blocks,
    )
    run_base_training(
        input_length=args.input_length, plain_sentences=plain_sentences, token_manager=token_manager,
        batch_size=args.batch_size, model_output_path=args.model_output_path, device=args.device, plain_gpt=plain_gpt,
        learning_rate=args.learning_rate,
    )


if __name__ == '__main__':
    run()
