import numpy as np
import torch

from .impl_0 import PlainSentences, TokenManager
from .impl_1 import run_base_training
from .impl_2 import TransformerBlock
from argparse import ArgumentParser
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, hidden_dim: int, max_seq_length: int):
        super().__init__()
        assert (hidden_dim % 2 == 0)

        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-np.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_seq_length, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1]]
    

class PositionalEmbedding(nn.Module):

    def __init__(self, hidden_dim: int, max_seq_length: int) -> None:
        super().__init__()
        self._embedding = torch.nn.parameter.Parameter(torch.zeros((max_seq_length, hidden_dim)), requires_grad=True)

    def forward(self, x):
        return x + self._embedding[: x.shape[1]]
        

class PlainGpt(nn.Module):
    def __init__(
            self, hidden_dim: int, vocab_size: int, num_heads: int, num_transformer_blocks: int,
            max_input_length: int, use_positional_embedding: bool,
    ) -> None:
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

        self._positional_stuff = (
            PositionalEmbedding(hidden_dim, max_input_length)
            if use_positional_embedding else
            PositionalEncoding(hidden_dim, max_input_length)
        )

    def forward(self, x):
        x = self._word_embedding(x)
        x = self._positional_stuff(x)
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
    arg_parser.add_argument('--use-positional-embedding', action='store_true', default=False)
    args = arg_parser.parse_args()
    print(args)

    plain_sentences = PlainSentences.load(args.data_path)
    token_manager = TokenManager.create(plain_sentences)
    plain_gpt = PlainGpt(
        hidden_dim=args.hidden_dim, vocab_size=token_manager.get_vocab_size(),
        num_heads=args.num_heads, num_transformer_blocks=args.num_transformer_blocks,
        use_positional_embedding=args.use_positional_embedding, max_input_length=args.input_length,
    )
    run_base_training(
        input_length=args.input_length, plain_sentences=plain_sentences, token_manager=token_manager,
        batch_size=args.batch_size, model_output_path=args.model_output_path, device=args.device, plain_gpt=plain_gpt,
        learning_rate=args.learning_rate,
    )


if __name__ == '__main__':
    run()
