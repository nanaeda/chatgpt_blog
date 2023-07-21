import numpy as np
import time
import torch

from .impl_0 import PlainSentences, TokenManager
from argparse import ArgumentParser
from numpy import random
from torch import nn
from typing import List


class PlainGpt(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int) -> None:
        super().__init__()

        print("hidden dim: %d" % (hidden_dim,))

        self._word_embedding = nn.Embedding(vocab_size, hidden_dim)
        self._norm = nn.LayerNorm(hidden_dim)
        self._last_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self._word_embedding(x)
        x = self._norm(x)
        x = self._last_layer(x)
        return x


def generate_sentence(
        model: PlainGpt, initial_sentence: List[int],
        max_length: int, token_manager: TokenManager,
        terminate_on_begin_token: bool, device: str,
) -> List[int]:
    """
    この関数では、
    1. ${initial_sentence}を初期の入力とする。
    2. 次の文字を予測する。
    3. 入力の最後に予測された文字を付け加え、新たな入力とする。
    4. 1に戻る。
    を繰り返すことにより、文章を生成します。

    3で付け加えた文字に関する部分のみを計算することで高速に文章を生成することもできるのですが、
    実装の簡略化のために毎回モデル全体を再計算しています。
    """
    model.eval()
    res = list(initial_sentence)
    while len(res) < max_length: 
        out = nn.functional.softmax(model(torch.tensor([res]).to(device)), dim=-1)[0, -1]
        out = out.to('cpu')  # GPUを使っている場合に備えてデータをCPUに持ってくる。
        res.append(random.choice(token_manager.get_vocab_size(), p=out.detach().numpy()))

        # beginトークンは文章を区切る時に使われる。複数文章生成する必要がない時はbeginトークンが来たら打ち切る。
        if terminate_on_begin_token and res[-1] == token_manager.get_begin_token_index():
            res = res[:-1]
            break
    model.train()

    return res


def run_base_training(
        input_length: int, plain_sentences: PlainSentences, plain_gpt: nn.Module, learning_rate: float,
        batch_size: int, model_output_path: str, token_manager: TokenManager, device: str,
):
    time_begin = time.time()
    
    plain_gpt.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(plain_gpt.parameters(), lr=learning_rate)

    losses = []
    for step in range(1000000000):
        # 1回トレーニングをする。
        nn_input = []
        for _ in range(batch_size):
            """
            入力を生成する。文章を1つだけセットするのではなく、[begin]トークンで区切って複数の文章を並べてセットする。
            例: [begin]吾輩は猫である。[strat]恥の多い生涯を送って来ました。[begin]トンネルを抜ける...
            """
            row = []
            while len(row) < input_length:
                row.append(token_manager.get_begin_token_index())
                row.extend(token_manager.to_indexes(plain_sentences.get_random_sentence()))
            nn_input.append(row[: input_length])
        nn_input = torch.tensor(np.array(nn_input)).to(device)

        nn_output = plain_gpt(nn_input)

        """
        {nn_output}には各ワードの予測確率のsoftmax適応前の値が入っている。
        そこで、{nn_output}から最も確率の高いワードを取り出すと、{nn_input}と{nn_output}は大体以下のようになっている。
        
        nn_input: [begin]吾輩は猫である。[strat]恥の多い生涯を
        nn_output: 吾輩は猫である。[strat]恥の多い生涯を送
        
        なので、モデルの予測が正確であれば、{nn_input}の最初1文字目を除いた文字列と、{nn_output}の最後1文字目を除いた文字列が一致する。
        """

        expected = nn_input[:, 1:]
        actual = nn_output[:, :-1]
        loss = criterion(
            actual.reshape(actual.shape[0] * actual.shape[1], actual.shape[2]),
            expected.reshape(expected.shape[0] * expected.shape[1]),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.to('cpu').detach().numpy())

        # ロスなどの情報を書き出し、モデルを保存する。
        # 実装例の複雑さを低くするため、本筋と関係のない実装を省く方針でいくので、交差検証などは省きます。
        if step % (10 if step < 200 else 100) == 0:
            generated_sentence = token_manager.to_str(generate_sentence(
                model=plain_gpt, initial_sentence=[token_manager.get_begin_token_index()], max_length=64, 
                token_manager=token_manager, terminate_on_begin_token=False, device=device,
            ))
            print("step %4d (%4d sec): loss=%0.6f, gen=%s" % (
                step, time.time() - time_begin, 
                np.mean(losses[-50:]), generated_sentence,
            ))

            torch.save(plain_gpt, model_output_path)


def run():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data-path', type=str)
    arg_parser.add_argument('--model-output-path', type=str)
    arg_parser.add_argument('--batch-size', type=int, default=32)
    arg_parser.add_argument('--hidden-dim', type=int, default=64)
    arg_parser.add_argument('--input-length', type=int, default=128)
    arg_parser.add_argument('--learning-rate', type=float, default=3e-4)
    arg_parser.add_argument('--device', type=str, default='cpu')
    args = arg_parser.parse_args()
    print(args)

    plain_sentences = PlainSentences.load(args.data_path)
    token_manager = TokenManager.create(plain_sentences)
    plain_gpt = PlainGpt(hidden_dim=args.hidden_dim, vocab_size=token_manager.get_vocab_size())
    run_base_training(
        input_length=args.input_length, plain_sentences=plain_sentences, token_manager=token_manager,
        batch_size=args.batch_size, model_output_path=args.model_output_path, device=args.device, plain_gpt=plain_gpt,
        learning_rate=args.learning_rate,
    )


if __name__ == '__main__':
    run()
