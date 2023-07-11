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
        terminate_on_start_token: bool, device: str,
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

        # startトークンは文章を区切る時に使われる。複数文章生成する必要がない時はstartトークンが来たら打ち切る。
        if terminate_on_start_token and res[-1] == token_manager.get_start_token_index():
            res = res[:-1]
            break
    model.train()

    return res


def run_base_training(
        input_length: int, plain_sentences: PlainSentences, plain_gpt: nn.Module,
        batch_size: int, model_output_path: str, token_manager: TokenManager, device: str='cpu',
):
    print("tain base gpt")

    print("input length: %d" % (input_length,))
    print("batch size: %d" % (batch_size,))

    time_begin = time.time()
    
    plain_gpt.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(plain_gpt.parameters(), lr=3e-4)

    losses = []
    for step in range(1000000000):
        # 1回トレーニングをする。
        nn_input = []
        for _ in range(batch_size):
            """
            入力を生成する。文章を1つだけセットするのではなく、[start]トークンで区切って複数の文章を並べてセットする。
            例: [start]吾輩は猫である。[strat]恥の多い生涯を送って来ました。[start]トンネルを抜ける...
            """
            row = []
            while len(row) < input_length:
                row.append(token_manager.get_start_token_index())
                row.extend(token_manager.to_indexes(plain_sentences.get_random_sentence()))
            nn_input.append(row[: input_length])
        nn_input = torch.tensor(np.array(nn_input)).to(device)

        nn_output = plain_gpt(nn_input)

        """
        この時点では{nn_input}と{nn_output}は大体以下のようになっている。
        
        nn_input: [start]吾輩は猫である。[strat]恥の多い生涯を
        nn_output: 吾輩は猫である。[strat]恥の多い生涯を送
        
        実際には{nn_output}は各ワードの予測確率のsoftmax適応前の値が入っているが、そこに目を瞑れば上記のようになっている。
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

        # 10回に1回、ロスなどの情報を書き出す。
        if step % 10 == 0:
            generated_sentence = token_manager.to_str(generate_sentence(
                model=plain_gpt, initial_sentence=[token_manager.get_start_token_index()], max_length=64, 
                token_manager=token_manager, terminate_on_start_token=False, device=device,
            ))
            print("step %4d (%4d sec): loss=%0.6f, gen=%s" % (
                step, time.time() - time_begin, 
                np.mean(losses[-50:]), generated_sentence,
            ))
      
        # 100回に1回、モデルを保存する。
        if step % 100 == 0:
            print("save models")
            torch.save(plain_gpt, model_output_path)


def run():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data-path', type=str)
    arg_parser.add_argument('--model-output-path', type=str)
    args = arg_parser.parse_args()

    plain_sentences = PlainSentences.load(args.data_path)
    token_manager = TokenManager.create(plain_sentences)
    plain_gpt = PlainGpt(hidden_dim=64, vocab_size=token_manager.get_vocab_size())
    run_base_training(
        input_length=128, plain_sentences=plain_sentences, token_manager=token_manager,
        batch_size=64, model_output_path=args.model_output_path, device='cpu', plain_gpt=plain_gpt,
    )


if __name__ == '__main__':
    # python -m code.impl_1 --data-path data/wiki-sentences.txt --model-output-path gen/base_model
    run()
