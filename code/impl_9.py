import numpy as np
import time
import torch

from .impl_6 import BpeTokenManager, BytePairEncoder
from argparse import ArgumentParser
from dataclasses import dataclass
from numpy import random
from torch import nn
from typing import List


@dataclass
class RankedAnswers:
    query: str
    ranked_answers: List[str]


class Data:
    def __init__(self, l: List[RankedAnswers]) -> None:
        self._l = l

    def get_random_ranked_answers(self) -> RankedAnswers:
        return self._l[random.randint(len(self._l))]

    @staticmethod
    def load(path: str, num_answers_per_query: int) -> 'Data':
        """
        読み込むファイルは以下のフォーマットであることを期待している。

        ------
        質問
        空行
        {num_answer_per_query}個の答え
        空行
        次の質問
        .
        .
        .
        """

        with open(path) as fp:
            lines = [line.strip() for line in fp]
        
        # 一応フォーマットをチェックしておく。
        assert (len(lines) % (num_answers_per_query + 3) == 0)
        for i in range(len(lines)):
            if i % (num_answers_per_query + 3) in [1, num_answers_per_query + 2]:
                assert (lines[i] == '')
        
        l = []
        for i in range(0, len(lines), (num_answers_per_query + 3)):
            l.append(RankedAnswers(
                query=lines[i],
                ranked_answers=lines[i + 2: i + 2 + num_answers_per_query],
            ))
        
        print("loaded %d data points" % (len(l),))
        
        return Data(l)


class RewardModel(nn.Module):
    """
    論文によると、元のGPTから最終embedding層を取り除いて、スカラーを出力するレイヤーに置き換えることでRewardモデルは作成される。
    ただ、実装の簡略化のため、ここでは最終層を取り除かずにそのままスカラーを出力するレイヤーを付け加えている。
    """

    def __init__(self, sft, vocab_size: int) -> None:
        super().__init__()
        self._sft = sft
        self._last_layer = nn.Linear(vocab_size, 1)
    
    def forward(self, x):
        x = self._sft(x)
        x = self._last_layer(x)
        return x


def train(
        data: Data, token_manager: BpeTokenManager, learning_rate: float,
        out_path: str, model: RewardModel, device: str,
):
    print("tain base gpt")

    time_begin = time.time()
    
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    for step in range(1000000000):
        ranked_answers = data.get_random_ranked_answers()

        # 各answerに対してスコアを計算する。
        scores = []
        for answer in ranked_answers.ranked_answers:
            nn_input = token_manager.to_indexes(
                ranked_answers.query + BytePairEncoder.TOKEN_PROMPT_SEP + answer
            )
            nn_input = torch.tensor(np.array([nn_input])).to(device)
            nn_output = model(nn_input)
            scores.append(nn_output[0, -1, 0])

        loss = 0
        for j in range(len(scores)):
            for i in range(j):
                # i番目のanswerの方が、j番目よりも人力採点では良いとされていて、高い点数を出してほしい。
                loss += -torch.log(torch.sigmoid(scores[i] - scores[j]))


        # すいません、公式の手法ではないのですが、点数が全体的に高くなりすぎたり低くなりすぎたりするので、真ん中に寄せるために0からの距離をロスに入れます。
        for score in scores:
            loss += 1e-2 * score * score

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.to('cpu').detach().numpy())

        # 10回に1回、ロスなどの情報を書き出す。
        if step % 10 == 0:
            print("step %4d (%4d sec): loss=%0.6f, scores=%s" % (
                step, time.time() - time_begin, 
                np.mean(losses[-50:]),
                np.array([score.to('cpu').detach().numpy() for score in scores])
            ))
      
            torch.save(model, out_path)


def run():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--sft-model-path', type=str)
    arg_parser.add_argument('--ranked-sample-path', type=str)
    arg_parser.add_argument('--learning-rate', type=float, default=1e-4)
    arg_parser.add_argument('--bpe-path', type=str)
    arg_parser.add_argument('--device', type=str, default='cpu')
    arg_parser.add_argument('--model-output-path', type=str)
    arg_parser.add_argument('--num-answers-per-query', type=int, default=5)
    args = arg_parser.parse_args()
    print(args)

    data = Data.load(args.ranked_sample_path, num_answers_per_query=args.num_answers_per_query)
    token_manager = BpeTokenManager.create(args.bpe_path)
    token_manager.validate()
    model = RewardModel(torch.load(args.sft_model_path), token_manager.get_vocab_size())
    train(
        data=data, token_manager=token_manager, device=args.device,
        out_path=args.model_output_path, model=model, learning_rate=args.learning_rate,
    )


if __name__ == '__main__':
    run()
