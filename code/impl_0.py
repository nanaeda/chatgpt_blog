from argparse import ArgumentParser
from dataclasses import dataclass
from numpy import random
from typing import List, Set

@dataclass(frozen=True)
class PlainSentences:
    """
    wiki-sentences.txtから文章を読み込んで保持するクラス
    """

    sentences: List[str]

    def get_random_sentence(self) -> str:
        return self.sentences[random.randint(len(self.sentences))]

    @staticmethod
    def load(path: str) -> 'PlainSentences':
        with open(path) as fp:
            return PlainSentences([line.strip() for line in fp])


class TokenManager:
    _TOKEN_BEGIN = '[begin]'

    def __init__(self, vocab: Set[str]) -> None:
        self._index_to_token = {i: w for i, w in enumerate(sorted(vocab))}
        self._token_to_index = {w: i for i, w in self._index_to_token.items()}

    def to_indexes(self, s: str) -> List[int]:
        assert TokenManager._TOKEN_BEGIN not in s  # [begin]が来るとバグる。あとでトークナイザーを再実装するので今はこれで。
        return [self._token_to_index[token] for token in s]
    
    def to_str(self, l: List[int]) -> str:
        return ''.join([self._index_to_token[index] for index in l])
    
    def get_vocab_size(self) -> int:
        return len(self._index_to_token)
    
    def get_begin_token_index(self) -> int:
        return self._token_to_index[self._TOKEN_BEGIN]
    
    @staticmethod
    def create(plain_sentences: PlainSentences) -> 'TokenManager':
        vocab = set()
        for s in plain_sentences.sentences:
            for letter in s:
              vocab.add(letter)
        vocab.add(TokenManager._TOKEN_BEGIN)
        return TokenManager(vocab)


def print_token_manager(token_manager: TokenManager):
    print("-------- TokenManager --------")
    print("vocab size: %d" % (token_manager.get_vocab_size(),))
    #　”こんにちは”を数字に変換してから文字列に再変換
    print(token_manager.to_str(
        [token_manager.get_begin_token_index()] +
        token_manager.to_indexes('こんにちは')
    ))
    print("-------- end --------\n")


def run():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data-path', type=str)
    args = arg_parser.parse_args()
    print(args)

    plain_sentences = PlainSentences.load(args.data_path)
    token_manager = TokenManager.create(plain_sentences)
    print_token_manager(token_manager)


if __name__ == '__main__':
    run()
