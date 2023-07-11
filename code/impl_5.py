import numpy as np
import torch

from .impl_0 import PlainSentences, TokenManager
from .impl_1 import run_base_training
from .impl_3 import PlainGpt
from argparse import ArgumentParser
from torch import nn
from typing import List


class MultiLetterTokenizer:
    def __init__(self, token_unknown: str) -> None:
        self._vocab = set([''])
        self._all_prefixes = set([''])
        self._token_unknown = token_unknown
        self.add_token(self._token_unknown)

    def add_token(self, token: str) -> str:
        self._vocab.add(token)
        for i in range(len(token)):
            self._all_prefixes.add(token[: i + 1])

    def _find_longest_match(self, s: str, cursor: int) -> int:
        """
        最長のマッチングを探す。
        高速化のために、辞書登録されている単語の接頭語と一致していなかった時に打ち切る。
        """
        length = 0
        longest_match = length
        while cursor + length + 1 <= len(s):
            substr = s[cursor: cursor + length + 1]
            if substr not in self._all_prefixes:
                break
            if substr in self._vocab:
                longest_match = length + 1
            length += 1

        return longest_match
      
    def tokenize(self, s: str) -> List[str]:
        res = []
        cursor = 0
        while cursor < len(s):
            length = self._find_longest_match(s, cursor)
            if length == 0:  # 未知語
                res.append(self._token_unknown)
                cursor += 1
            else:
                res.append(s[cursor: cursor + length])
                cursor += length
        
        return res
 

class TokenManager:
    _TOKEN_START = '[start]'
    _TOKEN_PAD = '[pad]'
    _TOKEN_UNKNOWN = '[unk]'
    _TOKEN_PROMPT_SEP = '[sep]'

    def __init__(self, index_to_token, token_to_index, tokenizer) -> None:
        self._index_to_token = index_to_token
        self._token_to_index = token_to_index
        self._tokenizer = tokenizer

    def to_indexes(self, s: str) -> List[int]:
        return [self._token_to_index[c] for c in self._tokenizer.tokenize(s)]
    
    def to_str(self, l: List[int]) -> str:
        return ''.join([self._index_to_token[index] for index in l])
    
    def get_vocab_size(self) -> int:
        return len(self._index_to_token)
    
    def get_start_token_index(self) -> int:
        return self._token_to_index[self._TOKEN_START]
    
    def get_pad_token_index(self) -> int:
        return self._token_to_index[self._TOKEN_PAD]
    
    def get_prompt_sep_token_index(self) -> int:
        return self._token_to_index[self._TOKEN_PROMPT_SEP]
    
    @staticmethod
    def _create_single_letter_vocab(sentences: List[str]) -> Set[str]:
        res = set()
        for s in sentences:
            res.update(list(s))
        res.add(TokenManager._TOKEN_PAD)
        res.add(TokenManager._TOKEN_START)
        res.add(TokenManager._TOKEN_UNKNOWN)
        res.add(TokenManager._TOKEN_PROMPT_SEP)
        return res
    
    @staticmethod
    def _create_word_piece_vocab(sentences: List[str], target_vocab_size: int = 12288, min_occur: int = 50) -> Set[str]:
        print("create word piece vocab")
        init_occur = dict()
        for s in sentences:
           for c in s:
              init_occur[c] = init_occur.get(c, 0) + 1

        vocab = set()
        for letter, occur in init_occur.items():
           if min_occur <= occur:
              vocab.add(letter)
        vocab.add(TokenManager._TOKEN_PAD)
        vocab.add(TokenManager._TOKEN_START)
        vocab.add(TokenManager._TOKEN_UNKNOWN)
        vocab.add(TokenManager._TOKEN_PROMPT_SEP)

        tokenizer = MultiLetterTokenizer(TokenManager._TOKEN_UNKNOWN)
        for token in vocab:
            tokenizer.add_token(token)
        
        num_loops = 4
        for loop_i in range(num_loops):
          print("vocab: %d" % (len(vocab),))
          assert (len(vocab) <= target_vocab_size)

          bigram_separator = ':'  # データ中に「:」という文字が出てくると詰む。

          unigram_occur = dict()
          bigram_occur = dict()
          for s in sentences:
              tokens = tokenizer.tokenize(s)
              for token in tokens:
                  unigram_occur[token] = unigram_occur.get(token, 0) + 1
              for token0, token1 in zip(tokens[: -1], tokens[1:]):
                  if token0 == TokenManager._TOKEN_UNKNOWN or token1 == TokenManager._TOKEN_UNKNOWN:
                      continue
                  bigram = token0 + bigram_separator + token1
                  bigram_occur[bigram] = bigram_occur.get(bigram, 0) + 1
          
          word_piece_scorese = []
          for bigram, occur in bigram_occur.items():
              if min_occur <= occur:
                  token0, token1 = bigram.split(':')
                  score = bigram_occur[bigram] / (unigram_occur[token0] * unigram_occur[token1])
                  word_piece_scorese.append((score, bigram))
          sorted_scores = list(reversed(sorted(word_piece_scorese)))

          # 最もスコアの高いbigramを1つだけ辞書に加えて全てのスコアを再計算することも可能な気もするが、
          # それの高速な実装はかなりキツイので複数のbigramを一気に加える。
          num_to_add = (target_vocab_size - len(vocab)) // (num_loops - loop_i) 
          for _, bigram in sorted_scores[: num_to_add]:
              token0, token1 = bigram.split(':')
              pure_bigram = token0 + token1
              print("merge: %s + %s)" % (token0, token1))
              vocab.add(pure_bigram)
              tokenizer.add_token(pure_bigram)
        
        return vocab
              
    @staticmethod
    def create(plain_senteces: PlainSentences, vocab_cache_dir_path: str, use_word_piece: bool) -> 'TokenManager':
        vocab_cache_path = os.path.join(vocab_cache_dir_path, 'word_piece_vocab.json' if use_word_piece else 'vocab.json')
        if not os.path.exists(vocab_cache_path):
            if use_word_piece:
                vocab = TokenManager._create_word_piece_vocab(plain_senteces.sentences)
            else:
                vocab = TokenManager._create_single_letter_vocab(plain_senteces.sentences) 
            with open(vocab_cache_path, 'w') as fp:
                json.dump({'vocab': list(vocab)}, fp)
        with open(vocab_cache_path) as fp:
            vocab = set(json.load(fp)['vocab'])

        index_to_token = {i: w for i, w in enumerate(vocab)}
        token_to_index = {w: i for i, w in index_to_token.items()}

        tokenizer = MultiLetterTokenizer(TokenManager._TOKEN_UNKNOWN)
        for token in vocab:
            tokenizer.add_token(token)

        return TokenManager(
            index_to_token=index_to_token,
            token_to_index=token_to_index,
            tokenizer=tokenizer,
        )

    def validate(self):
        print("-------- token manager validation --------")
        print("vocab size: %d" % (self.get_vocab_size(),))
        print('tokens for %s: %s' % ('珈琲', self.to_indexes('珈琲')))  # WordPieceで纏められるはず。
        print('tokens for %s: %s' % ('杞憂', self.to_indexes('杞憂')))  # WordPieceで纏められるはず。
        print(self.to_str(
            [self.get_start_token_index()] +
            self.to_indexes('こんにちは:こんばんは。') + 
            [self.get_pad_token_index()]
        ))
        print("-------- end --------\n")


def run():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--base-data-path', type=str)
    arg_parser.add_argument('--model-output-path', type=str)
    args = arg_parser.parse_args()

    plain_sentences = PlainSentences.load(args.base_data_path)
    token_manager = TokenManager.create(plain_sentences)
    plain_gpt = PlainGpt(
        hidden_dim=32, vocab_size=token_manager.get_vocab_size(),
        num_heads=4, max_input_length=128, num_blocks=6,
    )
    run_base_training(
        input_length=128, plain_sentences=plain_sentences, token_manager=token_manager,
        batch_size=64, model_output_path=args.model_output_path, device='cpu', plain_gpt=plain_gpt,
    )


if __name__ == '__main__':
    run()












