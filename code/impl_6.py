import MeCab

from .impl_0 import PlainSentences
from .impl_1 import run_base_training
from .impl_3 import PlainGpt
from argparse import ArgumentParser
from typing import List, Set, Tuple


class BytePairEncoder:
    TOKEN_BEGIN = '[begin]' # 文の初めに入れるトークン。
    TOKEN_UNKNOWN = '[unk]' # 知らない文字用のトークン。後に違うデータセットを使うので必要。
    TOKEN_PROMPT_SEP = '[sep]' # 質問と答えを分けるために使うトークン。後で使う。
    _mecab = MeCab.Tagger("-Owakati")

    def __init__(self, letters: List[str], merge_rules: List[Tuple[str, str]]) -> None:
        self._letters = set(letters)
        self._merge_rules = {rule: rank for rank, rule in enumerate(merge_rules)}

    def tokenize(self, s: str) -> List[str]:
        # 特殊トークンを見つけたら最優先で処理する。
        for token in [
            BytePairEncoder.TOKEN_BEGIN,
            BytePairEncoder.TOKEN_UNKNOWN,
            BytePairEncoder.TOKEN_PROMPT_SEP,
        ]:
            if token not in s:
                continue

            for i in range(len(s) - len(token) + 1):
                if s[i: i + len(token)] == token:
                    return self.tokenize(s[: i]) + [token] + self.tokenize(s[i + len(token):])

        # 特殊トークンはもうないので、MeCabでパースした後にマージ規則を適用する。
        res = []
        for word in self._mecab.parse(s).strip().split(' '):
            res.extend(self._tokenize_no_special_token(word))
        return res

    def _tokenize_no_special_token(self, s: str) -> List[str]:
        # 知らない文字をunknownトークンに置き換える。
        res = [(letter if letter in self._letters else BytePairEncoder.TOKEN_UNKNOWN) for letter in s]

        # 本来はマージ規則を１から全て適応していきたいが、速度的に厳しいので各ペアがマージできるかを確かめて、一番若い規則を適応する。
        while True:
            best_rank = len(self._merge_rules)
            best_index = -1

            for i in range(len(res) - 1):
                key = (res[i], res[i + 1])
                if key not in self._merge_rules:
                    continue

                rank = self._merge_rules[key]
                if best_rank < rank:
                    continue

                best_rank = rank
                best_index = i
            
            if best_index == -1:
                break

            merged = res[best_index] + res[best_index + 1]
            res = res[: best_index] + [merged] + res[best_index + 2:]
        
        return res
 

class BpeTokenManager:

    def __init__(self, index_to_token, token_to_index, encoder: BytePairEncoder) -> None:
        self._index_to_token = index_to_token
        self._token_to_index = token_to_index
        self._encoder = encoder

    def to_indexes(self, s: str) -> List[int]:
        return [self._token_to_index[c] for c in self._encoder.tokenize(s)]
    
    def to_str(self, l: List[int]) -> str:
        return ''.join([self._index_to_token[index] for index in l])
    
    def get_vocab_size(self) -> int:
        return len(self._index_to_token)
    
    def get_begin_token_index(self) -> int:
        return self._token_to_index[BytePairEncoder.TOKEN_BEGIN]
    
    def get_prompt_sep_token_index(self) -> int:
        return self._token_to_index[BytePairEncoder.TOKEN_PROMPT_SEP]
    
    @staticmethod
    def create(merge_rule_path: str) -> 'BpeTokenManager':
        letters = []
        merge_rules = []
        with open(merge_rule_path) as fp:
            for line in fp:
                tokens = line.strip().split(' ')
                if len(tokens) == 1:
                    letters.append(tokens[0])
                else:
                    assert len(tokens) == 2
                    merge_rules.append((tokens[0], tokens[1]))

        index = 0 
        index_to_token = dict()
        token_to_index = dict()
        
        for letter in letters:
            index_to_token[index] = letter
            token_to_index[letter] = index
            index += 1

        for rule in merge_rules:
            token = ''.join(rule)
            index_to_token[index] = token
            token_to_index[token] = index
            index += 1

        for token in [
            BytePairEncoder.TOKEN_BEGIN,
            BytePairEncoder.TOKEN_PROMPT_SEP,
            BytePairEncoder.TOKEN_UNKNOWN,
        ]:
            index_to_token[index] = token
            token_to_index[token] = index
            index += 1
            
        encoder = BytePairEncoder(letters, merge_rules)

        return BpeTokenManager(
            index_to_token=index_to_token,
            token_to_index=token_to_index,
            encoder=encoder,
        )

    def validate(self):
        print("-------- token manager validation --------")
        print("vocab size: %d" % (self.get_vocab_size(),))
        print('tokens for %s: %s' % ('ダイヤモンド', self.to_indexes('ダイヤモンド'))) # 1トークンにまとまる
        print('tokens for %s: %s' % ('北海道', self.to_indexes('北海道'))) # 1トークンにまとまる。
        print('tokens for %s: %s' % ('一石二鳥', self.to_indexes('一石二鳥'))) # 4文字のまま。
        # [begin]こんにちは[unk]こんばんは。[pad][sep]
        print(self.to_str(
            [self.get_begin_token_index()] +
            self.to_indexes('こんにちは:こんばんは。') + 
            [self.get_prompt_sep_token_index()]
        ))
        print("-------- end --------\n")


def run():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data-path', type=str)
    arg_parser.add_argument('--model-output-path', type=str)
    arg_parser.add_argument('--batch-size', type=int, default=128)
    arg_parser.add_argument('--hidden-dim', type=int, default=384)
    arg_parser.add_argument('--num-heads', type=int, default=8)
    arg_parser.add_argument('--num-transformer-blocks', type=int, default=12)
    arg_parser.add_argument('--input-length', type=int, default=128)
    arg_parser.add_argument('--learning-rate', type=float, default=3e-4)
    arg_parser.add_argument('--device', type=str, default='cpu')
    arg_parser.add_argument('--use-positional-embedding', action='store_true', default=False)
    arg_parser.add_argument('--bpe-path', type=str)
    args = arg_parser.parse_args()
    print(args)

    plain_sentences = PlainSentences.load(args.data_path)
    token_manager = BpeTokenManager.create(args.bpe_path)
    token_manager.validate()
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
