import json
import torch

from .impl_1 import run_base_training
from .impl_6 import BpeTokenManager, BytePairEncoder
from argparse import ArgumentParser
from dataclasses import dataclass
from numpy import random
from typing import List


@dataclass(frozen=True)
class Prompt:
    query: str
    reply: str


@dataclass(frozen=True)
class Prompts:
    prompts: List[Prompt]

    def get_random_prompt(self) -> Prompt:
        return self.prompts[random.randint(len(self.prompts))]
      
    def get_random_sentence(self) -> str:
        prompt = self.get_random_prompt()
        # 例: 立方体の表面積の式を導きなさい。[sep]立方体の表面積の式はS=6a^2であり、ここでaは立方体の一辺の長さです。
        return prompt.query + BytePairEncoder.TOKEN_PROMPT_SEP + prompt.reply
    
    @staticmethod
    def load(path: str, max_instruction_length: int, token_manager: BpeTokenManager) -> 'Prompts':
        print("loading prompts")
        with open(path) as fp:
            data = json.load(fp)
        print("Initially, loaded %d prompts" % (len(data),))

        prompts = []
        for entity in data:
            if entity['input'] != '':
                continue
            
            def normalize_data(s):
                return s.replace('\n', '').replace('「', '').replace('」', '')

            prompt = Prompt(
                query=normalize_data(entity['instruction']),
                reply=normalize_data(entity['output']),
            )

            query_tokens = token_manager.to_indexes(prompt.query)
            reply_tokens = token_manager.to_indexes(prompt.reply)

            if max_instruction_length < len(query_tokens):
                continue
            
            filter_by_unknown = False
            unknown_id = token_manager.to_indexes(BytePairEncoder.TOKEN_UNKNOWN)[0]
            for tokens in [query_tokens, reply_tokens]:
                unknown_count = 0
                for token in tokens:
                    unknown_count += 1 if token == unknown_id else 0
                if len(tokens) * 0.1 < unknown_count:
                    filter_by_unknown = True
            if filter_by_unknown:
                continue
              
            prompts.append(prompt)

        print("Selected %d prompts." % (len(prompts),))
        return Prompts(prompts)


def run():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--prompt-data-path', type=str)
    arg_parser.add_argument('--learning-rate', type=float, default=3e-4)
    arg_parser.add_argument('--input-length', type=int, default=128)
    arg_parser.add_argument('--bpe-path', type=str)
    arg_parser.add_argument('--base-gpt-path', type=str)
    arg_parser.add_argument('--batch-size', type=int, default=32)
    arg_parser.add_argument('--device', type=str, default='cpu')
    arg_parser.add_argument('--model-output-path', type=str)
    args = arg_parser.parse_args()
    print(args)

    token_manager = BpeTokenManager.create(args.bpe_path)
    prompts = Prompts.load(args.prompt_data_path, max_instruction_length=args.input_length, token_manager=token_manager)
    token_manager.validate()
    plain_gpt = torch.load(args.base_gpt_path)
    run_base_training(
        input_length=args.input_length, plain_sentences=prompts, token_manager=token_manager, learning_rate=args.learning_rate,
        batch_size=args.batch_size, model_output_path=args.model_output_path, device=args.device, plain_gpt=plain_gpt,
    )


if __name__ == '__main__':
    run()
