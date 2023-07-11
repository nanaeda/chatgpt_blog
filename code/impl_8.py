import torch

from .impl_1 import generate_sentence
from .impl_6 import BpeTokenManager, BytePairEncoder
from .impl_7 import Prompts
from argparse import ArgumentParser


def generate(
        prompts: Prompts, token_manager: BpeTokenManager, out_path: str, 
        sft, device: str, num_answers_per_query: int, max_length: int,
):
    with open(out_path, 'w') as fp:
        num_samples = 100
        for sample_index in range(num_samples):
            print("sampling (%3d/%3d)" % (sample_index, num_samples))
            
            prompt = prompts.get_random_prompt()
            fp.write("%s\n\n" % (prompt.query,))

            for _ in range(num_answers_per_query):
                while True:
                    # 例: 立方体の表面積の式を導きなさい。[sep]
                    initial_sentence = token_manager.to_indexes(BytePairEncoder.TOKEN_BEGIN + prompt.query + BytePairEncoder.TOKEN_PROMPT_SEP)

                    # 例: 立方体の表面積の式を導きなさい。[sep]立方体の表面積の式はS=6a^2であり、ここでaは立方体の一辺の長さです。
                    generated = generate_sentence(
                        model=sft, initial_sentence=initial_sentence,
                        max_length=int(max_length * 0.8), token_manager=token_manager, terminate_on_begin_token=True, device=device,
                    )

                    # Byte Pair Encodingで、デコード後にエンコードを行うと要素数が増えてしまう可能性があります。
                    # 以下の例では、"オブ" が初めにマージされてしまい要素数が2から7に増えてしまっています。
                    #
                    # 例
                    #   入力: ["マリオ", "ブラザーズ"]
                    #   デコード: ["マリオブラザーズ"]
                    #   再エンコード: ["マ", "リ", "オブ", "ラ", "ザ", "ー", "ズ"]
                    # 
                    # なので、デコード後にエンコードを再び行い、それが長さの上限を超えてないかを調べています。
                    if max_length < len(token_manager.to_indexes(token_manager.to_str(generated))):
                        print("too long")
                        continue

                    fp.write("%s\n" % (token_manager.to_str(generated[len(initial_sentence):]),))
                    break
            fp.write("\n")


def run():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--prompt-data-path', type=str)
    arg_parser.add_argument('--sft-model-path', type=str)
    arg_parser.add_argument('--sample-output-path', type=str)
    arg_parser.add_argument('--input-length', type=int, default=128)
    arg_parser.add_argument('--device', type=str, default='cpu')
    arg_parser.add_argument('--bpe-path', type=str)
    arg_parser.add_argument('--num-answers-per-query', type=int, default=5)
    args = arg_parser.parse_args()
    print(args)

    token_manager = BpeTokenManager.create(args.bpe_path)
    prompts = Prompts.load(args.prompt_data_path, max_instruction_length=args.input_length, token_manager=token_manager)
    token_manager.validate()
    sft = torch.load(args.sft_model_path)
    generate(
        prompts=prompts, token_manager=token_manager, device=args.device, max_length=args.input_length,
        out_path=args.sample_output_path, sft=sft, num_answers_per_query=args.num_answers_per_query,
    )


if __name__ == '__main__':
    run()
