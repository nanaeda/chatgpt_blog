import numpy as np
import time
import torch

from .impl_6 import BpeTokenManager, BytePairEncoder
from .impl_7 import Prompts
from .impl_9 import RewardModel
from argparse import ArgumentParser
from numpy import random
from torch import nn


def train(
        prompts: Prompts, token_manager: BpeTokenManager, input_length: int, num_steps_per_epoch: int,
        out_path: str, sft, instruct_gpt, reward_model: RewardModel, critic_model: RewardModel, device: str, surrogate_eps: float, 
        learning_rate: float, kl_coeff: float,
):
    print("tain base gpt")

    time_begin = time.time()
    
    sft.to(device)
    instruct_gpt.to(device)
    reward_model.to(device)

    sft.eval()
    reward_model.eval()
    
    instruct_gpt_optimizer = torch.optim.Adam(instruct_gpt.parameters(), lr=learning_rate)
    critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=learning_rate)

    for epoch in range(1000000000):

        # まずはランダムに選ばれたクエリに対して、答えを生成するということを繰り返す。
        data_list = []
        for _ in range(num_steps_per_epoch):
            query = prompts.get_random_prompt().query

            # queryに対する答えをinstruct_gptを使って生成する。
            initial_input = token_manager.to_indexes(BytePairEncoder.TOKEN_BEGIN + query + BytePairEncoder.TOKEN_PROMPT_SEP)

            nn_input = list(initial_input)
            while len(nn_input) < input_length:
                out = nn.functional.softmax(instruct_gpt(torch.tensor([nn_input]).to(device)), dim=-1)[0, -1]
                out = out.to('cpu')  # GPUを使っている場合に備えてデータをCPUに持ってくる。
                nn_input.append(random.choice(token_manager.get_vocab_size(), p=out.detach().numpy()))

                if nn_input[-1] == token_manager.get_begin_token_index():
                    nn_input = nn_input[:-1]
                    break
            
            if len(nn_input) == len(initial_input):
                continue
                
            data_list.append((
                initial_input,
                nn_input,
                instruct_gpt(torch.tensor([nn_input]).to(device)).to('cpu').detach(),
            ))
        
        # 上で集めたサンプルを使って、モデルを更新していく。
        policy_losses = []
        kl_losses = []
        critic_losses = []
        for data in data_list:
            initial_input = data[0]
            initial_input_len = len(initial_input)
            nn_input = data[1]

            old_pred_logits_detached = data[2].to(device)
            new_pred_logits = instruct_gpt(torch.tensor([nn_input]).to(device))
            sft_pred_logits_detached = sft(torch.tensor([nn_input]).to(device)).detach()

            old_pred_detached = torch.softmax(old_pred_logits_detached, dim=-1)
            new_pred = torch.softmax(new_pred_logits, dim=-1)
            sft_pred_detached = torch.softmax(sft_pred_logits_detached, dim=-1).detach()

            def take_action(pred_):
                index = np.array(nn_input).reshape([1, len(nn_input), 1])
                return torch.gather(pred_, dim=2, index=torch.tensor(index, dtype=torch.int64).to(device))[0, initial_input_len:]

            old_action_log_pred_detached = take_action(old_pred_detached.log())
            new_action_log_pred = take_action(new_pred.log())

            ratios = (new_action_log_pred - old_action_log_pred_detached).exp()

            # Advantage Actor-Critic
            reward_detached = reward_model(torch.tensor([nn_input]).to(device))[0, -1, 0].detach()
            bias = critic_model(torch.tensor([initial_input]).to(device))[0, -1, 0]
            advantage_detached = (reward_detached - bias).detach()

            # clipped surrogate objectiveの計算。
            unclipped = ratios * advantage_detached
            clipped = ratios.clamp(1 - surrogate_eps, 1 + surrogate_eps) * advantage_detached

            # この数値の呼び方がよくわからないのでpolicy lossと呼んでいます。
            policy_loss = -torch.min(unclipped, clipped).mean()

            # KL-divergence regularization.
            kl_loss = 0
            for i in range(initial_input_len, len(nn_input)):
                l = (new_pred[:, i] * (torch.log(new_pred[:, i]) - torch.log(sft_pred_detached[:, i]))).mean()
                assert 0 <= l + 1e-6 # 定義より0以上であると保証されている。1e-6は計算精度の心配から。
                kl_loss += kl_coeff * l

            loss = policy_loss + kl_loss

            instruct_gpt_optimizer.zero_grad()
            loss.backward()
            instruct_gpt_optimizer.step()

            policy_losses.append(policy_loss.to('cpu').detach().numpy())
            kl_losses.append(kl_loss.to('cpu').detach().numpy())

            # Criticモデルの訓練
            critic_loss = (reward_detached - bias) ** 2

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            critic_losses.append(critic_loss.to('cpu').detach().numpy())


        print("epoch %4d (%4d sec): policy-loss(?)=%0.6f, kl-loss=%0.6f, critic-loss=%0.6f: %s" % (
            epoch, time.time() - time_begin, 
            np.mean(policy_losses[-50:]), 
            np.mean(kl_losses[-50:]), 
            np.mean(critic_losses[-50:]),
            token_manager.to_str(data_list[0][1]),
        ))
    
        torch.save(instruct_gpt, out_path)


def run():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--prompt-data-path', type=str)
    arg_parser.add_argument('--sft-model-path', type=str)
    arg_parser.add_argument('--reward-model-path', type=str)
    arg_parser.add_argument('--bpe-path', type=str)
    arg_parser.add_argument('--device', type=str, default='cpu')
    arg_parser.add_argument('--model-output-path', type=str)
    arg_parser.add_argument('--input-length', type=int, default=128)
    arg_parser.add_argument('--num-steps-per-epoch', type=int, default=16)
    arg_parser.add_argument('--learning-rate', type=float, default=3e-4)
    arg_parser.add_argument('--surrogate-eps', type=int, default=0.1)
    arg_parser.add_argument('--kl-coeff', type=int, default=1e-2)
    args = arg_parser.parse_args()
    print(args)

    token_manager = BpeTokenManager.create(args.bpe_path)
    prompts = Prompts.load(args.prompt_data_path, 128, token_manager=token_manager)
    token_manager.validate()
    sft = torch.load(args.sft_model_path)
    instruct_gpt = torch.load(args.sft_model_path)
    reward_model = torch.load(args.reward_model_path)
    critic_model = torch.load(args.reward_model_path)
    train(
        input_length=args.input_length, prompts=prompts, token_manager=token_manager, 
        num_steps_per_epoch=args.num_steps_per_epoch, device=args.device, critic_model=critic_model,
        out_path=args.model_output_path, sft=sft, instruct_gpt=instruct_gpt, reward_model=reward_model,
        surrogate_eps=args.surrogate_eps, learning_rate=args.learning_rate, kl_coeff=args.kl_coeff,
    )


if __name__ == '__main__':
    run()
