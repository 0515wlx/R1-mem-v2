"""GRPO 奖励计算模块"""

import torch

class RewardCalculator:
    def calculate_relative_rewards(self, rewards):
        """
        计算相对奖励
        :param rewards: 原始奖励
        :return: 相对奖励
        """
        mean_reward = torch.mean(rewards)
        std_reward = torch.std(rewards)
        return (rewards - mean_reward) / std_reward