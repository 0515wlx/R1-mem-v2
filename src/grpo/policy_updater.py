"""GRPO 策略更新模块"""

import torch
import torch.nn.functional as F

class PolicyUpdater:
    def __init__(self, epsilon, beta):
        self.epsilon = epsilon
        self.beta = beta

    def update_policy(self, policy, old_policy, outputs, relative_rewards):
        """
        更新策略
        :param policy: 当前策略
        :param old_policy: 旧策略
        :param outputs: 输出序列
        :param relative_rewards: 相对奖励
        :return: 更新后的损失
        """
        # 计算策略比率
        ratios = []
        for output in outputs:
            log_probs = policy.log_prob(output)
            old_log_probs = old_policy.log_prob(output)
            ratios.append(torch.exp(log_probs - old_log_probs))

        # 计算裁剪后的损失
        clipped_ratios = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
        clipped_loss = torch.min(ratios * relative_rewards, clipped_ratios * relative_rewards)

        # 计算 KL 散度惩罚
        kl_div = self.beta * F.kl_div(policy.log_prob(outputs), old_policy.log_prob(outputs))

        # 最终损失
        return -(clipped_loss.mean() - kl_div)