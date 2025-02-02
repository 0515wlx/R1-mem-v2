"""初始化GRPO模块"""
from .policy_updater import PolicyUpdater
from .reward_calculator import RewardCalculator

class GRPO:
    def __init__(self, epsilon=0.2, beta=0.01):
        self.epsilon = epsilon
        self.beta = beta
        self.policy_updater = PolicyUpdater(epsilon, beta)
        self.reward_calculator = RewardCalculator()

    def optimize(self, policy, old_policy, outputs, rewards):
        relative_rewards = self.reward_calculator.calculate_relative_rewards(rewards)
        return self.policy_updater.update_policy(policy, old_policy, outputs, relative_rewards)