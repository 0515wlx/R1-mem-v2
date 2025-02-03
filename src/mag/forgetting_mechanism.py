"""遗忘机制模块"""

import torch

class ForgettingMechanism:
    def __init__(self, alpha=0.1, lru_threshold=0.5):
        self.alpha = alpha
        self.lru_threshold = lru_threshold

    def forget(self, memory):
        """
        更新记忆，应用遗忘机制
        :param memory: 当前记忆
        :return: 更新后的记忆
        """
        # 混合遗忘策略
        lru_mask = torch.rand_like(memory) > self.lru_threshold
        forgotten_memory = memory * (1 - self.alpha * lru_mask)
        return forgotten_memory