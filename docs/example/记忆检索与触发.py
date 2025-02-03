"""记忆检索与触发模块"""

import torch

class SurpriseMetric:
    def __init__(self, decay_rate, influence_factor):
        self.eta_t = decay_rate
        self.theta_t = influence_factor
        self.prev_surprise = torch.tensor(0.0)

    def should_trigger(self, loss_gradient, threshold=0.5):
        """
        根据梯度决定是否触发记忆检索
        :param loss_gradient: 损失梯度
        :param threshold: 触发阈值
        :return: 是否触发记忆检索
        """
        surprise = self.eta_t * self.prev_surprise - self.theta_t * loss_gradient
        self.prev_surprise = surprise
        return surprise > threshold