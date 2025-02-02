# 记忆检索与触发
import torch

class SurpriseMetric:
    def __init__(self, decay_rate, influence_factor):
        self.eta_t = decay_rate
        self.theta_t = influence_factor
        self.memory = dict()
        self.prev_surprise = torch.tensor(0.0)

    def update(self, loss_gradient):
        # 惊喜度量更新
        self.prev_surprise = self.eta_t * self.prev_surprise - self.theta_t * loss_gradient
        return self.prev_surprise