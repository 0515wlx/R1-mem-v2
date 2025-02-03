"""记忆检索模块"""

import torch
from torch import nn

class MemoryRetrieval(nn.Module):
    def __init__(self, memory_size, query_size):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_size))
        self.query_projection = nn.Linear(query_size, memory_size)
        # 冻结投影层参数
        for param in self.query_projection.parameters():
            param.requires_grad = False

    def forward(self, query_input):
        """
        检索记忆
        :param query_input: 查询输入
        :return: 检索到的记忆
        """
        query_projected = self.query_projection(query_input)
        return torch.mm(query_projected, self.memory.T)