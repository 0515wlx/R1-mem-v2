"""门控机制模块"""

import torch
import torch.nn as nn

class MemoryGating(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

    def concat_memory(self, x, memory_bank):
        """
        拼接记忆和输入
        :param x: 输入形状 (batch_size, seq_len, hidden_size)
        :param memory_bank: 记忆形状 (batch_size, memory_size, hidden_size)
        :return: 拼接后的输入形状 (batch_size, seq_len + memory_size, hidden_size)
        """
        return torch.cat([x, memory_bank], dim=1)

    def gate_output(self, attended_output):
        """
        门控输出
        :param attended_output: 滑动窗口注意力输出形状 (batch_size, seq_len + memory_size, hidden_size)
        :return: 门控后的输出形状 (batch_size, seq_len, hidden_size)
        """
        seq_len = attended_output.size(1) // 2
        return attended_output[:, :seq_len, :] * self.gate(attended_output)