"""滑动窗口注意力模块"""

import torch
import torch.nn as nn

class SlidingWindowAttention(nn.Module):
    def __init__(self, window_size=5, hidden_size=512):
        super().__init__()
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)

    def process(self, x):
        """
        处理输入
        :param x: 输入形状 (batch_size, seq_len, hidden_size)
        :return: 注意力输出形状 (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_size)

        # 窗口注意力
        outputs = []
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            window = x[start:end]
            output, _ = self.attention(x[i:i+1], window, window)
            outputs.append(output)

        return torch.cat(outputs, dim=0).transpose(0, 1)