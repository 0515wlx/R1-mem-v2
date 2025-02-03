"""输入文本处理模块"""

import torch
from torch import nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, input_text):
        """
        对输入文本进行编码
        :param input_text: 输入文本张量，形状为 (batch_size, seq_len)
        :return: 编码后的张量，形状为 (batch_size, seq_len, hidden_size)
        """
        embedded = self.embedding(input_text)
        return self.encoder(embedded)