# 输入文本处理
import torch
from torch.nn import Module, Linear

class TextEncoder(Module):
    def __init__(self, vocab_size, hidden_size):
        super(TextEncoder, self).__init__()
        self.embedding = Linear(vocab_size, hidden_size)
        # 可以添加更多的层或结构

    def forward(self, input_text):
        # 实现输入文本的编码逻辑
        embedded = self.embedding(input_text)
        return embedded