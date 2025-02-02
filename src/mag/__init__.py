"""初始化MAG模块"""
from .gating_mechanism import MemoryGating
from .swin_attention import SlidingWindowAttention

class MAG:
    def __init__(self):
        self.gating = MemoryGating()
        self.attention = SlidingWindowAttention()

    def process_input(self, x, memory_bank):
        concatenated_input = self.gating.concat_memory(x, memory_bank)
        attended_output = self.attention.process(concatenated_input)
        return self.gating.gate_output(attended_output)