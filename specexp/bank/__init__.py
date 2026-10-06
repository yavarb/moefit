"""CPU-side expert bank: holds the full quantized expert set; executes
selected experts from prediction triggers; returns aggregates only.
Boundary rule: expert weights never cross to the host (Metal) side.
"""
from .cpu_experts import CPUExpertBank
