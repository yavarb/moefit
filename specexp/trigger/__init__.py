"""N-gram triggered expert prediction.

Key: hash of last n token IDs (same ngram_size as the models PLE block).
Value per (ngram-key, layer): candidate expert set. Two trigger flavors:
  - oracle-freq: empirical top experts conditioned on ngram key
  - output-cached: (deferred) cached aggregate outputs for repeated contexts
"""
from .ngram_table import NgramExpertTable
