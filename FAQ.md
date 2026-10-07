# FAQ for someone about to buy a 32 GB Mac for this

**Will it be as fast as my friend's 128 GB Mac?**
No, and the gap comes from the chip before it comes from the RAM. A base
M4 or M4 Pro has far less memory bandwidth than an M4 Max, and decode on
this model reads about 1.26 GB of expert weights per token from DRAM, so
the chip sets the ceiling: 23.6 tok/s simulated on an M4 Pro with full
LRU, 45.1 on an M4 Max (55 with a pinned hot-set). The 128 GB M4 Max
measured 57.4 tok/s. A 32 GB M4 Pro with 64 to 128 experts per layer
resident simulates at 14 to 26 tok/s\. Paging costs
you a little at the tight end, and the chip costs you the rest.

**Will this kill my SSD?**
Reads do not wear flash; only writes do, and this scheme writes nothing
to the SSD during decode. The expert files are read from SSD into RAM
and re-read as needed. What you should care about is read bandwidth and
heat: plain LRU streams about 177 to 560 MB per token depending on how
many experts per layer stay resident (128 down to 32), which at 15 tok/s
is several terabytes read per day. The pinned hot-set at 192 experts per
layer on a 48 GB Mac streams 138 MB per token. Choose the row of the
SETUP.md table with the lowest SSD traffic you can afford.

**Can I use this with LM Studio or Ollama?**
Partly. The prefetch tool warms file pages in the macOS unified-memory
cache, and any engine that memory-maps the checkpoint benefits from warm
pages, but the tool patches no engine. Today it runs in agent-integrated
mode, where your own code feeds it token ids on stdin. The routing sidecar needs an engine that exposes router decisions
or offline trace collection with the scripts in `experiments/`, which
require the model loaded under MLX on a 128 GB Mac.

**What about MoE models that are not Qwen?**
The estimator reads any HuggingFace-format safetensors directory and
classifies each tensor as routed expert, shared expert, embedding table,
or resident floor from its tensor name, and Mixtral- and DeepSeek-style
names are matched.
The simulation numbers are specific to this checkpoint's geometry (512
experts per layer, top-10, 2.69 MiB each, a 4.6 GiB floor). A model with
a larger floor or fewer, bigger experts behaves differently; run
`estimate.py` on it and read the floor and per-expert size it prints.

**Why not just quantize harder?**
The checkpoint is already 4-bit. Going to 3 or 2 bits on a 512-expert
MoE costs quality in ways that are hard to see in a quick test, and it
does not change the structure of the problem: a 64 GB Mac still cannot
hold the experts plus the n-gram table, and decode is still bound by
bytes per token through DRAM. Paging keeps the model byte-exact and
trades SSD traffic for RAM instead of trading quality for RAM.

**Is 24 GB enough?**
Only at the chip's ceiling of 11.8 tok/s, with 15 GiB usable after the
OS reserve and the 4.6 GiB floor taking a fifth of that. The OS will
sometimes push on the floor and the engine will stall. Treat 24 GB as
marginal and 32 GB as the comfortable minimum.


**Does Apple's memory compression or swap help?**
No. Expert weights are random-looking 4-bit data and do not compress.
Swap moves the same bytes through the same SSD with worse scheduling
than the paging policy. Never resize swap for this.
