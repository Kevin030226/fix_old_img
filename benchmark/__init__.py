"""Benchmark suite (plan §3.10.3).

Three entry points, each runnable standalone:

    python -m benchmark.latency      p50/p95/p99 per task type
    python -m benchmark.throughput   queue throughput / worker saturation
    python -m benchmark.memory       peak RSS (+ GPU) per stage

All three default to a **synthetic stage** so they measure the platform
(scheduling, persistence, artifact I/O, queueing) on any machine, with no
weights and no GPU. Pass ``--real`` to run the actual model chain instead; that
path requires the weights to be present and is marked accordingly.
"""
