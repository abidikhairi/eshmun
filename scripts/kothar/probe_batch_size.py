"""
Empirically find the largest --per-device-batch-size that fits in memory for
pretrain.py's actual model/block-size/optimizer, on whatever GPU this runs
on. Does a real forward+backward+AdamW step at each candidate batch size
(not just forward) since the optimizer moment buffers are part of the real
memory budget.

Usage:
    python3 scripts/kothar/probe_batch_size.py
    python3 scripts/kothar/probe_batch_size.py --block-size 1024 --candidates 1,2,4,8,16,32,64
"""

import argparse

import torch

from eshmun.models.eshmun import EshmunForCausalLM

STUDENT_ID = "khairi/Kothar-student-seed-409M"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--candidates", default="1,2,4,8,16,32,64,96,128")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = [int(c) for c in args.candidates.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = EshmunForCausalLM.from_pretrained(STUDENT_ID, dtype=torch.float32).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    vocab_size = model.config.vocab_size

    largest_ok = None
    for batch_size in candidates:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            input_ids = torch.randint(0, vocab_size, (batch_size, args.block_size), device=device)
            out = model(input_ids=input_ids, labels=input_ids)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda":
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                free_gb, total_gb = (x / 1e9 for x in torch.cuda.mem_get_info())
                print(f"batch_size={batch_size:>4d}  peak_allocated={peak_gb:6.2f}GB  free={free_gb:6.2f}GB / total={total_gb:6.2f}GB  OK")
            else:
                print(f"batch_size={batch_size:>4d}  OK (cpu, no memory stat)")
            largest_ok = batch_size
        except torch.OutOfMemoryError:
            print(f"batch_size={batch_size:>4d}  OOM")
            break

    print(f"\nlargest batch size that fit: {largest_ok}")


if __name__ == "__main__":
    main()
