import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers.data import DataCollatorForLanguageModeling

from eshmun.models.zero import EshmunZero, EshmunZeroTokenizer, EshmunZeroConfig
from eshmun.tokenization import EshmunTokenizer
from eshmun.data.dataset import SwissProtDataset, SwissProtCollator



def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("eshmun.zero.train")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    return logger


def main(args):
    log_path = os.path.join(args.output_dir, "train.log")
    logger = setup_logger(log_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device=%s  attn_impl=%s  use_rope=%s", device, args.attn_impl, args.use_rope)

    tokenizer = EshmunZeroTokenizer()

    config = EshmunZeroConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attn_impl=args.attn_impl,
        use_rope=args.use_rope,
        top_k=args.top_k,
        window_size=args.window_size,
        num_kv_heads=args.num_kv_heads,
    )
    model = EshmunZero(config).to(device)  # type: ignore[arg-type]
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model parameters: %d (%.2fM)", n_params, n_params / 1e6)

    dataset = SwissProtDataset(file_path=args.data_path, tokenizer=tokenizer)
    # data_collator = SwissProtCollator(tokenizer=tokenizer)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.25, random_replace_prob=0.0)
    
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator
    )

    num_steps = len(data_loader)
    total_steps = num_steps * args.num_epochs
    num_training_steps = total_steps // args.gradient_accumulation_steps

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    best_loss = float("inf")
    best_checkpoint_dir: str | None = None

    epoch_bar = tqdm(range(args.num_epochs), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        epoch_loss = 0.0
        step_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}", unit="step", leave=False)
        for step, batch in enumerate(step_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            loss = output.loss / args.gradient_accumulation_steps
            loss.backward()

            raw_loss = loss.item() * args.gradient_accumulation_steps
            epoch_loss += raw_loss

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % args.logging_steps == 0:
                lr = scheduler.get_last_lr()[0]
                processed_tokens = int((batch['input_ids'] != tokenizer.pad_token_id).sum().item())

                step_bar.set_postfix(loss=f"{raw_loss:.4f}", lr=f"{lr:.2e}")
                
                logger.info(
                    f"epoch={epoch + 1:02d}  step={step:06d}/{total_steps:06d}  loss={raw_loss:.6f}  lr={lr:.6f}  tokens={processed_tokens:05d}",
                )

        avg_loss = epoch_loss / len(data_loader)
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}", best=f"{best_loss:.4f}")
        logger.info("epoch=%d complete  avg_loss=%.4f", epoch + 1, avg_loss)

        if avg_loss < best_loss:
            # Remove the previous best checkpoint before saving the new one
            if best_checkpoint_dir is not None:
                import shutil
                shutil.rmtree(best_checkpoint_dir, ignore_errors=True)

            best_loss = avg_loss
            best_checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}")
            model.save_pretrained(best_checkpoint_dir)
            logger.info("new best checkpoint  avg_loss=%.4f  saved to %s", best_loss, best_checkpoint_dir)
        else:
            logger.info("no improvement  avg_loss=%.4f  best=%.4f  checkpoint skipped", avg_loss, best_loss)

    model.save_pretrained(args.output_dir)
    logger.info("model saved to %s", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train EshmunGPT from scratch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--data_path", default="data/sequences.txt",
                        help="Path to the training sequences text file.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save the trained model and train.log.")

    # Model architecture
    parser.add_argument("--attn_impl", default="mha",
                        choices=["mha", "sparse", "sliding_window", "token_filter", "qwen", "gqa", "compressed", "gated"],
                        help="Attention implementation.")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--intermediate_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_kv_heads", type=int, default=4,
                        help="Number of key/value heads (used by GQA).")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=5,
                        help="Top-k value stored in config (used by sparse attention).")
    parser.add_argument("--window_size", type=int, default=10,
                        help="Sliding window size (used by sliding_window and gated attention).")
    parser.add_argument("--use_rope", action="store_true", default=True,
                        help="Use rotary position embeddings.")

    # Training
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=50)

    args = parser.parse_args()
    main(args)
