import argparse
import torch
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup

from eshmun.models.gpt import EshmunGPT, EshmunGPTConfig
from eshmun.tokenization import EshmunTokenizer
from eshmun.data.dataset import SwissProtDataset, SwissProtCollator


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = EshmunTokenizer.from_pretrained("data/eshmun-gpt/tokenizer")

    vocab_size = len(tokenizer)
    max_seq_len = 512
    pad_token_id = tokenizer.pad_token_id
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    hidden_size = 256
    intermediate_size = 512
    dropout = 0.1
    num_kv_heads = 4
    num_heads = 8
    num_layers = 8
    attn_impl = "mha"
    top_k = 5
    window_size = 10
    use_rope = True
    batch_size = 12

    config = EshmunGPTConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        dropout=dropout,
        max_seq_len=max_seq_len,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        attn_impl=attn_impl,
        use_rope=use_rope,
        top_k=top_k,
        window_size=window_size,
        num_kv_heads=num_kv_heads,
    )
    model = EshmunGPT(config).to(device)
    
    dataset = SwissProtDataset(file_path="data/sequences.txt", tokenizer=tokenizer)
    data_collator = SwissProtCollator(tokenizer=tokenizer)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator
    )

    learning_rate = 2e-4
    weight_decay = 0.01
    gradient_clip_value = 1.0
    gradient_accumulation_step = 8
    warmup_steps = 100
    num_steps = len(data_loader)
    num_epochs = 1
    total_steps = num_steps * num_epochs
    num_training_steps = total_steps // gradient_accumulation_step
    logging_steps = 50
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )

    for epoch in range(num_epochs):
        for step, batch in enumerate(data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            loss = output.loss / gradient_accumulation_step
            loss.backward()
            
            if (step + 1) % gradient_accumulation_step == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_value
                )
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % logging_steps == 0:
                processed_tokens = sum(batch['input_ids'] != tokenizer.pad_token_id)
                print(
                    f"""Step [{step:06d}/{total_steps:06d}]\tLoss: {loss.item() * gradient_accumulation_step:.4f}\t""" +
                    f"""Learing Rate: {scheduler.get_last_lr()[0]:.10f}\tProcessed Tokens: {processed_tokens:,d}"""
                )
                
    save_model_path = f'data/eshmun-gpt/models/gpt-{attn_impl}{"-rope" if use_rope else ""}'
    
    model.save_pretrained(save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main(args)
