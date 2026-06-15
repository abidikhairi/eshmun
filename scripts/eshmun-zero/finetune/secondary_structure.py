"""
Finetune a linear secondary-structure prediction head on top of a frozen,
pretrained EshmunZero backbone.

Expects CSV datasets (e.g. exported from CASP12/CASP13 or CB513) with one
sequence per row and a per-residue label string of the same length, e.g.:

    sequence,label
    MKTAYIAKQRQ,CCHHHHHHHCC

Only the classification head (a single linear layer on top of the backbone's
final hidden states) is trained; all EshmunZero parameters stay frozen.
"""

import argparse
import logging
import os

import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from eshmun.models.zero.tokenization import EshmunZeroTokenizer
from eshmun.common.finetuning_commons import get_hidden_states, load_model


def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("eshmun.zero.finetune.secondary_structure")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    return logger


class SecondaryStructureDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer: EshmunZeroTokenizer,
        label_to_id: dict[str, int],
        sequence_column: str,
        label_column: str,
        max_length: int,
    ):
        df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
        self.sequences = df[sequence_column].astype(str).str.replace(" ", "").tolist()
        self.labels = df[label_column].astype(str).str.replace(" ", "").tolist()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        seq = self.sequences[index]
        label_str = self.labels[index][: len(seq)]

        encoding = self.tokenizer(seq, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]

        bos_offset = 1 if self.tokenizer.add_bos_token else 0
        num_residues = input_ids.shape[0] - bos_offset - (1 if self.tokenizer.add_eos_token else 0)

        labels = torch.full_like(input_ids, -100)
        for i in range(num_residues):
            labels[bos_offset + i] = self.label_to_id[label_str[i]]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_fn(batch, pad_token_id: int):
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class SecondaryStructureHead(nn.Module):
    def __init__(self, backbone, num_labels: int):
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.backbone.eval()

        self.classifier = nn.Linear(backbone.config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None = None):
        with torch.no_grad():
            hidden_states = get_hidden_states(self.backbone, input_ids, attention_mask)

        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        return logits, loss


@torch.no_grad()
def evaluate(model: SecondaryStructureHead, data_loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0

    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits, loss = model(**batch)

        total_loss += loss.item() * batch["labels"].numel()

        preds = logits.argmax(dim=-1)
        mask = batch["labels"] != -100
        total_correct += (preds[mask] == batch["labels"][mask]).sum().item()
        total_count += mask.sum().item()

    return total_loss / total_count, total_correct / total_count


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_path = os.path.join(args.output_dir, "finetune.log")
    logger = setup_logger(log_path)
    logger.info("device=%s  model=%s  labels=%s", device, args.model, args.labels)

    tokenizer = EshmunZeroTokenizer()
    label_to_id = {label: idx for idx, label in enumerate(args.labels.split(","))}

    backbone = load_model(args.model, device)
    model = SecondaryStructureHead(backbone, num_labels=len(label_to_id)).to(device)

    train_dataset = SecondaryStructureDataset(
        args.train_csv, tokenizer, label_to_id, args.sequence_column, args.label_column, backbone.config.max_seq_len
    )
    pad_token_id = int(tokenizer.pad_token_id)  # type: ignore[arg-type]
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_token_id)
    )

    eval_loader = None
    if args.eval_csv is not None:
        eval_dataset = SecondaryStructureDataset(
            args.eval_csv, tokenizer, label_to_id, args.sequence_column, args.label_column, backbone.config.max_seq_len
        )
        eval_loader = DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_token_id)
        )

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.learning_rate)

    best_metric = float("-inf") if eval_loader is not None else float("inf")
    best_head_path = os.path.join(args.output_dir, "classifier_head.pt")
    os.makedirs(args.output_dir, exist_ok=True)

    epoch_bar = tqdm(range(args.num_epochs), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        model.backbone.eval()

        epoch_loss = 0.0
        step_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="step", leave=False)
        for batch in step_bar:
            batch = {k: v.to(device) for k, v in batch.items()}

            _, loss = model(**batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        logger.info("epoch=%d  train_loss=%.6f", epoch + 1, avg_loss)

        if eval_loader is not None:
            eval_loss, eval_acc = evaluate(model, eval_loader, device)
            logger.info("epoch=%d  eval_loss=%.6f  eval_acc=%.4f", epoch + 1, eval_loss, eval_acc)
            epoch_bar.set_postfix(train_loss=f"{avg_loss:.4f}", eval_acc=f"{eval_acc:.4f}")

            if eval_acc > best_metric:
                best_metric = eval_acc
                torch.save(model.classifier.state_dict(), best_head_path)
                logger.info("new best checkpoint  eval_acc=%.4f  saved to %s", best_metric, best_head_path)
        else:
            epoch_bar.set_postfix(train_loss=f"{avg_loss:.4f}")

            if avg_loss < best_metric:
                best_metric = avg_loss
                torch.save(model.classifier.state_dict(), best_head_path)
                logger.info("new best checkpoint  train_loss=%.6f  saved to %s", best_metric, best_head_path)

    print(f"Best classifier head saved to {best_head_path} (metric={best_metric:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune a secondary-structure prediction head on a frozen EshmunZero backbone.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-csv", type=str, required=True, help="Path to the training CSV (e.g. CASP12, CASP13).")
    parser.add_argument("--eval-csv", type=str, default=None, help="Path to the evaluation CSV (e.g. CB513).")
    parser.add_argument("--model", type=str, required=True, help="Path to a pretrained EshmunZero model.")
    parser.add_argument("--sequence-column", type=str, default="sequence", help="CSV column containing sequences.")
    parser.add_argument("--label-column", type=str, default="label", help="CSV column containing per-residue secondary-structure labels.")
    parser.add_argument("--labels", type=str, default="H,E,C", help="Comma-separated label vocabulary (e.g. 'H,E,C' for DSSP3 or 'H,E,C,T,S,G,B,I' for DSSP8).")
    parser.add_argument("--batch-size", type=int, default=8, help="Training/evaluation batch size.")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for the classification head.")
    parser.add_argument("--output-dir", type=str, default="experiments/eshmun-zero/secondary-structure", help="Directory for logs and the best checkpoint.")

    main(parser.parse_args())
