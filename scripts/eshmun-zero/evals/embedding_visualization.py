import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm

from eshmun.models.zero.tokenization import EshmunZeroTokenizer

from eshmun.common.finetuning_commons import get_hidden_states, load_model, maskable_positions


def embed_sequence(model, tokenizer, seq: str, device) -> torch.Tensor:
    encoding = tokenizer(seq, truncation=True, max_length=model.config.max_seq_len, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        hidden_states = get_hidden_states(model, input_ids, attention_mask)[0]

    positions = maskable_positions(input_ids[0].cpu(), tokenizer)
    return hidden_states[positions].mean(dim=0)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = EshmunZeroTokenizer()
    model = load_model(args.model, device)

    df = pd.read_csv(args.csv)
    sequences = df[args.sequence_column].astype(str).tolist()
    labels = df[args.label_column].tolist()

    embeddings = []
    for seq in tqdm(sequences, desc="Extracting embeddings"):
        embeddings.append(embed_sequence(model, tokenizer, seq, device).cpu().numpy())

    embeddings = np.stack(embeddings)

    tsne = TSNE(n_components=2, perplexity=min(args.perplexity, len(sequences) - 1), random_state=args.seed)
    coords = tsne.fit_transform(embeddings)

    plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], args.label_column: labels})

    sns.set_theme(context="paper", style="ticks", font_scale=1.4)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue=args.label_column,
        palette="deep",
        s=60,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
        ax=ax,
    )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(args.title)
    sns.despine(ax=ax)
    ax.legend(title=args.label_column, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")

    pdf_output = os.path.splitext(args.output)[0] + ".pdf"
    fig.savefig(pdf_output, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract sequence embeddings, run t-SNE, and visualize colored by label.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file with 'sequence' and label columns.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained EshmunZero model.")
    parser.add_argument("--sequence-column", dest="sequence_column", type=str, default="sequence", help="CSV column containing sequences.")
    parser.add_argument("--label-column", dest="label_column", type=str, default="label", help="CSV column containing labels for color-coding.")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for t-SNE.")
    parser.add_argument("--title", type=str, default="t-SNE Projection of EshmunZero Embeddings", help="Plot title.")
    parser.add_argument("--output", type=str, default="embedding_tsne.png", help="Where to save the plot (PNG and PDF).")

    main(parser.parse_args())
