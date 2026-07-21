import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

SRC = "data/ag_news_train.csv"
OUT = "data/raw_dataset.csv"

N_SAMPLE = 6000

LABEL_MAP = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

FOREIGN_SNIPPETS = [
    "Ceci est un texte en français inséré dans le document.",
    "Dies ist ein deutscher Satz, der eingefügt wurde.",
    "این یک جمله فارسی است که وارد شده است.",
    "这是插入的中文句子。",
    "Este es un fragmento de texto en español.",
    "Это предложение на русском языке.",
]

MOJIBAKE_SNIPPETS = [
    "The companyÃ¢â‚¬â„¢s profits rose",
    "âœ“ Confirmed: â€œofficialâ€ sources say",
    "PriceÂ increaseÂ ofÂ 5%",
]

HTML_SNIPPETS = [
    "<p>Breaking: markets react to <b>Fed decision</b>.</p>",
    "<div class='article'>Full story <a href='#'>here</a></div>",
]


def load_source(n):
    df = pd.read_csv(SRC, header=None, names=["label", "title", "description"])
    df["label_name"] = df["label"].map(LABEL_MAP)
    df["text"] = (df["title"].str.strip() + ". " + df["description"].str.strip())
    df["text"] = df["text"].str.replace(r"\\", " ", regex=True)

    sample = df.sample(n=n, random_state=42).reset_index(drop=True)
    sample["doc_id"] = ["doc_%06d" % i for i in range(len(sample))]
    return sample[["doc_id", "label_name", "text"]]


def inject_edge_cases(df):
    df = df.copy()
    n = len(df)
    rng = np.random.RandomState(42)

    null_idx = rng.choice(n, size=int(n * 0.01), replace=False)
    df.loc[null_idx, "text"] = None

    remaining = list(set(range(n)) - set(null_idx))
    empty_idx = rng.choice(remaining, size=int(n * 0.005), replace=False)
    df.loc[empty_idx, "text"] = ""

    remaining = list(set(remaining) - set(empty_idx))
    ws_idx = rng.choice(remaining, size=int(n * 0.005), replace=False)
    df.loc[ws_idx, "text"] = "   \n\t  "

    remaining = list(set(remaining) - set(ws_idx))
    mixed_idx = rng.choice(remaining, size=int(n * 0.03), replace=False)
    for i in mixed_idx:
        snippet = random.choice(FOREIGN_SNIPPETS)
        df.loc[i, "text"] = f"{df.loc[i, 'text']} {snippet}"

    remaining = list(set(remaining) - set(mixed_idx))
    moji_idx = rng.choice(remaining, size=int(n * 0.02), replace=False)
    for i in moji_idx:
        snippet = random.choice(MOJIBAKE_SNIPPETS)
        df.loc[i, "text"] = f"{df.loc[i, 'text']} {snippet}"

    remaining = list(set(remaining) - set(moji_idx))
    html_idx = rng.choice(remaining, size=int(n * 0.015), replace=False)
    for i in html_idx:
        snippet = random.choice(HTML_SNIPPETS)
        df.loc[i, "text"] = f"{snippet} {df.loc[i, 'text']}"

    dup_sample = df.sample(n=int(n * 0.01), random_state=7)
    df = pd.concat([df, dup_sample], ignore_index=True)

    remaining2 = list(set(range(len(df))) - set(null_idx) - set(empty_idx) - set(ws_idx))
    excess_idx = rng.choice(remaining2, size=int(n * 0.02), replace=False)
    for i in excess_idx:
        if pd.notna(df.loc[i, "text"]):
            df.loc[i, "text"] = df.loc[i, "text"].replace(" ", "   ")

    return df.sample(frac=1, random_state=1).reset_index(drop=True)


if __name__ == "__main__":
    df = load_source(N_SAMPLE)
    df = inject_edge_cases(df)
    print(f"Final raw dataset size: {len(df)} rows")
    df.to_csv(OUT, index=False)
    print(f"Saved to {OUT}")