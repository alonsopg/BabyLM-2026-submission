from __future__ import annotations

from datasets import load_dataset


def load_text_dataset(name: str, split: str = "train", text_column: str = "text", validation_size: int = 2000, seed: int = 0):
    dataset = load_dataset(name, split=split)
    dataset = dataset.filter(lambda row: row.get(text_column) is not None and len(row[text_column].strip()) > 0)
    split_data = dataset.train_test_split(test_size=validation_size, seed=seed, shuffle=True)
    return split_data["train"], split_data["test"]


def tokenize_and_group(train_ds, val_ds, tokenizer, text_column: str, max_length: int, num_proc: int = 2):
    def tokenize(batch):
        return tokenizer(batch[text_column], truncation=False, add_special_tokens=False)

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names, num_proc=num_proc)
    val_tok = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names, num_proc=num_proc)

    def group(batch):
        concat = sum(batch["input_ids"], [])
        total = (len(concat) // max_length) * max_length
        input_ids = [concat[i: i + max_length] for i in range(0, total, max_length)]
        return {"input_ids": input_ids, "attention_mask": [[1] * max_length for _ in input_ids]}

    train_grouped = train_tok.map(group, batched=True, batch_size=1000, remove_columns=train_tok.column_names, num_proc=num_proc)
    val_grouped = val_tok.map(group, batched=True, batch_size=1000, remove_columns=val_tok.column_names, num_proc=num_proc)
    return train_grouped, val_grouped
