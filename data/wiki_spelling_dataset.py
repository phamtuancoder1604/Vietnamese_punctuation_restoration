import json
import re
from typing import List, Dict

import torch
from tokenizers.normalizers import NFKC, Lowercase, Sequence
from torch.utils.data import Dataset, DataLoader

from tokenizer import (
    word_tokenizer,
    pre_char_tokenizer, char_tokenizer,
    num_max_char, num_max_word
)
from utils.common import tokenize_with_span, SpecialTokens, all_special_tokens


def check_file(fpath):
    """
    Check if there is some specific error that we cannot handle
    Such as vàThế -> và Thế
    """
    count = 0
    with open(fpath) as fp:
        for line_idx, line in enumerate(fp.readlines()):
            anno = json.loads(line)
            if anno["text"].find('\r') >= 0:
                print("Exist: \r")
                raise ValueError(f"Got \r at line: {line_idx}")

            for mistake in anno["mistakes"]:
                if len(re.findall(r"[\s]+", mistake["text"])) > 0 or len(
                        re.findall(r"\s+", mistake["suggest"][0])) > 0:
                    print(f"Line: {line_idx} {mistake}")
                    count += 1


def prepare_data(fpath):
    tokens: List[List[str]] = []  # Batch of sentences' tokens
    labels: List[List[int]] = []  # Batch of sentences' labels
    subs: List[Dict[int, str]] = []  # Batch with dictionary mapping id of misspelled token to correction

    normalizer = Sequence([NFKC(), Lowercase()])

    with open(fpath,encoding="utf-8") as fp:
        for line_idx, line in enumerate(fp.readlines()):
            anno = json.loads(line)
            text = anno["text"]
            text = normalizer.normalize_str(text)

            gen = tokenize_with_span(text=text)
            # Might miss cases like {'suggest': ['thua".\nKhi'], 'text': 'thua."\nKhi', 'start_offset': '9383'}

            cur_tokens, cur_labels, cur_subs = [], [], {}
            for pos, (start, end) in enumerate(gen):
                is_misspelled = False
                sub = ""
                token = text[start:end]

                for mistake in anno["mistakes"]:
                    if start == int(mistake["start_offset"]):
                        # Skip case error because of english
                        if "english" not in mistake["suggest"][0].lower():
                            sub = mistake["suggest"][0]
                            is_misspelled = True
                        else:
                            is_misspelled = False
                        break

                # -2 for [CLS] and [SEP] tokens
                if len(cur_tokens) == num_max_word - 2:
                    tokens.append(cur_tokens)
                    labels.append(cur_labels)
                    subs.append(cur_subs)
                    cur_tokens, cur_labels, cur_subs = [], [], {}
                else:
                    cur_tokens.append(token)
                    if is_misspelled:
                        cur_subs[len(cur_tokens) - 2] = sub  # Map token's id to its substitution
                        cur_labels.append(1)
                    else:
                        cur_labels.append(0)
    return tokens, labels, subs


def wiki_spelling_collator(batch):
    batch_origin_tokens, batch_detection_labels, batch_subs = zip(*batch)

    batch_detection_lbs = [[0] + item + [0] * (num_max_word - len(item) - 1) for item in batch_detection_labels]
    # Truncate to maximum number of words
    batch_detection_lbs = [item[:num_max_word] for item in batch_detection_lbs]
    batch_detection_lbs = torch.LongTensor(batch_detection_lbs)
    batch_detection_lbs[batch_detection_lbs != 0] = 1

    batch_origin_tokens = [' '.join(tks) for tks in batch_origin_tokens]
    # Pad to max position embedding
    batch_origin_enc = word_tokenizer(batch_origin_tokens, padding=True, truncation=True,
                                      max_length=num_max_word, return_tensors="pt")
    # Create batch of char ids = batch(sent 1) "stack on" batch(sent 2)
    batch_sent_words = []

    _, seq_word_len = batch_origin_enc["input_ids"].shape
    for sent_idx, tokens in enumerate(batch_origin_tokens):
        tokens = tokens.split()
        tokens = [SpecialTokens.cls] + tokens[:seq_word_len - 2] + [SpecialTokens.sep]
        tokens = tokens + [SpecialTokens.pad] * (seq_word_len - len(tokens))
        for word_idx, word in enumerate(tokens):
            if word in all_special_tokens:
                batch_sent_words.append(SpecialTokens.unk)
            else:
                batch_sent_words.append(word)

    batch_char_tok = [' '.join(pre_char_tokenizer.pre_tokenize(word)) for word in batch_sent_words]
    batch_char_enc = char_tokenizer(batch_char_tok, padding=True, truncation=True,
                                    max_length=num_max_char, return_tensors="pt")
    assert (batch_char_enc["input_ids"].size(0) / len(batch_origin_tokens) == batch_origin_enc["input_ids"].size(1)), \
        f'ERROR {batch_char_enc["input_ids"].size(0)} {len(batch_origin_tokens)} {batch_origin_enc["input_ids"].size(1)}'
    assert (batch_origin_enc["input_ids"].size() == batch_detection_lbs.size()), \
        f'[ERROR] {batch_origin_enc["input_ids"].size()} {batch_detection_lbs.size()}'

    # Mark label of PADDING position to be -100, so this will not contribute to detection loss
    padding_mask = batch_origin_enc["attention_mask"].type(torch.bool)
    batch_detection_lbs[~padding_mask] = -100

    return {
        "word_input_ids": batch_origin_enc["input_ids"],
        "word_attention_mask": batch_origin_enc["attention_mask"],
        "word_token_type_ids": batch_origin_enc["token_type_ids"],
        "char_input_ids": batch_char_enc["input_ids"],
        "char_attention_mask": batch_char_enc["attention_mask"],
        "char_token_type_ids": batch_char_enc["token_type_ids"],
        "correction_labels": batch_subs,
        "origin_sequences": [seq.split() for seq in batch_origin_tokens],
        "detection_labels": batch_detection_lbs,
    }


class WikiSpellingDataset(Dataset):
    def __init__(self, fpath: str):
        super().__init__()
        self.tokens: List[List[str]] = []  # Batch of sentences' tokens
        self.labels: List[List[int]] = []  # Batch of sentences' labels
        self.subs: List[Dict[int, str]] = []  # Batch with dictionary mapping id of misspelled token to correction
        self.tokens, self.labels, self.subs = prepare_data(fpath)

    def __getitem__(self, index):
        origin_tokens = self.tokens[index]
        detection_labels = self.labels[index]
        subs = self.subs[index]

        return origin_tokens, detection_labels, subs

    def __len__(self):
        return len(self.tokens)


if __name__ == '__main__':
    import pickle
    path = "spelling_test.json"
    ds = WikiSpellingDataset(path)
    pickle.dump(ds, open("runs/dummy.pkl", "wb"))

    ds = pickle.load(open("runs/dummy.pkl", "rb"))
    dl = DataLoader(ds, collate_fn=wiki_spelling_collator,
                    batch_size=1, drop_last=False, shuffle=False)

    for nm, inputs in enumerate(dl):
        if torch.sum(inputs["detection_labels"]) > 0:
            print("HERE")
        pass
