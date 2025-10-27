import re

from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize.util import align_tokens


# Word and Char special tokens share the same names, but different in nature
class SpecialTokens:
    pad: str = "[PAD]"
    unk: str = "[UNK]"
    cls: str = "[CLS]"
    sep: str = "[SEP]"


all_special_tokens = [SpecialTokens.pad, SpecialTokens.unk, SpecialTokens.cls, SpecialTokens.sep]


def tokenize_with_span(text):
    raw_tokens = wordpunct_tokenize(text)

    # Convert converted quotes back to original double quotes
    # Do this only if original text contains double quote(s) or double
    # single-quotes (because '' might be transformed to `` if it is
    # treated as starting quotes).
    if ('"' in text) or ("''" in text):
        # Find double quotes and converted quotes
        matched = [m.group() for m in re.finditer(r"``|'{2}|\"", text)]

        # Replace converted quotes back to double quotes
        tokens = [
            matched.pop(0) if tok in ['"', "``", "''"] else tok
            for tok in raw_tokens
        ]
    else:
        tokens = raw_tokens

    yield from align_tokens(tokens, text)
