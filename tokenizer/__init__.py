from transformers import PreTrainedTokenizerFast

from tokenizer.word_char_tokenizer import PreCharTokenizer, PreWordTokenizer
from utils.common import SpecialTokens

char_tokenizer = PreTrainedTokenizerFast(tokenizer_file="spell_model/char_tokenizer.json")
word_tokenizer = PreTrainedTokenizerFast(tokenizer_file="spell_model/word_tokenizer.json")
# Work around for missing pad_token
char_tokenizer.pad_token = SpecialTokens.pad
char_tokenizer.cls_token = SpecialTokens.cls
char_tokenizer.sep_token = SpecialTokens.sep

word_tokenizer.pad_token = SpecialTokens.pad
word_tokenizer.cls_token = SpecialTokens.cls
word_tokenizer.sep_token = SpecialTokens.sep

pre_char_tokenizer = PreCharTokenizer()
pre_word_tokenizer = PreWordTokenizer()

num_max_word = 192
num_max_char = 16

__all__ = [
    PreWordTokenizer,
    PreCharTokenizer,
    pre_char_tokenizer,
    pre_word_tokenizer,
    char_tokenizer,
    word_tokenizer,
    num_max_word,
    num_max_char
]
