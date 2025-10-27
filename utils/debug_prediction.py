import logging

import torch
from transformers import PreTrainedTokenizerFast

from utils.common import SpecialTokens

# # configure logging at the root level of Lightning
# logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# configure logging on module level, redirect to file
logger = logging.getLogger("pytorch_lightning.debug")
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler("debug_prediction.log"))


def debug_prediction(detection_logits: torch.FloatTensor,
                     correction_logits: torch.FloatTensor,
                     correction_labels: torch.LongTensor,
                     char_input_ids: torch.LongTensor,
                     word_input_ids: torch.LongTensor,
                     detection_labels: torch.LongTensor,
                     word_tokenizer: PreTrainedTokenizerFast,
                     char_tokenizer: PreTrainedTokenizerFast
                     ):
    """
    Predicted sequence of first element in the Batch

    Args:
        detection_logits: logits output of detection head with shape B x word_seq_len x 2
        correction_logits: logits output of correction head with shape B x word_seq_len x num_vocab
        correction_labels: labels of shape B x Word Seq Len, where ignored tokens equal `-100`
        char_input_ids: shape (B x word_seq_len) x char_seq_len
        word_input_ids: shape B x word_seq_len
        detection_labels: detection ground truth of shape B x word_seq_len
        word_tokenizer: PreTrainedTokenizer to map ids to words
        char_tokenizer: PreTrainedTokenizer to map ids to chars
    """
    batch_size, word_seq_len, _ = detection_logits.shape

    det_preds = torch.argmax(detection_logits, dim=-1).detach().cpu().numpy()
    corr_preds = torch.argmax(correction_logits, dim=-1).detach().cpu().numpy()

    # Convert to human-readable
    raw_words = word_tokenizer.convert_ids_to_tokens(correction_labels[0])
    noise_words = word_tokenizer.convert_ids_to_tokens(word_input_ids[0])

    corr_pred_words = word_tokenizer.convert_ids_to_tokens(corr_preds[0])
    det_gt = detection_labels.detach().cpu().numpy()[0]
    det_preds = det_preds[0]
    noise_chars = char_input_ids.view(batch_size, word_seq_len, -1)[0]

    for word_idx, word in enumerate(noise_words):
        if word == SpecialTokens.unk:
            chars = noise_chars[word_idx].cpu().tolist()
            sep_idx = chars.index(3)  # [SEP] has id == 3
            chars = chars[1:sep_idx]  # Get chars in between [CLS] and [SEP]
            word_from_char = char_tokenizer.convert_ids_to_tokens(chars)
            word_from_char = ''.join(word_from_char)
            noise_words[word_idx] = word_from_char

    pad_idx = raw_words.index(SpecialTokens.sep) + 1
    raw_words = raw_words[:pad_idx]
    noise_words = noise_words[:pad_idx]
    det_gt = det_gt[:pad_idx]
    det_preds = det_preds[:pad_idx]
    corr_pred_words = corr_pred_words[:pad_idx]

    logger.info(f"Raw:       {raw_words}")
    logger.info(f"Noise:     {noise_words}")
    logger.info(f"Corr_Pred: {corr_pred_words}")
    logger.info(f"Det_GT:    {det_gt}")
    logger.info(f"Det_Pred:  {det_preds}")
