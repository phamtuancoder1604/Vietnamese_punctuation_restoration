import numpy as np
import torch

from sklearn.metrics import f1_score, precision_score, recall_score


def compute_detection_metrics(detection_logits: torch.FloatTensor,
                              detection_labels: torch.LongTensor):
    """
    Compute detection accuracy, precision, recall, f1
    Args:
        detection_logits: detection logits of shape Batch x Sequence Length x 2
        detection_labels: binary detection labels of shape Batch x Sequence Length
    Returns:
        metrics: dictionary of precision, recall, f1
        batch_size: size of the batch over which we average the metrics
    """
    det_preds = torch.argmax(detection_logits, dim=-1)  # Shape Batch x Sequence Length
    batch_size = det_preds.size(0)

    det_preds = det_preds.view(-1).cpu().numpy()  # Shape (Batch x Sequence Length) x 1
    det_labels = detection_labels.view(-1).cpu().numpy()  # Shape (Batch x Sequence Length) x 1

    # Only take Non-padding position into account
    # Accept other special character [CLS] [SEP] [UNK]
    valid_index = np.where(det_labels >= 0)[0]
    det_labels = det_labels[valid_index]
    det_preds = det_preds[valid_index]

    # zero_division=0 to suppress warning when training
    f1 = f1_score(y_true=det_labels, y_pred=det_preds, zero_division=0)
    precision = precision_score(y_true=det_labels, y_pred=det_preds, zero_division=0)
    recall = recall_score(y_true=det_labels, y_pred=det_preds, zero_division=0)

    return {"f1": f1, "precision": precision, "recall": recall}, batch_size


def compute_correction_metrics(correction_logits: torch.FloatTensor,
                               correction_labels: torch.LongTensor,
                               detection_labels: torch.LongTensor):
    """
    Compute correct accuracy, precision, recall, f1 on truly misspelled data
    Args:
        correction_logits: correction logits of shape Batch x Sequence Length x Num Vocab
        correction_labels: correction labels of shape Batch x Sequence Length
        detection_labels: binary detection labels of shape Batch x Sequence Length
    Returns:
        metrics: dictionary of precision, recall, f1
        batch_size: size of the batch over which we average the metrics
    """
    corr_preds = torch.argmax(correction_logits, dim=-1)  # Shape Batch x Sequence Length
    batch_size = corr_preds.size(0)

    det_labels = detection_labels.view(-1)
    corr_labels = correction_labels.view(-1)
    corr_preds = corr_preds.view(-1)

    # Only take error position into account
    valid_indexes = torch.where(det_labels > 0)[0]

    # Case no error, return 0
    if valid_indexes.size(0) == 0:
        return {"f1": 0, "precision": 0, "recall": 0}, batch_size

    corr_preds = torch.index_select(corr_preds, 0, valid_indexes).cpu().numpy()  # Shape (num valid) x 1
    corr_labels = torch.index_select(corr_labels, 0, valid_indexes).cpu().numpy()  # Shape (num valid) x 1

    # zero_division=0 to suppress warning when training
    f1 = f1_score(y_true=corr_labels, y_pred=corr_preds, average="micro", zero_division=0)
    precision = precision_score(y_true=corr_labels, y_pred=corr_preds, average="micro", zero_division=0)
    recall = recall_score(y_true=corr_labels, y_pred=corr_preds, average="micro", zero_division=0)

    return {"f1": f1, "precision": precision, "recall": recall}, batch_size


if __name__ == '__main__':
    torch.manual_seed(22)
    num_vocab = 120
    d_logits = torch.randn(2, 5, 2)  # Shape B x seq_len x 2
    d_labels = torch.randint(low=0, high=2, size=(2, 5))  # Shap B x seq_len

    print(d_logits)
    print(d_labels)

    c_logits = torch.randn(2, 5, num_vocab)  # Shape B x seq_len x num_vocabs
    c_labels = torch.randint(low=0, high=num_vocab, size=(2, 5))  # Shap B x seq_len

    d_metrics, _ = compute_detection_metrics(d_logits, d_labels)
    print(d_metrics)

    c_metrics, _ = compute_correction_metrics(c_logits, c_labels, d_labels)
    print(c_metrics)
