import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AlbertConfig

from sklearn.metrics import f1_score, precision_score, recall_score
from main import SpellChecker
from data.wiki_spelling_dataset import WikiSpellingDataset, wiki_spelling_collator
from models.metrics import compute_detection_metrics
from params import Param
from tokenizer import word_tokenizer


def evaluate(ckpt_path: str):
    params = Param()
    char_cfg = AlbertConfig().from_json_file("spell_model/char_model/config.json")
    word_cfg = AlbertConfig().from_json_file("spell_model/word_model/config.json")
    checker = SpellChecker.load_from_checkpoint(ckpt_path,
                                                # Kwargs
                                                char_config=char_cfg,
                                                word_config=word_cfg,
                                                params=params)
    # Test dataset
    # path = "./data/spelling_test.json"
    # test_ds = WikiSpellingDataset(path)

    test_ds = pickle.load(open("runs/dummy.pkl", "rb"))
    test_loader = DataLoader(test_ds,
                             collate_fn=wiki_spelling_collator,
                             batch_size=1,
                             drop_last=False,
                             shuffle=False,
                             num_workers=1)

    # trainer = pl.Trainer()
    # trainer.test(checker, dataloaders=test_loader)
    # with torch.no_grad():
    #     for idx, inputs in enumerate(test_loader):
    #         inputs.pop("correction_labels")
    #         detection_labels = inputs.pop("detection_labels")
    #         origin_sequences = inputs.pop("origin_sequences")
    #
    #         if torch.sum(detection_labels) == 0:
    #             continue
    #
    #         outputs = checker(inputs)
    #
    #         tokens = word_tokenizer.convert_ids_to_tokens(inputs["word_input_ids"][0])
    #         sequence = origin_sequences[0]
    #         det_pred = torch.argmax(outputs["detection_logits"], dim=-1)[0]
    #         det_gt = detection_labels[0]
    #
    #         pred_idx = torch.nonzero(det_pred, as_tuple=True)[0]
    #         print(f"Sequence:  {sequence}")
    #         print(f"Tokens:    {tokens}")
    #         print(f"Det_GT:    {det_gt}")
    #         print(f"Det_Pred:  {det_pred}")
    #         for i in pred_idx:
    #             print(tokens[i], end=', ')
    #         print()
    #         for i in pred_idx:
    #             print(sequence[i - 1], end=', ')
    #         print()
    #
    #         det_metrics, _ = compute_detection_metrics(detection_logits=outputs["detection_logits"],
    #                                                    detection_labels=detection_labels)
    #         print(det_metrics)
    #         pass

    gts = torch.LongTensor([])
    preds = torch.LongTensor([])
    with torch.no_grad():
        for idx, inputs in tqdm(enumerate(test_loader), total=len(test_ds)):
            inputs.pop("correction_labels")
            inputs.pop("origin_sequences")
            det_gts = inputs.pop("detection_labels")

            outputs = checker(inputs)
            det_preds = torch.argmax(outputs["detection_logits"], dim=-1)

            gts = torch.cat([gts, det_gts.reshape(-1)])
            preds = torch.cat([preds, det_preds.reshape(-1)])

        f1 = f1_score(y_true=gts, y_pred=preds, zero_division=0)
        precision = precision_score(y_true=gts, y_pred=preds, zero_division=0)
        recall = recall_score(y_true=gts, y_pred=preds, zero_division=0)

    print(f"F1:        {f1:.5f}")
    print(f"PRECISION: {precision:.5f}")
    print(f"RECALL:    {recall:.5f}")


if __name__ == '__main__':
    evaluate("runs/last.ckpt")
