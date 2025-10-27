from typing import Optional

from dataclasses import dataclass
from packaging import version

import torch
from torch import nn
from transformers import AlbertConfig, AlbertModel
from transformers.activations import ACT2FN
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPooling

from models.losses import compute_detection_loss, compute_correct_loss


class AlbertWordEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.word_embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.forward
    def forward(
            self,
            input_ids: torch.LongTensor,  # Shape B x Seq Len
            chars_embeds: torch.Tensor,  # Shape B x Seq Len x Emb Dim
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values_length=0
    ):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids,
        # solves issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        words_embeds = self.word_embeddings(input_ids)  # Shape B x Seq Len x Word Emb Dim
        inputs_embeds = torch.cat([words_embeds, chars_embeds], dim=-1)  # Shape B x Seq Len x (Word + Char) Emb Dim
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class AlbertWordEncoder(AlbertModel):
    """
    Override the base AlberModel since it does not allow passing both input_ids and inputs_embeds
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.embeddings = AlbertWordEmbeddings(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            chars_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and chars_embeds is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("You have to specify both input_ids and inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else chars_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, chars_embeds=chars_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0])) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@dataclass
class SpellCheckerOutput(ModelOutput):
    """
    Base class for spell checker.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*) :
             Summed loss.
        detection_loss: (`torch.FloatTensor` of shape `(1,)`, *optional*) :
            Detection loss.
        correction_loss: (`torch.FloatTensor` of shape `(1,)`, *optional*) :
            Correction loss.
        detection_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 2)`):
            Detection scores (before SoftMax).
        correction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_length)`):
            Correction scores (before SoftMax).
    """

    loss: Optional[torch.FloatTensor] = None
    detection_loss: Optional[torch.FloatTensor] = None
    correction_loss: Optional[torch.FloatTensor] = None
    detection_logits: Optional[torch.FloatTensor] = None
    correction_logits: Optional[torch.FloatTensor] = None


class DetectionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.word_embedding_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.word_embedding_size)
        self.decoder = nn.Linear(config.word_embedding_size, 2, bias=True)
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        prediction_scores = hidden_states
        return prediction_scores


class CorrectionHead(nn.Module):
    def __init__(self,
                 config,
                 word_embeddings: Optional[nn.Embedding] = None):
        """
        Detection and Classification Head
        Args:
            config: word level configuration file
            word_embeddings: the shared word embedding layer
        """
        super().__init__()

        self.layer_norm = nn.LayerNorm(config.word_embedding_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.word_embedding_size)
        self.decoder = nn.Linear(config.word_embedding_size, config.vocab_size, bias=True)

        if word_embeddings:
            # https://github.com/pytorch/examples/blob/main/word_language_model/model.py#L31
            self.decoder.weight = word_embeddings.weight

        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)

        prediction_scores = hidden_states
        return prediction_scores


class AlbertSpellChecker(nn.Module):
    def __init__(self, char_config, word_config):
        super().__init__()
        self.num_classes = word_config.vocab_size

        self.char_encoder = AlbertModel(char_config, add_pooling_layer=True)
        self.word_encoder = AlbertWordEncoder(word_config, add_pooling_layer=False)
        self.detection_head = DetectionHead(config=word_config)

        self.correction_head = CorrectionHead(config=word_config,
                                              word_embeddings=self.word_encoder.embeddings.word_embeddings)

    def forward(self,
                word_input_ids: Optional[torch.LongTensor] = None,  # Shape B x Seq Len
                word_attention_mask: Optional[torch.LongTensor] = None,  # Shape B x Seq Len
                word_token_type_ids: Optional[torch.LongTensor] = None,  # Shape B x Seq Len

                char_input_ids: Optional[torch.LongTensor] = None,  # Shape (B x Seq Len) x Num Char
                char_attention_mask: Optional[torch.LongTensor] = None,  # Shape (B x Seq Len) x Num Char
                char_token_type_ids: Optional[torch.LongTensor] = None,  # Shape (B x Seq Len) x Num Char
                correction_labels: Optional[torch.LongTensor] = None,  # Shape (B x Seq Len)
                detection_labels: Optional[torch.LongTensor] = None,  # Shape (B x Seq Len)
                ):
        char_outputs = self.char_encoder(
            input_ids=char_input_ids,
            attention_mask=char_attention_mask,
            token_type_ids=char_token_type_ids
        )
        char_embeddings = char_outputs["pooler_output"]  # Shape (B x Seq Len) x Char Emb Dim
        num_words, char_dim = char_embeddings.size()
        batch_size = word_input_ids.size(0)
        assert num_words // batch_size == num_words / batch_size, f"Not integer: {num_words} / {batch_size}"

        sequence_length = num_words // batch_size
        # Reshape into B x Seq Len x Char Emb Dim
        char_embeddings = char_embeddings.reshape((batch_size, sequence_length, char_dim))

        word_outputs = self.word_encoder.forward(
            input_ids=word_input_ids,
            attention_mask=word_attention_mask,
            token_type_ids=word_token_type_ids,
            chars_embeds=char_embeddings
        )

        detection_logits = self.detection_head(word_outputs[0])
        correction_logits = self.correction_head(word_outputs[0])

        detection_loss = 0.
        correction_loss = 0.

        if detection_labels is not None:
            detection_loss = compute_detection_loss(detection_logits=detection_logits,
                                                    detection_labels=detection_labels)
            if correction_labels is not None:
                correction_loss = compute_correct_loss(correction_logits=correction_logits,
                                                       correction_labels=correction_labels,
                                                       detection_labels=detection_labels)

        loss = correction_loss + detection_loss

        return SpellCheckerOutput(loss, detection_loss, correction_loss, detection_logits, correction_logits)


if __name__ == '__main__':
    char_cfg = AlbertConfig()
    char_cfg.hidden_size = 256
    char_cfg.num_hidden_layers = 4
    char_cfg.num_attention_heads = 8  # hidden_size % num_attention_heads == 0
    char_cfg.max_position_embeddings = 16
    char_cfg.intermediate_size = 768
    char_cfg.vocab_size = 227  # Tobe update == real vocab size
    char_cfg.pad_token_id = 0  # == position of [PAD]
    char_cfg.embedding_size = 128
    print(char_cfg)
    char_cfg.save_pretrained(save_directory="./spell_model/char_model")

    word_cfg = AlbertConfig()
    word_cfg.hidden_size = 768  # Tobe update == 768
    word_cfg.num_hidden_layers = 12  # Tobe update == 12
    word_cfg.num_attention_heads = 12  # Tobe update == 12
    word_cfg.max_position_embeddings = 192
    word_cfg.intermediate_size = 3072  # Tobe update == 3072
    word_cfg.vocab_size = 30000  # Tobe update == real vocab size
    word_cfg.pad_token_id = 0  # == position of [PAD]
    word_cfg.embedding_size = 128 + 256  # == Word Emb Size + Char Hidden Size
    word_cfg.word_embedding_size = 128
    word_cfg.classifier_dropout_prob = 0.2
    word_cfg.classifier_hidden_size = 256
    print(word_cfg)
    word_cfg.save_pretrained(save_directory="./spell_model/word_model")

    inp_word = {
        'input_ids': torch.LongTensor([[2, 1, 1, 256, 1, 3, 0, 0, 0, 0],
                                       [2, 1, 252, 142, 76, 80, 1, 60, 5, 3]]),
        'token_type_ids': torch.LongTensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        'attention_mask': torch.LongTensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    }

    inp_char = {
        'input_ids': torch.LongTensor([[2, 1, 3, 0, 0, 0, 0],
                                       [2, 1, 3, 0, 0, 0, 0],
                                       [2, 1, 3, 0, 0, 0, 0],
                                       [2, 12, 51, 4, 3, 0, 0],
                                       [2, 1, 3, 0, 0, 0, 0],
                                       [2, 1, 3, 0, 0, 0, 0],
                                       [2, 1, 3, 0, 0, 0, 0],
                                       [2, 1, 3, 0, 0, 0, 0],
                                       [2, 1, 3, 0, 0, 0, 0],
                                       [2, 1, 3, 0, 0, 0, 0],
                                       [2, 1, 3, 0, 0, 0, 0],
                                       [2, 1, 3, 0, 0, 0, 0],
                                       [2, 6, 5, 64, 3, 0, 0],
                                       [2, 63, 3, 0, 0, 0, 0],
                                       [2, 4, 5, 17, 4, 9, 3],
                                       [2, 13, 15, 3, 0, 0, 0],
                                       [2, 1, 3, 0, 0, 0, 0],
                                       [2, 5, 51, 4, 3, 0, 0],
                                       [2, 25, 3, 0, 0, 0, 0],
                                       [2, 1, 3, 0, 0, 0, 0]]),
        'token_type_ids': torch.LongTensor([[0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0]]),
        'attention_mask': torch.LongTensor([[1, 1, 1, 0, 0, 0, 0],
                                            [1, 1, 1, 0, 0, 0, 0],
                                            [1, 1, 1, 0, 0, 0, 0],
                                            [1, 1, 1, 1, 1, 0, 0],
                                            [1, 1, 1, 0, 0, 0, 0],
                                            [1, 1, 1, 0, 0, 0, 0],
                                            [1, 1, 1, 0, 0, 0, 0],
                                            [1, 1, 1, 0, 0, 0, 0],
                                            [1, 1, 1, 0, 0, 0, 0],
                                            [1, 1, 1, 0, 0, 0, 0],
                                            [1, 1, 1, 0, 0, 0, 0],
                                            [1, 1, 1, 0, 0, 0, 0],
                                            [1, 1, 1, 1, 1, 0, 0],
                                            [1, 1, 1, 0, 0, 0, 0],
                                            [1, 1, 1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 0, 0, 0],
                                            [1, 1, 1, 0, 0, 0, 0],
                                            [1, 1, 1, 1, 1, 0, 0],
                                            [1, 1, 1, 0, 0, 0, 0],
                                            [1, 1, 1, 0, 0, 0, 0]])
    }

    model = AlbertSpellChecker(char_config=char_cfg, word_config=word_cfg)

    outputs = model(
        word_input_ids=inp_word["input_ids"],
        word_attention_mask=inp_word["attention_mask"],
        word_token_type_ids=inp_word["token_type_ids"],

        char_input_ids=inp_char["input_ids"],
        char_attention_mask=inp_char["attention_mask"],
        char_token_type_ids=inp_char["token_type_ids"]
    )

    print(outputs.detection_logits.shape, outputs.correction_logits.shape)
