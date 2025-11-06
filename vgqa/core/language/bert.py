import torch
import torch.nn.functional as F

from torch import nn
from vgqa.utils.video_containers import NestedTensor

from pytorch_pretrained_bert.modeling import BertModel
from transformers import RobertaModel, RobertaTokenizerFast


class BERT(nn.Module):
    """BERT text encoder with configurable layer extraction."""

    def __init__(self, name: str, train_bert: bool, enc_num, pretrain_weight):
        super().__init__()
        if name == 'bert-base-uncased':
            self.num_channels = 768
        else:
            self.num_channels = 1024

        self.enc_num = enc_num
        self.bert = BertModel.from_pretrained(pretrain_weight)

        if not train_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):
        """Encode text through BERT and return specified layer output."""
        if self.enc_num > 0:
            all_encoder_layers, _ = self.bert(
                tensor_list.tensors, token_type_ids=None, attention_mask=tensor_list.mask
            )
            xs = all_encoder_layers[self.enc_num - 1]
        else:
            xs = self.bert.embeddings.word_embeddings(tensor_list.tensors)

        mask = tensor_list.mask.to(torch.bool)
        mask = ~mask
        out = NestedTensor(xs, mask)
        return out


class Roberta(nn.Module):
    """RoBERTa text encoder with feature dimension resizing."""

    def __init__(self, name, outdim, freeze=False) -> None:
        super().__init__()
        self.body = RobertaModel.from_pretrained(name)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_name_or_path=name)

        if freeze:
            for p in self.body.parameters():
                p.requires_grad_(False)

        config = self.body.config
        self.resizer = FeatureResizer(
            input_feat_size=config.hidden_size,
            output_feat_size=outdim,
            dropout=0.1,
        )

    def forward(self, texts, device):
        """Encode text and return resized features for cross-modal fusion."""
        tokenized = self.tokenizer.batch_encode_plus(texts, padding="longest", return_tensors="pt").to(device)
        encoded_text = self.body(**tokenized)
        text_cls = encoded_text.pooler_output

        text_memory = encoded_text.last_hidden_state.transpose(0, 1)
        text_attention_mask = tokenized.attention_mask.ne(1).bool()

        text_memory_resized = self.resizer(text_memory)
        text_cls_resized = self.resizer(text_cls)

        return (text_attention_mask, text_memory_resized, text_memory), text_cls_resized


class FeatureResizer(nn.Module):
    """Resize feature dimensions with linear projection and layer normalization."""

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        """Project features to target dimension with normalization and dropout."""
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output