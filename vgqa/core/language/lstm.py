import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchtext


class RNNEncoder(nn.Module):
    """RNN-based text encoder with GloVe embeddings."""

    def __init__(
        self,
        vocab_dir,
        hidden_size,
        bidirectional: bool = False,
        dropout_p: float = 0,
        n_layers: int = 1,
        rnn_type: str = 'lstm',
    ):
        super(RNNEncoder, self).__init__()

        vocab = load_vocab(vocab_dir)
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=True)
        word_embed_size = vocab.vectors.shape[1]

        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type.upper())(
            word_embed_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_p,
        )
        self.num_dirs = 2 if bidirectional else 1
        self.variable_lengths = True

    def forward(self, text_data):
        """Encode text through RNN and return final hidden states."""
        text_tensors = text_data.tensors
        text_masks = text_data.mask

        input_lengths = (text_masks != 0).sum(1)
        input_lengths_list = input_lengths.data.cpu().numpy().tolist()

        sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist()
        sort_ixs = np.argsort(input_lengths_list)[::-1].tolist()
        s2r = {s: r for r, s in enumerate(sort_ixs)}
        recover_ixs = [s2r[s] for s in range(len(input_lengths_list))]

        sort_ixs = text_masks.data.new(sort_ixs).long()
        recover_ixs = text_masks.data.new(recover_ixs).long()

        text_tensors = text_tensors[sort_ixs]
        text_masks = text_masks[sort_ixs]

        embedded = self.embedding(text_tensors)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_input_lengths_list, batch_first=True)

        output, hidden = self.rnn(embedded)

        embedded, _ = nn.utils.rnn.pad_packed_sequence(embedded, batch_first=True)
        embedded = embedded[recover_ixs]

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = output[recover_ixs]

        sent_output = []
        for ii in range(output.shape[0]):
            sent_output.append(output[ii, int(input_lengths_list[ii] - 1), :])
        return torch.stack(sent_output, dim=0)


def load_vocab(vocab_dir):
    """Load or download GloVe vocabulary and embeddings."""
    vocab_pth = os.path.join(vocab_dir,'vocab.pth')
    if not os.path.exists(vocab_pth):
        # Download GloVe 300d embeddings
        vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"](cache=vocab_dir)
        # Add unknown token
        vocab.itos.extend(['<unk>'])
        vocab.stoi['<unk>'] = vocab.vectors.shape[0]
        vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
        torch.save(vocab,vocab_pth)
    else:
        vocab = torch.load(vocab_pth)

    return vocab