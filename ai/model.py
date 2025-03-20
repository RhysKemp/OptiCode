import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(Seq2SeqModel, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim,
                               num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim,
                               num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, trg):
        # Encode the source sequence
        _, (hidden, cell) = self.encoder(src)

        # Decode the target sequence
        outputs, _ = self.decoder(trg, (hidden, cell))

        outputs = self.fc(outputs)
        return outputs
