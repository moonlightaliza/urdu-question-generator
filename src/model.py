import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, embedding, hidden_dim = 512):
        super().__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(
            input_size = embedding.embedding_dim,
            hidden_size = hidden_dim,
            num_layers = 2,
            bias = True,
            batch_first = True,
            dropout = 0.3,
            bidirectional = True
        )
    def forward(self, source, lengths):
        embedded = self.embedding(source)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first = True, enforce_sorted = False)    
        output, h,c = self.lstm(packed_input)
        unpacked_output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True)

        return unpacked_output, h, c
