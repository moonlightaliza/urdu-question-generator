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
        output, (h,c) = self.lstm(packed_input)
        unpacked_output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True)

        return unpacked_output, h, c


# E = embedding size
# H = encoder hidden dim
# S = source length, T = target length, B = batch size
# V = vocabulary

import torch.nn as nn
import torch.nn.functional as F
import torch

PAD_ID = 0
BOS_ID = 2
EOS_ID = 3

MAX_DECODE_STEPS = 64

class BahdanauAttention(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.W1 = nn.Linear(H, H)
        self.W2 = nn.Linear(2*H, H)
        self.V = nn.Linear(H, 1)
    
    def forward(self, query, keys, mask=None):
        query = query.unsqueeze(1) # [B, H] -> [B, 1, H]
        scores = self.V(torch.tanh(self.W1(query) + self.W2(keys))) # [B, S, 1]
        scores = scores.squeeze(-1) # [B, S]

        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        attn_w = F.softmax(scores, dim=-1).unsqueeze(1)  # [B, 1, S]
        context_vec = torch.bmm(attn_w, keys) # [B, 1, 2H]

        return context_vec, attn_w


class Decoder(nn.Module):
    def __init__(self, embedding, H, V, E, dropout = 0.1):
        super().__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(input_size=E + 2*H, num_layers=2, hidden_size=H, batch_first=True, dropout=dropout)
        self.attention = BahdanauAttention(H)
        self.out = nn.Linear(H, V)

    
    def forward_step(self, inp, hidden, cell, encoder_out, src_mask=None):
        embed = self.embedding(inp) # (B, 1, E)
        query = hidden[-1] # (B, H)
        context, attn = self.attention(query, encoder_out, src_mask) # (B, 1, 2H)
        x = torch.cat([embed, context], dim = -1) # (B, 1, E+2H)

        output, (hidden, cell) = self.lstm(x, (hidden, cell)) # (2, B, H)
        logit = self.out(output) # (B, V)

        return logit, hidden, cell, attn 

    
    def forward(self, encoder_outs, decoder_h, decoder_c, target=None, src_mask=None):
        B = encoder_outs.size(0)
        decoder_outs = []
        attn_weights = []

        if target is not None:
            decoder_in = target[:, 0].unsqueeze(1)
            steps = target.size(1) - 1

        else:
            decoder_in = torch.full((B,1), BOS_ID, device=encoder_outs.device)
            steps = MAX_DECODE_STEPS


        for i in range(steps):
            out, hidden, cell, attn = self.forward_step(decoder_in, decoder_h, decoder_c, encoder_outs, src_mask)
            decoder_outs.append(out)
            attn_weights.append(attn)

            if target is not None: # training
                decoder_in = target[:, i + 1].unsqueeze(1)
            else: # greedy
                next_token = out[:, -1, :].argmax(dim=-1)
                decoder_in = next_token.unsqueeze(1)

                if torch.all(next_token == EOS_ID):
                    break

            decoder_outs = torch.cat(decoder_outs, dim=1)
            attentions = torch.cat(attn_weights, dim=1)
        
        return decoder_outs, decoder_h, decoder_c, attentions
        
class Bridge(nn.Module):
    def __init__(self, H):
        super().__init__()
        """
        Bridge logic: encoder's h/c come out of nn.LSTM as [num_layers*2, B, H] = [4, B, H] (layer0-fwd, layer0-bwd, layer1-fwd, layer1-bwd, in that order). For each of the 2 layers: concat that layer's fwd+bwd [B,H]+[B,H] → [B,2H], then Linear(2H,H) + tanh → [B,H]. Stack the 2 layers → [2,B,H]. Do this separately for h and c
        """
        self.layer = nn.Linear(2*H, H)

    def forward(self, encoder_h, encoder_c):
        # l0 fwd, lo bwd, l1 fwd, l1 bwd = [B, H] each
        l0 = torch.cat([encoder_h[0], encoder_h[1]], dim=1)
        l1 = torch.cat([encoder_h[2], encoder_h[3]], dim=1)

        c0 = torch.cat([encoder_c[0], encoder_c[1]], dim=1)
        c1 = torch.cat([encoder_c[2], encoder_c[3]], dim=1)

        l0 = torch.tanh(self.layer(l0))
        l1 = torch.tanh(self.layer(l1))
        c0 = torch.tanh(self.layer(c0))
        c1 = torch.tanh(self.layer(c1))

        decoder_h = torch.stack([l0, l1]) # [2, B, H]
        decoder_c = torch.stack([c0, c1]) # [2, B, H]
        return decoder_h, decoder_c



class Seq2SeqModel(nn.Module):
    def __init__(self, in_dim, V, H, E, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(in_dim, E)
        self.encoder = Encoder(self.embedding, H)
        self.bridge = Bridge(H)
        self.decoder = Decoder(self.embedding, H, V, E, dropout)

    
    def forward(self, source, lengths, target, is_train=True):
        encoder_out, encoder_h, encoder_c = self.encoder(source, lengths)
        decoder_h, decoder_c = self.bridge(encoder_h, encoder_c)
        lens_t = torch.as_tensor(lengths, device=source.device)
        longest_len = torch.arange(encoder_out.size(1), device=source.device).unsqueeze(0)
        src_mask = longest_len < lens_t.unsqueeze(1)
        
        if is_train:
            decoder_out, _, _, decoder_attn = self.decoder(encoder_out, decoder_h, decoder_c, target=target, src_mask=src_mask)
        else:
            decoder_out, _, _, decoder_attn = self.decoder(encoder_out, decoder_h, decoder_c, src_mask=src_mask)
        
        return decoder_out, decoder_attn



