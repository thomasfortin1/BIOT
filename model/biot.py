import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_
import numpy as np
from linear_attention_transformer import LinearAttentionTransformer
import mup

def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return nn.init.normal_(tensor, mean=0.0, std=sigma)
    return init_

class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size=256, n_freq=101, encoder_var=1):
        super().__init__()
        self.init_method = init_method_normal((encoder_var / n_freq) ** .5)
        self.projection = nn.Linear(n_freq, emb_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.init_method(self.projection.weight)
        constant_(self.projection.bias, 0.) 

    def forward(self, x):
        """
        x: (batch, 1, freq, time)
        out: (batch, time, emb_size)
        """
        b, _, _, _ = x.shape
        x = x.squeeze(1).permute(0, 2, 1)
        x = self.projection(x)
        return x


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes, use_mup=False, readout_zero_init=False, output_mult=1.0):
        super().__init__()
        if use_mup:
            self.clshead = nn.Sequential(
                nn.ELU(),
                mup.MuReadout(emb_size, n_classes, readout_zero_init=readout_zero_init, output_mult=output_mult)
            )
            print('readout_zero_init', readout_zero_init)
            print('output_mult', output_mult)
        else:
            self.clshead = nn.Sequential(
                nn.ELU(),
                nn.Linear(emb_size, n_classes),
            )

    def forward(self, x):
        out = self.clshead(x)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class BIOTEncoder(nn.Module):
    def __init__(
        self,
        emb_size=256,
        encoder_var=1,
        heads=8,
        depth=4,
        n_channels=16,
        n_fft=200,
        hop_length=100,
        scaling=None,
        use_mup=False,
        **kwargs
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.encoder_var = encoder_var

        self.patch_embedding = PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1,
            encoder_var=encoder_var
        )
        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            encoder_var=encoder_var,
            heads=heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=0.2,  # dropout right after self-attention layer
            attn_dropout=0.2,  # dropout post-attention
            output_mult=1,
            # local_attn_window_size=emb_size,
            # scaling=scaling
            use_mup=use_mup
            # use_mup=True
        )
        self.positional_encoding = PositionalEncoding(emb_size)

        # channel token, N_channels >= your actual channels
        self.channel_tokens = nn.Embedding(n_channels, emb_size)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )
        if use_mup:
            print("using mup")
        else:
            print("not using mup")

    def stft(self, sample):
        signal = []
        for s in range(sample.shape[1]):
            spectral = torch.stft(
                sample[:, s, :],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                normalized=False,
                center=False,
                onesided=True,
                return_complex=True,
            )
            signal.append(spectral)
        stacked = torch.stack(signal).permute(1, 0, 2, 3)
        return torch.abs(stacked)

    def forward(self, x, n_channel_offset=0, perturb=False):
        """
        x: [batch_size, channel, ts]
        output: [batch_size, emb_size]
        """
        emb_seq = []
        for i in range(x.shape[1]):
            channel_spec_emb = self.stft(x[:, i : i + 1, :])
            channel_spec_emb = self.patch_embedding(channel_spec_emb)
            batch_size, ts, _ = channel_spec_emb.shape
            # (batch_size, ts, emb)
            channel_token_emb = (
                self.channel_tokens(self.index[i + n_channel_offset])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )
            # (batch_size, ts, emb)
            channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb)

            # perturb
            if perturb:
                ts = channel_emb.shape[1]
                ts_new = np.random.randint(ts // 2, ts)
                selected_ts = np.random.choice(range(ts), ts_new, replace=False)
                channel_emb = channel_emb[:, selected_ts]
            emb_seq.append(channel_emb)

        # (batch_size, 16 * ts, emb)
        emb = torch.cat(emb_seq, dim=1)
        # (batch_size, emb)
        emb = self.transformer(emb).mean(dim=1)
        return emb


# supervised classifier module
class BIOTClassifier(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_classes=6, readout_zero_init=False, output_mult=1.0, **kwargs):
        super().__init__()
        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth, **kwargs)
        self.classifier = ClassificationHead(emb_size, n_classes, use_mup=kwargs.get("use_mup", False), readout_zero_init=readout_zero_init, output_mult=output_mult)

    def forward(self, x):
        x = self.biot(x)
        x = self.classifier(x)
        return x


# unsupervised pre-train module
class UnsupervisedPretrain(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_channels=18, encoder_var=1, use_mup=False, **kwargs):
        super(UnsupervisedPretrain, self).__init__()
        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth, n_channels=n_channels, encoder_var=encoder_var, use_mup=use_mup, **kwargs)
        self.fn1 = nn.Linear(emb_size, emb_size)
        # self.fn2 = mup.MuReadout(emb_size, emb_size, bias=False) # include output mult later
        self.fn2 = nn.Linear(emb_size, emb_size)
        self.gelu = nn.GELU()
        self.init_method = init_method_normal((encoder_var/emb_size) ** .5)
        if use_mup:
            print("using mup")
            self.reset_parameters()
        else:
            print("not using mup")

    def reset_parameters(self):
        # nn.init.kaiming_normal_(self.fn1.weight, a=1, mode='fan_in')
        # nn.init.kaiming_normal_(self.fn2.weight, a=1, mode='fan_in')
        
        self.init_method(self.fn2.weight)
        self.init_method(self.fn1.weight)
        # nn.init.zeros_(self.fn2.weight)
        constant_(self.fn1.bias, 0.)
        constant_(self.fn2.bias, 0.)
        # self.fn2.weight.data.zero_()
        # self.fn2.bias.data.zero_()

    def forward(self, x, n_channel_offset=0):
        emb = self.biot(x, n_channel_offset, perturb=True)
        emb = self.fn2(self.gelu(self.fn1(emb)))
        pred_emb = self.biot(x, n_channel_offset)
        return emb, pred_emb


# supervised pre-train module
class SupervisedPretrain(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, **kwargs):
        super().__init__()
        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth)
        self.classifier_chb_mit = ClassificationHead(emb_size, 1)
        self.classifier_iiic_seizure = ClassificationHead(emb_size, 6)
        self.classifier_tuab = ClassificationHead(emb_size, 1)
        self.classifier_tuev = ClassificationHead(emb_size, 6)

    def forward(self, x, task="chb-mit"):
        x = self.biot(x)
        if task == "chb-mit":
            x = self.classifier_chb_mit(x)
        elif task == "iiic-seizure":
            x = self.classifier_iiic_seizure(x)
        elif task == "tuab":
            x = self.classifier_tuab(x)
        elif task == "tuev":
            x = self.classifier_tuev(x)
        else:
            raise NotImplementedError
        return x


if __name__ == "__main__":
    x = torch.randn(16, 2, 2000)
    model = BIOTClassifier(n_fft=200, hop_length=200, depth=4, heads=8)
    out = model(x)
    print(out.shape)

    model = UnsupervisedPretrain(n_fft=200, hop_length=200, depth=4, heads=8)
    out1, out2 = model(x)
    print(out1.shape, out2.shape)
