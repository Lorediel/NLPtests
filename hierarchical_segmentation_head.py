from torch import nn
import torch


class SegmentationHead(nn.Module):
    def __init__(self, input_size):
        super(SegmentationHead, self).__init__()

        self.firstClassificator = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.LayerNorm(input_size),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(input_size, 2),
            nn.Dropout(0.1),
        )

        self.fakeClassificator = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.LayerNorm(input_size),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(input_size, 2),
            nn.Dropout(0.1),
        )

        self.realClassificator = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.LayerNorm(input_size),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(input_size, 2),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        fake_or_real_logits = self.firstClassificator(x)
        # take the max of the logits
        fake_or_real = torch.argmax(fake_or_real_logits, dim=1)
        fake_array = []
        final_logits = []
        for f in fake_or_real: 
            if fake_or_real == 0:
                # fake
                fake_array.append("fake")
                fake_logits = self.fakeClassificator(x)
                final_logits.append(fake_logits)
            else:
                # real
                fake_array.append("real")
                real_logits =  self.realClassificator(x)
                final_logits.append(real_logits)
        return fake_array, torch.tensor(final_logits)