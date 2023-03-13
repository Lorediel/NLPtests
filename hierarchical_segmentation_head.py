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
        fake_indexes = []
        for f in fake_or_real: 
            if f == 0:
                # fake
                fake_array.append("fake")
                fake_indexes.append(1)
                real_indexes.append(0)
            else:
                # real
                fake_array.append("real")
                fake_indexes.append(0)
                real_indexes.append(1)
                #real_logits =  self.realClassificator(x)
                #final_logits.append(real_logits)
        # take only the x that are fake
        fake_indexes = torch.tensor(fake_indexes).to(self.device)
        fake_x = torch.masked_select(x, fake_indexes)
        fake_logits = self.fakeClassificator(fake_x)
        
        # take only the x that are real
        real_indexes = torch.tensor(real_indexes).to(self.device)
        real_x = torch.masked_select(x, fake_indexes)
        real_logits = self.realClassificator(real_x)

        for index in range(len(fake_or_real)):
            if fake_or_real[index] == 0:
                final_logits.append(fake_logits[index])
            else:
                final_logits.append(real_logits[index])
        
        return fake_array, torch.stack(final_logits, dim=0)