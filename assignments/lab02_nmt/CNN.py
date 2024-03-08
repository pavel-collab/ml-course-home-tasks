import torch
from torch import nn

class CnnTranslator(nn.Module):
    def __init__(self,
                 emdbed_size,
                 hidden_size,
                 num_classes=4):
        super().__init__()

        self.embeddings = nn.Embedding(len(word2idx), embedding_dim=emdbed_size)
        self.cnn = nn.Sequential(
            nn.Conv1d(emdbed_size, hidden_size, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )

        self.cl = nn.Sequential(
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        x = self.embeddings(x)
        x = x.permutate(0, 2, 1)
        x = self.cnn(x)
        prediction = self.cl(x)
        return prediction