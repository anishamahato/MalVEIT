class ClassificationHead(nn.Module):

    def __init__(self, latent_dim=256, num_classes=25):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, z):

        logits = self.classifier(z)

        return logits
