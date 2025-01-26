import torch.nn as nn


class LetterRecognitionModel(nn.Module):
    def __init__(self, num_letters: int):
        super(LetterRecognitionModel, self).__init__()
        # image size: 28x28
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                1, 32, kernel_size=3, stride=1, padding=1
            ),  # [1, 28, 28] -> [32, 28, 28]
            nn.ReLU(),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=1, padding=1
            ),  # [32, 28, 28] -> [64, 28, 28]
            nn.ReLU(),
            nn.Conv2d(
                64, 128, kernel_size=3, stride=2, padding=1
            ),  # [64, 28, 28] -> [128, 14, 14]
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, num_letters),
            nn.Softmax(dim=1),
        )

    def forward(self, letter_image):
        # Feature extraction
        x = self.conv_layers(letter_image)

        # Letter prediction
        letter = self.fc_layers(x)
        return letter.unsqueeze(1)


class FontLetterRecognitionModel(nn.Module):
    def __init__(self, num_letters: int, num_fonts: int):
        super(FontLetterRecognitionModel, self).__init__()
        # image size: 28x28
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                1, 32, kernel_size=3, stride=1, padding=1
            ),  # [1, 28, 28] -> [32, 28, 28]
            nn.ReLU(),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=1, padding=1
            ),  # [32, 28, 28] -> [64, 28, 28]
            nn.ReLU(),
            nn.Conv2d(
                64, 128, kernel_size=3, stride=2, padding=1
            ),  # [64, 28, 28] -> [128, 14, 14]
            nn.ReLU(),
        )
        self.letter_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, num_letters),
            nn.Softmax(dim=1),
        )

        self.font_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, num_fonts),
            nn.Softmax(dim=1),
        )

    def forward(self, letter_image):
        # Feature extraction
        x = self.conv_layers(letter_image)

        # Letter prediction
        letter = self.letter_head(x)
        font = self.font_head(x)
        return letter.unsqueeze(1), font.unsqueeze(1)
