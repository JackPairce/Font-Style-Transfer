from torch.utils.data import dataset
from typing import List
from torch import Tensor
import numpy as np
from torchvision import transforms


class LetterDataset(dataset.Dataset):
    def __init__(self, data: List[dict], transform=None):
        self.data = data

        all_letters = list(set([d["letter"] for d in data]))
        all_letters.sort()

        self.letter_to_idx = {letter: idx for idx, letter in enumerate(all_letters)}

        self.transform = transform or transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    def total_letters(self):
        return len(self.letter_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        image = data["image"]
        letter = data["letter"]

        letter_onehot = np.zeros(len(self.letter_to_idx))
        letter_onehot[self.letter_to_idx[letter]] = 1

        if self.transform:
            image = self.transform(image).float()

        # to tensor
        letter_onehot = Tensor(letter_onehot).unsqueeze(0).float()

        return image, letter_onehot


class FontLetterDataset(dataset.Dataset):
    def __init__(self, data: List[dict], transform=None):
        self.data = data

        all_letters = list(set([d["letter"] for d in data]))
        all_letters.sort()

        all_fonts = list(set([d["font"] for d in data]))
        all_fonts.sort()

        self.letter_to_idx = {letter: idx for idx, letter in enumerate(all_letters)}
        self.font_to_idx = {font: idx for idx, font in enumerate(all_fonts)}

        self.transform = transform or transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    def total_letters(self):
        return len(self.letter_to_idx)

    def total_fonts(self):
        return len(self.font_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        image = data["image"]
        letter = data["letter"]
        font = data["font"]

        letter_onehot = np.zeros(len(self.letter_to_idx))
        letter_onehot[self.letter_to_idx[letter]] = 1

        font_onehot = np.zeros(len(self.font_to_idx))
        font_onehot[self.font_to_idx[font]] = 1

        if self.transform:
            image = self.transform(image).float()

        # to tensor
        letter_onehot = Tensor(letter_onehot).unsqueeze(0).float()
        font_onehot = Tensor(font_onehot).unsqueeze(0).float()

        return image, letter_onehot, font_onehot
