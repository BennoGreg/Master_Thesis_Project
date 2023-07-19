import os

from torch.utils.data import Dataset
import pandas as pd
import librosa
import numpy as np


class NSynthDataset(Dataset):

    def __init__(self, filename, path):
        self.filename = filename
        self.path = path
        self.data_path = os.path.join(path, "audio")
        self.samples = pd.read_json(os.path.join(path, filename)).transpose()



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples.iloc[idx, :]
        filename = "{}.wav".format(item.name)
        samplerate = item.loc["sample_rate"]
        notestr = item.note_str
        (waveform, _) = librosa.load(os.path.join(self.data_path, filename), sr=samplerate)
        return waveform, notestr, samplerate

    def get_samples_of(self, instrument_family, instrument_source):
        return self.samples[(self.samples['instrument_family'] == instrument_family) & (self.samples['instrument_source'] == instrument_source)]

    def shrink_to_single_instrument(self, instrument_family, instrument_source):
        self.samples = self.get_samples_of(instrument_family, instrument_source).sort_values(by='pitch')

    def remove_samples_of(self, instrument_family, instrument_source):
        self.samples = self.samples[(self.samples['instrument_family'] != instrument_family) | (self.samples['instrument_source'] != instrument_source)]


