import glob
import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset


def get_mgrid(sidelen, dim=1):
    # copied from https://github.com/YannickStruempler/inr_based_compression
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid
    
class AudioFileDataset(torch.utils.data.Dataset):
    def __init__(self, filename, start_time_sec, end_time_sec):
        self.data = []
        self.rate = 22050
        for p in glob.glob(filename, recursive=True):
            try:
                sample, rate = sf.read(p, always_2d=True)
            except: # if file is broken and cannot be parsed, skip it
                continue
            sample = sample.astype(np.float32)
            if rate != self.rate:
                sample = librosa.resample(sample, rate, self.rate)
            sample = np.mean(sample, axis=1)
            sample = np.pad(sample, ((0, end_time_sec*self.rate)), mode='constant')
            sample = sample[start_time_sec*self.rate:end_time_sec*self.rate]
            self.data.append(sample)
        
        print(len(self.data[0]))
        self.timepoints = get_mgrid(len(self.data[0]), 1)

    def get_num_samples(self):
        return self.timepoints.shape[0]

    def __len__(self):
        if isinstance(self.data, list):
            return len(self.data)
        else:
            return 1

    def __getitem__(self, idx):
        if isinstance(self.data, list):
            amplitude = self.data[idx]
        else:
            amplitude = self.data
        
        scale = np.max(np.abs(amplitude))
        amplitude = (amplitude / scale)
        amplitude = torch.Tensor(amplitude).view(-1, 1)
        return {'timepoints': self.timepoints, 'amplitude': amplitude}

class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, start_time_sec, end_time_sec):
        self.rate = 22050
        self.start_time_sec = start_time_sec
        self.end_time_sec = end_time_sec
        self.data = load_dataset("librispeech_asr", "clean", split="train.100")
        self.librispeech_sr = self.data[0]['audio']['sampling_rate']
        
        self.timepoints = get_mgrid(self.rate*(end_time_sec-start_time_sec), 1)

    def get_num_samples(self):
        return self.timepoints.shape[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        amplitude = self.data[idx]['audio']['array']
        amplitude = librosa.resample(amplitude, orig_sr=self.librispeech_sr, target_sr=self.rate).astype(np.float32)
        
        amplitude = np.pad(amplitude, (0, self.end_time_sec*self.rate), mode='constant')
        assert (self.end_time_sec-self.start_time_sec)*self.rate < len(amplitude)
        amplitude = amplitude[self.start_time_sec*self.rate:self.end_time_sec*self.rate]
        scale = np.max(np.abs(amplitude))
        amplitude = (amplitude / scale)
        amplitude = torch.Tensor(amplitude).view(-1, 1)
        return {'timepoints': self.timepoints, 'amplitude': amplitude}