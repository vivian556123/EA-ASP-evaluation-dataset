import numpy as np
import os
import h5py
import soundfile as sf
from scipy import signal
import fire
import random
from tqdm import tqdm
import pypeln as pl


def norm_wav(wav):
    #  norm wav value to [-1.0, 1.0]
    norm = max(np.absolute(wav))
    if norm > 1e-5:
        wav = wav / norm
    return wav


class AugmentWAV_H5(object):
    def __init__(self, musan_h5_idx, rir_h5_idx):

        self.noise_snr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.num_noise = {'noise': [1, 1], 'speech': [3, 7], 'music': [1, 1]}

        self.noise_f_dict = None
        self.rir_f_dict = None

        self.noise_f_set = set()
        self.rir_f_set = set()

        self.noise_dict = {}
        self.rir_list = []

        with open(musan_h5_idx, 'r') as f:
            for line in f.readlines():
                segs = line.strip().split()
                noise_type = segs[0].split('/')[0]

                if noise_type not in self.noise_dict:
                    self.noise_dict[noise_type] = []
                h5_file_name = '/'.join(segs[2].split('/')[-2:])
                # utt_name  utt_len  idx_for_h5_f
                self.noise_dict[noise_type].append((segs[0], int(segs[1]), h5_file_name))

                self.noise_f_set.add(segs[2])

        with open(rir_h5_idx, 'r') as f:
            for line in f.readlines():
                segs = line.strip().split()

                h5_file_name = '/'.join(segs[2].split('/')[-2:])
                self.rir_list.append((segs[0], int(segs[1]), h5_file_name))
                self.rir_f_set.add(segs[2])

    def get_random_chunk_start(self, data_len, chunk_len):
        adjust_chunk_len = min(data_len, chunk_len)
        chunk_start = random.randint(0, data_len - adjust_chunk_len)
        return chunk_start, adjust_chunk_len

    def additive_noise(self, noise_type, audio, audio_sr=16000):
        '''
        :param noise_type: 'noise', 'speech', 'music'
        :param audio: numpy array, (audio_len,)
        '''
        if self.noise_f_dict is None:
            self.noise_f_dict = {}
            for h5_file_path in self.noise_f_set:
                h5_file_name = '/'.join(h5_file_path.split('/')[-2:])
                self.noise_f_dict[h5_file_name] = h5py.File(h5_file_path, 'r')

        audio = audio.astype(np.float32)
        audio = norm_wav(audio)
        audio_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)
        audio_len = audio.shape[0]
        if audio_sr == 8000:
            audio_len = audio_len * 2

        num_noise = self.num_noise[noise_type]
        noise_idx_list = random.sample(self.noise_dict[noise_type], random.randint(num_noise[0], num_noise[1]))

        noise_list = []
        for noise_idx, noise_len, file_name in noise_idx_list:
            chunk_start, chunk_len = self.get_random_chunk_start(noise_len, audio_len)
            noise = self.noise_f_dict[file_name][noise_idx][chunk_start:chunk_start+chunk_len]
            noise = np.resize(noise, (audio_len,)).astype(np.float32)
            noise = norm_wav(noise)
            if audio_sr == 8000:
                noise = noise[::2]

            noise_snr = random.uniform(self.noise_snr[noise_type][0], self.noise_snr[noise_type][1])
            noise_db = 10 * np.log10(np.mean(noise ** 2) + 1e-4)
            noise_list.append(np.sqrt(10 ** ((audio_db - noise_db - noise_snr) / 10)) * noise)

        return np.sum(np.stack(noise_list), axis=0) + audio

    def reverberate(self, audio, audio_sr=16000):
        '''
        :param audio: numpy array, (audio_len,)
        '''
        if self.rir_f_dict is None:
            self.rir_f_dict = {}
            for h5_file_path in self.rir_f_set:
                h5_file_name = '/'.join(h5_file_path.split('/')[-2:])
                self.rir_f_dict[h5_file_name] = h5py.File(h5_file_path, 'r')

        audio = audio.astype(np.float32)
        audio = norm_wav(audio)
        audio_len = audio.shape[0]

        rir_idx, rir_len, file_name = random.choice(self.rir_list)
        rir_audio = self.rir_f_dict[file_name][rir_idx][()]

        if audio_sr == 8000:
            rir_audio = rir_audio[::2]
        rir_audio = rir_audio.astype(np.float32)
        rir_audio = rir_audio / np.sqrt(np.sum(rir_audio ** 2))

        return signal.convolve(audio, rir_audio, mode='full')[:audio_len]

    def add_noise(self, audio, audio_sr=16000):
        augtype = random.randint(1, 4)
        if augtype == 1:
            audio = self.reverberate(audio, audio_sr)
        elif augtype == 2:
            audio = self.additive_noise('music', audio, audio_sr)
        elif augtype == 3:
            audio = self.additive_noise('speech', audio, audio_sr)
        elif augtype == 4:
            audio = self.additive_noise('noise', audio, audio_sr)

        return audio


def mix_utt(audio1_path, audio2_path, overlap_ratio=0.0, snr=0.0):
    audio1, _ = sf.read(audio1_path)
    audio2, _ = sf.read(audio2_path)

    audio1 = norm_wav(audio1)
    audio2 = norm_wav(audio2)
    audio1_len = audio1.shape[0]
    audio2_len = audio2.shape[0]

    mix_len = (audio1_len+audio2_len)/(1.0+overlap_ratio)
    mix_len = int(mix_len)
    overlap_len = int(mix_len*overlap_ratio)
    audio2_len = mix_len + overlap_len - audio1_len
    audio2 = audio2[:audio2_len]

    audio1_db = 10 * np.log10(np.mean(audio1 ** 2) + 1e-4)
    audio2_db = 10 * np.log10(np.mean(audio2 ** 2) + 1e-4)
    audio2 = np.sqrt(10 ** ((audio1_db - audio2_db - snr) / 10)) * audio2

    mix_audio = np.zeros(mix_len)
    mix_audio[:audio1_len] = mix_audio[:audio1_len] + audio1
    mix_audio[-audio2_len:] = mix_audio[-audio2_len:] + audio2

    mix_audio = norm_wav(mix_audio)

    return mix_audio


musan_h5_idx = '/mnt/lustre/sjtu/home/czy97/sid/Domain-Adaptation/data/noise_h5/musan_h5_idx'
rir_h5_idx = '/mnt/lustre/sjtu/home/czy97/sid/Domain-Adaptation/data/noise_h5/rir_h5_idx'
aug_obj = AugmentWAV_H5(musan_h5_idx, rir_h5_idx)


def aug_func(values):
    line, data_dir = values
    test_wav_name = line.strip().split()[2]
    test_wav_path = os.path.join(data_dir, test_wav_name)

    audio, sr = sf.read(test_wav_path)
    aug_audio = aug_obj.add_noise(audio, sr)
    aug_audio = norm_wav(aug_audio)

    store_name = test_wav_name[:-4] + '-mix.wav'
    return aug_audio, store_name


def mix_func(values):
    line, data_dir = values
    segs = line.strip().split()

    test_wav_name, mix_wav_name, snr, overlap_ratio = segs[2], segs[-3], float(segs[-2]), float(segs[-1])
    test_wav_path = os.path.join(data_dir, test_wav_name)
    mix_wav_path = os.path.join(data_dir, mix_wav_name)

    new_audio = mix_utt(test_wav_path, mix_wav_path, overlap_ratio, snr)
    new_audio = norm_wav(new_audio)

    store_name = test_wav_name[:-4] + '-mix.wav'
    return new_audio, store_name


def aug_mix_func(values):
    line, data_dir = values
    segs = line.strip().split()

    test_wav_name, mix_wav_name, snr, overlap_ratio = segs[2], segs[-3], float(segs[-2]), float(segs[-1])
    test_wav_path = os.path.join(data_dir, test_wav_name)
    mix_wav_path = os.path.join(data_dir, mix_wav_name)

    new_audio = mix_utt(test_wav_path, mix_wav_path, overlap_ratio, snr)
    new_audio = aug_obj.add_noise(new_audio)
    new_audio = norm_wav(new_audio)

    store_name = test_wav_name[:-4] + '-mix.wav'
    return new_audio, store_name


def mix_aug_trial(trial_path, store_dir, data_dir='/mnt/lustre/sjtu/shared/data/raa/oxford/voxceleb1_wav_v2', mix=True, aug=True, num_process=4):

    with open(trial_path, 'r') as f:
        lines = f.readlines()

    if mix and aug:
        process_func = aug_mix_func
    elif mix:
        process_func = mix_func
    else:
        process_func = aug_func

    data_dir_list = [data_dir] * len(lines)
    t_bar = tqdm(ncols=100, total=len(lines))
    for new_audio, store_name in pl.process.map(process_func, zip(lines, data_dir_list), workers=num_process,
                                                             maxsize=num_process+1):
        t_bar.update()

        store_path = os.path.join(store_dir, store_name)
        store_folder = os.path.dirname(store_path)
        os.makedirs(store_folder, exist_ok=True)

        sf.write(store_path, new_audio, 16000)

    t_bar.close()


def main(wav1, wav2, store_path, overlap_ratio, snr):
    musan_h5_idx = '/mnt/lustre/sjtu/home/czy97/sid/Domain-Adaptation/data/noise_h5/musan_h5_idx'
    rir_h5_idx = '/mnt/lustre/sjtu/home/czy97/sid/Domain-Adaptation/data/noise_h5/rir_h5_idx'

    aug_obj = AugmentWAV_H5(musan_h5_idx, rir_h5_idx)

    new_wav = mix_utt(wav1, wav2, overlap_ratio, snr)
    new_wav = aug_obj.add_noise(new_wav)
    new_wav = norm_wav(new_wav)

    sf.write(store_path, new_wav, 16000)

if __name__ == "__main__":
    fire.Fire(mix_aug_trial)
