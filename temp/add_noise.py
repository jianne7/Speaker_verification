import os
import shutil
import glob
import math
import random
import numpy as np
from tqdm import tqdm

import librosa
import soundfile

def get_white_noise(signal,SNR) :
    #RMS value of signal
    RMS_s=math.sqrt(np.mean(signal**2))
    #RMS values of noise
    RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
    #Additive white gausian noise. Thereore mean=0
    #Because sample length is large (typically > 40000)
    #we can use the population formula for standard daviation.
    #because mean=0 STD=RMS
    STD_n=RMS_n
    noise=np.random.normal(0, STD_n, signal.shape[0])
    return 

def get_noise_from_sound(signal, noise, SNR):
    RMS_s = math.sqrt(np.mean(signal ** 2))
    # required RMS of noise
    RMS_n = math.sqrt(RMS_s ** 2 / (pow(10, SNR / 10)))

    # current RMS of noise
    RMS_n_current = math.sqrt(np.mean(noise ** 2))
    noise = noise * (RMS_n / RMS_n_current)

    return noise

def get_speaker(path):
    files = glob.glob(f'{path}/*')
    spk_id_list = []
    for f in tqdm(files):
        spk_id = f.split('/')[-1]
        spk_id_list.append(spk_id)
    # spk_ids = random.sample(spk_id_list, num_spks)
    return spk_id_list

def get_utterance(path, spk_id):
    utts = glob.glob(f'{path}/{spk_id}/*.wav')
    utt = random.choice(utts)       
    return utt

random.seed(777)

raw_path = '/home/ujinne/workspaces/datasets/speaker_id/data'
noise_path = '/home/ujinne/workspaces/datasets/speaker_id/RIRS_NOISES/pointsource_noises'
# save_path = '/home/ubuntu/speaker_recognition/preprocessed'
# os.makedirs(save_path, exist_ok=True)

spk_ids = get_speaker(raw_path)
random.shuffle(spk_ids)
enrolled_spks = spk_ids[:100]
impostor_spks = spk_ids[100:]
noise_files = glob.glob(f'{noise_path}/*.wav')

print("------------------Start making Noise data------------------")
for spk_id in tqdm(enrolled_spks):
    utt = get_utterance(raw_path, spk_id)
    audio, _ = librosa.load(utt, sr=16000)
    noise, _ = librosa.load(random.choice(noise_files), sr=16000)

    # length nomalization
    start = random.randint(0, len(noise) - len(audio))
    split_noise = noise[start : start + len(audio)]

    # get noise
    snr = np.random.choice([2, 3, 4, 5, 6], p=[0.1, 0.2, 0.2, 0.2, 0.3])
    noise = get_noise_from_sound(audio, split_noise, snr)
    signal_noise = audio + noise
    
    filename = utt.split('/')[-1].split('.')[0]
    enrolled_path = f'{save_path}/enrolled/{spk_id}'
    os.makedirs(enrolled_path, exist_ok=True)

    if utt.endswith('.wav'):
        shutil.copyfile(utt, f"{enrolled_path}/{filename}.wav")
    else:
        soundfile.write(f"{enrolled_path}/{filename}.wav", audio, 16000) # save original file to wav
    
    mixed_path = f'{save_path}/enrolled_noise/{spk_id}'
    os.makedirs(mixed_path, exist_ok=True)
    soundfile.write(f"{mixed_path}/{filename}.wav", signal_noise, 16000) # save mixed file
print("------------------Finished making Noise data------------------")
print()
print("------------------Start making Imposters' data------------------")
for spk_id in tqdm(impostor_spks):
    utt = get_utterance(raw_path, spk_id)
    filename = utt.split('/')[-1].split('.')[0]
    impostor_path = f'{save_path}/impostors/{spk_id}'
    os.makedirs(impostor_path, exist_ok=True)

    if utt.endswith('.wav'):
        shutil.copyfile(utt, f"{impostor_path}/{filename}.wav")
    else:
        audio, _ = librosa.load(utt, sr=16000)
        soundfile.write(f"{impostor_path}/{filename}.wav", audio, 16000)
print("------------------Finished making Impostor data------------------")

