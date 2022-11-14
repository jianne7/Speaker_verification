import os
import glob
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F

from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from speechbrain.pretrained import EncoderClassifier

'''
Compare fine-tuning model with original speechbrain model
'''
classifier_voice = EncoderClassifier.from_hparams(source="/home/ujinne/workspaces/projects/speaker/results/spkrec-ecapa")
classifier_speech = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
cosine_sim = torch.nn.CosineSimilarity(dim=-1)

path = '/home/ujinne/workspaces/datasets/speaker_id/data'
files = sorted([f for f in glob.glob(f'{path}/*') if os.path.isdir(f)])

# def extract_data(path, file_list, enrolled=False):
#     spk_id_list = []
#     for f in tqdm(files):
#         spk_id = f.split('/')[-1]
#         spk_id_list.append(spk_id)

#     if enrolled == True:
#         utt_list = []
#         for spk_id in spk_id_list:
#             utts = glob.glob(f'{path}/{spk_id}/*.wav')
#             # utt_list += utts[:2]
#             utt_list.append(utts[0])
#     else:
#         utt_list = []
#         for spk_id in spk_id_list:
#             utts = glob.glob(f'{path}/{spk_id}/*.wav')
#             # utt_list += utts[:2]
#             # utt_list.append(random.choice(utts[1:]))
#             utt_list.append(utts[-1]) 
#     return spk_id_list, utt_list

# enrol_spk, enrol_utt = extract_data(path, files, enrolled=True)
# test_spk, test_utt = extract_data(path, files)

def extract_spk(path, file_list):
    spk_id_list = []
    for f in tqdm(files):
        spk_id = f.split('/')[-1]
        spk_id_list.append(spk_id)
    return spk_id_list

def extract_utt(path, spk_id_list, enrolled=False):
    if enrolled == True:
        utt_list = []
        for spk_id in spk_id_list:
            utts = glob.glob(f'{path}/{spk_id}/*.wav')
            # utt_list += utts[:2]
            utt_list.append(utts[0])
    else:
        utt_list = []
        for spk_id in spk_id_list:
            utts = glob.glob(f'{path}/{spk_id}/*.wav')
            # utt_list += utts[:2]
            # utt_list.append(random.choice(utts[1:]))
            utt_list.append(utts[-1])
    return utt_list

def extract_data(path, file_list, enrolled=False):
    spk_id_list = extract_spk(path, file_list)
    if enrolled==True:
        utt_list = extract_utt(path, spk_id_list[:100], enrolled=True)
        return spk_id_list[:100], utt_list

    spk_id_list = random.choices(spk_id_list[:100], k=50) + random.choices(spk_id_list[100:], k=50)
    spk_id_list = spk_id_list + spk_id_list
    utt_list = extract_utt(path, spk_id_list)
    return spk_id_list, utt_list

labels = ([1] * 50+ [0] * 50) * 2
# labels = [1, 1, 1, 1, 0, 0, 0] + [1, 1, 1, 1, 0, 0, 0]
enrol_spk, enrol_utt = extract_data(path, files, enrolled=True)
test_spk, test_utt = extract_data(path, files)
test_spk, test_utt, labels = shuffle(test_spk, test_utt, labels)

# Enrollment of speakers
voice_embs=[]
for utt in enrol_utt:
    wav, _ = torchaudio.load(utt) # .wav, .flac 가능
    emb = classifier_voice.encode_batch(wav) # emb.size()=(1, 1, 192)
    voice_embs.append(emb)
enrolled_voice = torch.cat(voice_embs, dim=1)

speech_embs=[]
for utt in enrol_utt:
    wav, _ = torchaudio.load(utt) # .wav, .flac 가능
    emb = classifier_speech.encode_batch(wav) # emb.size()=(1, 1, 192)
    speech_embs.append(emb)
enrolled_speech = torch.cat(speech_embs, dim=1)

# Calculate cosine similarity between enrolled embeddings and test utterances
for spk, utt in zip(test_spk, test_utt):
    wav, _ = torchaudio.load(utt) # .wav, .flac 가능
    emb_voice = classifier_voice.encode_batch(wav)
    emb_speech = classifier_speech.encode_batch(wav)
    sim_voice = cosine_sim(emb_voice, enrolled_voice)
    sim_speech = cosine_sim(emb_speech, enrolled_speech)
    max_sim_voice, max_ind_voice = torch.max(sim_voice, dim=-1)
    max_sim_speech, max_ind_speech = torch.max(sim_speech, dim=-1)
    print(f'Voiceprint Model  || input_spk : {spk:9s}   |max_sim : {max_sim_voice.item():.4f}   |selected_spk : {enrol_spk[max_ind_voice]}')
    print(f'Speechbrain Model || input_spk : {spk:9s}   |max_sim : {max_sim_speech.item():.4f}   |selected_spk : {enrol_spk[max_ind_speech]}')
    print('='*100)

