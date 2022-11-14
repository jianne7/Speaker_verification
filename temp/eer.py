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
from speechbrain.utils.metric_stats import EER

classifier_voice = EncoderClassifier.from_hparams(source="/home/ujinne/workspaces/projects/speaker/results/spkrec-ecapa")
classifier_speech = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
cosine_sim = torch.nn.CosineSimilarity(dim=-1)

path = '/home/ujinne/workspaces/datasets/speaker_id/data'
files = sorted([f for f in glob.glob(f'{path}/*') if os.path.isdir(f)])

def index2char(speaker_id: list):
    txt2id = {f'{txt}':i for i, txt in enumerate(speaker_id)}
    id2txt = {int(value):key for key, value in txt2id.items()}
    return txt2id, id2txt

def compute_eer(scores, labels):
    """
    Compute the equal error rate score
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label = 1) # output=fpr, tpr, threshold
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer

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
    # spk_id_list = spk_id_list + spk_id_list
    utt_list = extract_utt(path, spk_id_list)
    return spk_id_list, utt_list

labels = [1] * 50 + [0] * 50
# labels = ([1] * 25+ [0] * 25) * 2
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

print("")
print("------------ EER Speechbrain version ------------")
pos_scores = []
neg_scores = []
for i, (spk, utt) in enumerate(zip(test_spk, test_utt)):
    if labels[i] == 1:
        enrol_id = spk
        enrol_idx = enrol_spk.index(enrol_id)
        enrol = enrolled_voice[0][enrol_idx]
    else:
        enrol_id = random.choice(enrol_spk)
        enrol_idx = enrol_spk.index(enrol_id)
        enrol = enrolled_voice[0][enrol_idx]
    
    test_id = spk
    wav, _ = torchaudio.load(utt)
    test = classifier_voice.encode_batch(wav)

    enrol_rep = enrol.repeat(enrolled_voice.shape[0], 1, 1)
    score_e = cosine_sim(enrol_rep, enrolled_voice)
    mean_e = torch.mean(score_e, dim=-1)
    std_e = torch.std(score_e, dim=-1)

    test_rep = test.repeat(enrolled_voice.shape[0], 1, 1)
    score_t = cosine_sim(test_rep, enrolled_voice)
    mean_t = torch.mean(score_t, dim=-1)
    std_t = torch.std(score_t, dim=-1)

    score = cosine_sim(enrol, test)[0]

    score_e = (score - mean_e) / std_e
    score_t = (score - mean_t) / std_t
    score = 0.5 * (score_e+score_t)

    if labels[i]==1:
        pos_scores.append(score.item())
    else:
        neg_scores.append(score.item())

eer, th = EER(torch.tensor(pos_scores), torch.tensor(neg_scores))
print(f'Voiceprint  || Equal Error Rate(%): {eer*100:.4f}')

pos_scores = []
neg_scores = []
for i, (spk, utt) in enumerate(zip(test_spk, test_utt)):
    if labels[i] == 1:
        enrol_id = spk
        enrol_idx = enrol_spk.index(enrol_id)
        enrol = enrolled_speech[0][enrol_idx]
    else:
        enrol_id = random.choice(enrol_spk)
        enrol_idx = enrol_spk.index(enrol_id)
        enrol = enrolled_speech[0][enrol_idx]
    
    test_id = spk
    wav, _ = torchaudio.load(utt)
    test = classifier_speech.encode_batch(wav)

    enrol_rep = enrol.repeat(enrolled_speech.shape[0], 1, 1)
    score_e = cosine_sim(enrol_rep, enrolled_speech)
    mean_e = torch.mean(score_e, dim=-1)
    std_e = torch.std(score_e, dim=-1)

    test_rep = test.repeat(enrolled_speech.shape[0], 1, 1)
    score_t = cosine_sim(test_rep, enrolled_speech)
    mean_t = torch.mean(score_t, dim=-1)
    std_t = torch.std(score_t, dim=-1)

    score = cosine_sim(enrol, test)[0]

    score_e = (score - mean_e) / std_e
    score_t = (score - mean_t) / std_t
    score = 0.5 * (score_e+score_t)

    if labels[i]==1:
        pos_scores.append(score.item())
    else:
        neg_scores.append(score.item())

# print(f'pos_scores : {pos_scores}')
# print(f'neg_scores : {neg_scores}')
eer, th = EER(torch.tensor(pos_scores), torch.tensor(neg_scores))
print(f'Speechbrain || Equal Error Rate(%): {eer*100:.4f}')
print("")

print("="* 100)
print("")
print("------------ EER function version ------------")
# Calculate cosine similarity between enrolled embeddings and test utterances
preds_voice=[]
preds_speech=[]
for spk, utt in zip(test_spk, test_utt):
    wav, _ = torchaudio.load(utt) # .wav, .flac 가능
    emb_voice = classifier_voice.encode_batch(wav)
    emb_speech = classifier_speech.encode_batch(wav)
    sim_voice = cosine_sim(emb_voice, enrolled_voice)
    sim_speech = cosine_sim(emb_speech, enrolled_speech)
    max_sim_voice, max_ind_voice = torch.max(sim_voice, dim=-1)
    max_sim_speech, max_ind_speech = torch.max(sim_speech, dim=-1)
    # print(f'Voiceprint Model  || input_spk : {spk:9s}   |max_sim : {max_sim_voice.item():.4f}   |selected_spk : {enrol_spk[max_ind_voice]}')
    # print(f'Speechbrain Model || input_spk : {spk:9s}   |max_sim : {max_sim_speech.item():.4f}   |selected_spk : {enrol_spk[max_ind_speech]}')
    # print('='*100)

    pred_voice = F.softmax(sim_voice.squeeze(0), dim=-1)
    preds_voice.append(pred_voice[max_ind_voice].item())
    pred_speech = F.softmax(sim_speech.squeeze(0), dim=-1)
    preds_speech.append(pred_speech[max_ind_speech].item())

eer_voice = compute_eer(preds_voice, labels)
eer_speech = compute_eer(preds_speech, labels)
print(f'Voiceprint  || Equal Error Rate(%): {eer_voice*100:.4f}')
print(f'Speechbrain || Equal Error Rate(%): {eer_speech*100:.4f}')
print("")
