import os
import glob
import shutil
import logging
import math
import random
import numpy as np
from tqdm import tqdm
import librosa
import soundfile
import torchaudio

import torch
import torch.nn
import torch.nn.functional as F

from speechbrain.pretrained import EncoderClassifier

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def get_logger(name: str, file_path: str, stream=False) -> logging.RootLogger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path)

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if stream:
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger

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

def extract_files(path, spk_id):
    '''
    Extract files per speaker (5 sec or more)
    '''
    utts = glob.glob(f'{path}/{spk_id}/*.wav')
    utt_list = []
    for utt in utts:
        duration = librosa.get_duration(filename=utt)
        # 5초 이상인 파일만 선택
        if duration >= 5:
            utt_list.append(utt)
        else:
            pass
    return utt_list

# Speaker Enrollment
def enrollment(wav_list, model):
    spk_ids = []
    embeddings = []
    for utt in tqdm(wav_list):
        wav, _ = torchaudio.load(utt) # .wav, .flac 가능
        emb = model.encode_batch(wav) # emb.size()=(1, 1, 192)
        # print(f'emb:{emb.shape}')
        embeddings.append(emb)
        spk_ids.append(utt.split('/')[-2])
    enrolled_embs = torch.cat(embeddings, dim=1) # enrolled_embs.size()=(1, num_spks, 192)
    return spk_ids, enrolled_embs

def compute_cossim(input_emb, emb_matirx):
    '''
    input_emb.size() = (1, 1, 192)
    emb_matrix.size() = (1, num_spks, 192)
    '''
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    similarity = cosine_sim(input_emb, emb_matirx) # similarity.size()=(1, 100)

    return similarity

def compute_eer(scores, labels):
    """
    Compute the equal error rate score
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label = 1) # output=fpr, tpr, threshold
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
   
    return eer

def extract_data(filelist):
    for file_path in tqdm(filelist):
        spk_id = file_path.split('/')[-2]
        filename = file_path.split('/')[-1]
        extract_path = f'/home/ujinne/workspaces/datasets/speaker_id/extract_data/{spk_id}'
        os.makedirs(extract_path, exist_ok=True)
        shutil.copy(file_path, f'{extract_path}/{filename}')


if __name__=="__main__":
    random.seed(1234) #235

    raw_path = '/home/ujinne/workspaces/datasets/speaker_id/data'
    # noise_path = '/home/ujinne/workspaces/datasets/speaker_id/RIRS_NOISES/pointsource_noises'
    noise_path = '/home/ujinne/workspaces/datasets/speaker_id/Noise'
    save_path = '/home/ujinne/workspaces/datasets/speaker_id/mixed_data'
    # SNR = 2.5 # 20, 25
    n_spk = 100

    print("Extracting Speakers...")
    spks = get_speaker(raw_path)
    extracted_data=[]
    for spk in spks:
        utt_list = extract_files(raw_path, spk)
        if len(utt_list) != 0:
            extracted_data.append(random.choice(utt_list))
        else:
            pass
    print()

    random.shuffle(extracted_data)
    test_files = random.sample(extracted_data, n_spk)
    # 100명의 등록화자
    enrolled_speakers = test_files[:n_spk//2]
    # 100명의 미등록화자
    impostor_speakers = test_files[n_spk//2:]


    # 100명의 등록화자 + 노이즈 음성
    print("Making Noise data...")
    if os.path.exists(f'{save_path}') == True:
        shutil.rmtree(f'{save_path}')
    # print(f'SNR : {SNR}')
    noise_files = glob.glob(f'{noise_path}/*.wav')
    for file_path in tqdm(enrolled_speakers):
        audio, _ = librosa.load(file_path, sr=16000)
        noise, _ = librosa.load(random.choice(noise_files), sr=16000)

        # length nomalization
        while len(noise) - len(audio) < 0:
            noise = np.concatenate([noise, noise])

        # SNR = random.choice([3, 4, 5, 6])
        SNR = 5
        start = random.randint(0, len(noise) - len(audio))
        split_noise = noise[start : start + len(audio)]

        # get noise
        noise = get_noise_from_sound(audio, split_noise, SNR)
        signal_noise = audio + noise
        
        spk_id = file_path.split('/')[-2]
        filename = file_path.split('/')[-1].split('.')[0]
        # mixed_path = f'{save_path}/noise_snr{SNR}/{spk_id}'
        mixed_path = f'{save_path}/{spk_id}'
        os.makedirs(mixed_path, exist_ok=True)
        soundfile.write(f"{mixed_path}/{filename}.wav", signal_noise, 16000) # save mixed file
    print()

    # 100명의 노이즈+등록화자
    # noise_speakers = glob.glob(f'{save_path}/noise_snr{SNR}/**/*.wav')
    noise_speakers = glob.glob(f'{save_path}/**/*.wav')

    # Model load
    # classifier = EncoderClassifier.from_hparams(source="/home/ujinne/workspaces/projects/speaker/results/spkrec-ecapa")
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    
    # Speaker enrollment
    print("Enrolling Speakers...")
    spk_ids, enrolled_embs = enrollment(enrolled_speakers, classifier)
    print()

    # Test with enrolled speakers(+noise) + impostor speakers data
    test_data = noise_speakers + impostor_speakers
    labels = [1 for _ in range(len(noise_speakers))] + [0 for _ in range(len(impostor_speakers))]
    # print(len(test_data))
    logger = get_logger(name=f'Speaker Recognition', file_path=f'voiceprint.log', stream=True)
    # logger = get_logger(name=f'Speaker Recognition', file_path=f'log/speechbrain_{SNR}.log', stream=True)
    
    print("Computing Equal Error Rate...")
    preds = []
    test_embs = []
    for i, utt in enumerate(test_data):
        spk = utt.split('/')[-2]
        wav, _ = torchaudio.load(utt)
        emb = classifier.encode_batch(wav)
        cos_sim = compute_cossim(emb, enrolled_embs)
        max_sim, max_ind = torch.max(cos_sim, dim=-1)
        test_embs.append(emb)

        pred = F.softmax(cos_sim.squeeze(), dim=-1) # prediction score
        preds.append(pred[max_ind].item())

        if max_sim >= 0.5:
        # if max_sim > 0.25:
            logger.info(f'input_speaker: {spk} | similar_speaker: {spk_ids[max_ind]:7s} | label: {labels[i]}   | max_sim: {round(max_sim.item(), 2):.2f} | result: ACCEPT')
            # print(f'input_spk : {spk} |similar_speaker : {spk_ids[max_ind]:7s} |Label : {labels[i]} |max_sim : {round(max_sim.item(), 2):.2f} |ACCEPT')
        else:
            logger.info(f'input_speaker: {spk} | similar_speaker: unknown | label: {labels[i]}   | max_sim: {round(max_sim.item(), 2):.2f} | result: REJECT')
            # print(f'input_spk : {spk} |similar_speaker : unknown |Label : {labels[i]} |max_sim : {round(max_sim.item(), 2):.2f} |REJECT')

    eer = compute_eer(preds, labels)
    # print(f'Equal Error Rate(%) : {eer * 100:.4f}')
    logger.info(f'Equal Error Rate : {eer*100:.4f} (%)')
    logger.info("="*150)

