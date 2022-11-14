import os
import glob
import shutil
import random
from tqdm import tqdm
import wave

def clean_text(label_path):
    with open(label_path, 'r') as file:
        labels = file.readlines()
        for label in tqdm(labels):
            path = label.split('\t')[0].split('.')[0]+'_clean.txt'
            text = label.split('\t')[1]
            with open(f'/home/ujinne/workspaces/datasets/speaker_id/KsponSpeech/{path}', 'w') as f:
                f.write(text)

def pcm2wav(pcm_path, save_path):
    with open(pcm_path, 'rb') as pcmfile:
        pcmdata = pcmfile.read()
    with wave.open(save_path+'.wav', 'wb') as wavfile:
        wavfile.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
        wavfile.writeframes(pcmdata)

def extract_files(raw_path, save_path):
    pcm_files = sorted(glob.glob(f'{raw_path}/**/*.pcm'))
    extraced_pcm = random.sample(pcm_files, 100)

    if os.path.exists(f'{save_path}') == True:
        shutil.rmtree(f'{save_path}')

    for pcm_path in tqdm(extraced_pcm):
        filename = pcm_path.split('/')[-1].split('.')[0]
        txt_path = f'{pcm_path.split(".")[0]}_clean.txt'
        if not os.path.exists(save_path):
            os.mkdir(save_path)  
        shutil.copy(txt_path, f'{save_path}/{filename}.txt')
        pcm2wav(pcm_path, f'{save_path}/{filename}')


if __name__=="__main__":
    random.seed(1234) #235
    
    label_path = '/home/ujinne/workspaces/projects/speaker/ksponspeech/data/transcripts.txt'
    raw_path = '/home/ujinne/workspaces/datasets/speaker_id/KsponSpeech/KsponSpeech_01'
    save_path = '/home/ujinne/workspaces/datasets/speaker_id/asr_extract_data'

    clean_text(label_path)
    extract_files(raw_path, save_path)