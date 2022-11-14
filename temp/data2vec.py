import glob
import random
from tqdm import tqdm
import numpy as np
import librosa

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn

from transformers import Wav2Vec2FeatureExtractor, Data2VecAudioForXVector, Wav2Vec2ForXVector

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, MiniBatchKMeans

from warnings import filterwarnings
filterwarnings("ignore")
torch.cuda.empty_cache()

N = 6 # the number of speakers per mini-batch
M = 2 # the number of utterences per speaker

CUDA_VISIBLE_DEVICES=1

def index2char(speaker_id: list):
    txt2id = {f'{txt}':i for i, txt in enumerate(speaker_id)}
    # txt2id['unknown'] = len(txt2id)+1
    id2txt = {int(value):key for key, value in txt2id.items()}

    return txt2id, id2txt

class AudioDataset(Dataset):
    def __init__(self, root):
        
        self.utts_per_speaker = M
        self.root_path = root
        
        files = glob.glob('{}/*'.format(self.root_path))
        self.spk_id_list = []
        
        for f in tqdm(files):
            spk_id = f.split('/')[-1]
            self.spk_id_list.append(spk_id)

        self.txt2id, _ = index2char(self.spk_id_list)
        
    def __len__(self):
        return len(self.spk_id_list)
    
    def __getitem__(self, idx: int):
        
        spk_id = self.spk_id_list[idx]
        
        # utts = glob.glob('{}/{}/*.pt'.format(self.root_path, spk_id))
        utts = glob.glob('{}/{}/*.wav'.format(self.root_path, spk_id))
        # utt_list = random.sample(utts, self.utts_per_speaker)
        # utt = random.choice(utts)
        if len(utts) >= self.utts_per_speaker:
            utt_list = random.sample(utts, self.utts_per_speaker)
        else:
            utt_list = random.choices(utts, k=self.utts_per_speaker)
        
        wav_list = []
        ids_list = []
        for utt in utt_list:
            wav, _ = librosa.load(utt)
            # print(f'wav:{wav.shape}')
            # ids = int(self.txt2id.get(spk_id))
            ids  = spk_id
            wav_list.append(wav)
            ids_list.append(ids)
        
        return wav_list, ids_list

def audio_collate_fn(batch):
    audio = []
    spk_ids = []
    for wav, ids in batch:
        audio += wav
        spk_ids += ids

    return audio, spk_ids

def main():
    path = '/home/ujinne/workspaces/datasets/speaker_id/vp-audio'
    dataset = AudioDataset(path)
    # print(len(dataset), dataset[0])
    dataloader = DataLoader(dataset,
                            shuffle=True,
                            batch_size=N,
                            collate_fn=audio_collate_fn,
                            )

    # feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("hf-internal-testing/tiny-random-data2vec-xvector")
    # model = Data2VecAudioForXVector.from_pretrained("hf-internal-testing/tiny-random-data2vec-xvector")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sv") # anton-l/wav2vec2-base-superb-sv
    model = Wav2Vec2ForXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv") # facebook/wav2vec2-base-960h

    for audio, spk_ids in dataloader:
        # inputs = feature_extractor([aud for aud in audio], sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        del audio
        break

    with torch.no_grad():
        embeddings = model(**inputs).embeddings
        # print(f'embeddings:{embeddings}')
        del inputs
        # embeddings = np.array(torch.nn.functional.normalize(embeddings, dim=-1))
        print(f'shape : {embeddings.shape}')

    # embeddings = np.array(embeddings)
    # print(f'shape : {embeddings.shape}')

    # the resulting embeddings can be used for cosine similarity-based retrieval
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    cos_matrix = []
    for i in range(len(embeddings)):
        similarity = []
        for j in range(len(embeddings)):
            similarity.append(round(cosine_sim(embeddings[i], embeddings[j]).item(), 2))
        cos_matrix.append(similarity)

    plt.figure(figsize=(10,8))
    sns.heatmap(cos_matrix, 
                annot=True,
                annot_kws={'fontsize' : 7},
                cmap="YlGnBu", # YlGnBu, Spectral, YlOrRd
                linewidths=.5,
                vmin=0.5, 
                vmax=1) 

    plt.xticks(np.arange(0.5, len(spk_ids), 1), spk_ids, fontsize=6)
    plt.yticks(np.arange(0.5, len(spk_ids), 1), spk_ids, fontsize=6)

    plt.ylabel('Speaker ID')
    # plt.xlabel('Speaker ID')
    plt.savefig(f'cos_matrix.png', format='png')
    print("="*50, "SAVE THE IMAGE FILE", "="*50)

    # tsne = TSNE(n_components=2, perplexity=10) #random_state=785)
    # transformed = tsne.fit_transform(np.array(embeddings))
    # print(f'tsne shape:{transformed.shape}')

    # data = {
    #         "dim_X": transformed[:, 0],
    #         "dim_Y": transformed[:, 1],
    #         "label": spk_ids #dataset['speaker_id'] #spk_ids,
    #         }
    # # print(f'data type:{type(data)}')

    # plt.figure(figsize=(13, 9))
    # sns.scatterplot(
    #                 x="dim_X",
    #                 y="dim_Y",
    #                 hue="label",
    #                 # palette=sns.color_palette(n_colors=20),
    #                 palette='pastel',
    #                 data=data,
    #                 legend="full",
    #                 s=200
    #             )

    # # spk_ids 표시
    # for i in range(len(transformed)):
    #     plt.text(x=transformed[:,0][i], 
    #                 y=transformed[:,1][i], 
    #                 s=spk_ids[i], fontsize=8, 
    #                 horizontalalignment='left',
    #                 verticalalignment='center')

    # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # plt.tight_layout()
    # plt.savefig(f'tsne.png', format='png')
    

if __name__=="__main__":
    main()