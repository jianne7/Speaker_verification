import glob
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torchaudio

from speechbrain.pretrained import EncoderClassifier

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def get_speaker(path, num_spks: int):
    files = glob.glob(f'{path}/*')
    spk_id_list = []
    for f in tqdm(files):
        spk_id = f.split('/')[-1]
        spk_id_list.append(spk_id)
    spk_ids = random.sample(spk_id_list, num_spks)

    return spk_ids

def get_utterance(path, spk_id, num_utt):
    utt_list = glob.glob(f'{path}/{spk_id}/*.wav')
    utts = random.sample(utt_list, num_utt)
        
    return utts

def sampling_data(path, num_spks: int, num_utt):
    spk_ids = get_speaker(path, num_spks)
    spks = []
    utts = []
    for spk in spk_ids:
        utt_list = get_utterance(path, spk, num_utt)
        utts += utt_list
        spks += [spk] * num_utt
    # utts = [get_utterance(path, spk_id, num_utt) for spk_id in spk_ids]

    return spks, utts

def compute_cossim(input_emb, emb_matirx):
    '''
    input_emb.size() = (1, 1, 192)
    emb_matrix.size() = (1, num_spks, 192)
    '''
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    similarity = cosine_sim(input_emb, emb_matirx) # similarity.size()=(1, 100)

    return similarity

if __name__=="__main__":
    path = '/home/ujinne/workspaces/datasets/speaker_id/vp-audio'

    # classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    # classifier = EncoderClassifier.from_hparams(source="/home/ujinne/workspaces/projects/speaker/results/spkrec-ecapa")
    num_spks = 6 # the number of speakers per mini-batch
    num_utt = 2 # the number of utterences per speaker

    spks, utts = sampling_data(path, num_spks, num_utt)

    embeddings = []
    for utt in utts:
        wav, _ = torchaudio.load(utt)
        emb = classifier.encode_batch(wav)
        embeddings.append(emb)

    emb1 = torch.cat(embeddings, dim=0) # 20, 1, 192
    emb2 = torch.cat(embeddings, dim=1) # 1, 20, 192
    cos_sim = compute_cossim(emb1, emb2)

    plt.figure(figsize=(10,8))
    sns.heatmap(cos_sim, 
                annot=True,
                annot_kws={'fontsize' : 7},
                cmap="YlGnBu", # YlGnBu, Spectral, YlOrRd
                linewidths=.5,
                vmin=0, 
                vmax=0.5) 

    plt.xticks(np.arange(0.5, len(spks), 1), spks, fontsize=6)
    plt.yticks(np.arange(0.5, len(spks), 1), spks, fontsize=6)

    plt.ylabel('Speaker ID')
    # plt.xlabel('Speaker ID')
    plt.savefig(f'cos_matrix.png', format='png')
    print("="*50, "SAVE THE IMAGE FILE", "="*50)

    # # T-SNE
    # embeddings = np.array(torch.stack(embeddings).squeeze(1)) # embeddings.shape=(40, 192)
    # tsne = TSNE(n_components=2, verbose=1) #, random_state=785)
    # transformed = tsne.fit_transform()
    # print(f'tsne shape:{transformed.shape}')

    # data = {
    #         "dim_X": transformed[:, 0],
    #         "dim_Y": transformed[:, 1],
    #         "label": spk_ids
    #         }

    # plt.figure(figsize=(15,12))
    # sns.scatterplot(x="dim_X",
    #                 y="dim_Y",
    #                 hue="label",
    #                 # palette=sns.color_palette(n_colors=20),
    #                 palette='pastel',
    #                 data=data,
    #                 legend="full",
    #                 s=200)

    # # spk_ids 표시
    # for i in range(len(transformed)):
    #     plt.text(x=transformed[:,0][i], 
    #              y=transformed[:,1][i], 
    #              s=spk_ids[i], fontsize=8, 
    #              horizontalalignment='left',
    #              verticalalignment='center')

    # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # plt.tight_layout()
    # plt.savefig(f'tsne.png', format='png')