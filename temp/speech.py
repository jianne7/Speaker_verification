import os
import glob
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F

from speechbrain.pretrained import EncoderClassifier

# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.manifold import TSNE

from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# random.seed(235)
num_spks = 10 # the number of speakers per mini-batch
M = 2 # the number of utterences per speaker

def enrolled_speaker(path, num_spks):
    files = [f for f in glob.glob(f'{path}/*') if os.path.isdir(f)]
    # files.remove(f"{path}/RIRS_NOISES")
    spk_id_list = []
    for f in tqdm(files):
        spk_id = f.split('/')[-1]
        spk_id_list.append(spk_id)
    spk_ids = spk_id_list[:num_spks]
    spk_ids = random.sample(spk_ids, int(num_spks//2))

    utt_list = []
    for spk_id in spk_ids:
        if len(glob.glob(f'{path}/{spk_id}/*.wav')) != 0:
            utts = glob.glob(f'{path}/{spk_id}/*.wav')
        else:
            id_list = glob.glob(f'{path}/{spk_id}/*')
            utts = glob.glob(f'{random.choice(id_list)}/*.wav')   
        utt_list.append(utts[-1])
    return spk_ids, utt_list

def impostor(path, num_spks):
    files = [f for f in glob.glob(f'{path}/*') if os.path.isdir(f)]
    # files.remove(f"{path}/RIRS_NOISES")
    spk_id_list = []
    for f in tqdm(files):
        spk_id = f.split('/')[-1]
        spk_id_list.append(spk_id)
    spk_ids = spk_id_list[num_spks:]
    spk_ids = random.sample(spk_ids, int(num_spks//2))

    utt_list = []
    for spk_id in spk_ids:
        if len(glob.glob(f'{path}/{spk_id}/*.wav')) != 0:
            utts = glob.glob(f'{path}/{spk_id}/*.wav')
        else:
            id_list = glob.glob(f'{path}/{spk_id}/*')
            utts = glob.glob(f'{random.choice(id_list)}/*.wav')   
        utt_list.append(utts[0])
    return spk_ids, utt_list

def enrollment(path, num_spks, model):
    files = [f for f in glob.glob(f'{path}/*') if os.path.isdir(f)]
    # files.remove(f"{path}/RIRS_NOISES")
    spk_id_list = []
    for f in tqdm(files):
        spk_id = f.split('/')[-1]
        spk_id_list.append(spk_id)
    spk_ids = spk_id_list[:num_spks]

    utt_list = []
    for spk_id in spk_ids:
        if len(glob.glob(f'{path}/{spk_id}/*.wav')) != 0:
            utts = glob.glob(f'{path}/{spk_id}/*.wav')
        else:
            id_list = glob.glob(f'{path}/{spk_id}/*')
            utts = glob.glob(f'{random.choice(id_list)}/*.wav')   
        utt_list.append(utts[0])

    embeddings = []
    for utt in utt_list:
        wav, _ = torchaudio.load(utt) # .wav, .flac 가능
        emb = model.encode_batch(wav) # emb.size()=(1, 1, 192)
        # print(f'emb:{emb.shape}')
        embeddings.append(emb)

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

if __name__=="__main__":
    path = '/home/ujinne/workspaces/datasets/speaker_id/test_data'

    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    # classifier = EncoderClassifier.from_hparams(source="/home/ujinne/workspaces/projects/speaker/results/spkrec-ecapa")
    enrolled_spks, enrolled_embs = enrollment(path, num_spks, classifier)

    enrol_spks, enrol_utts = enrolled_speaker(path, num_spks)
    impostor_spks, impostor_utts = impostor(path, num_spks)
    labels = [1 for _ in range(int(num_spks//2))] + [0 for _ in range(int(num_spks//2))]
    spks = enrol_spks + impostor_spks
    utts = enrol_utts + impostor_utts
    spks, utts, labels = shuffle(spks, utts, labels)

    preds = []
    test_embs = []
    for i, (spk, utt) in enumerate(zip(spks, utts)):
        wav, _ = torchaudio.load(utt)
        emb = classifier.encode_batch(wav)
        cos_sim = compute_cossim(emb, enrolled_embs)
        max_sim, max_ind = torch.max(cos_sim, dim=-1)
        test_embs.append(emb)

        pred = F.softmax(cos_sim.squeeze(), dim=-1) # prediction score
        preds.append(pred[max_ind].item())

        if max_sim >= 0.5:
        # if max_sim > 0.25:
            print(f'input_spk : {spk}    |similar_speaker : {enrolled_spks[max_ind]}    |Enrollment : {labels[i]}    |max_sim : {round(max_sim.item(), 2)}    |ACCEPT')
        else:
            print(f'input_spk : {spk}    |similar_speaker : {enrolled_spks[max_ind]}    |Enrollment : {labels[i]}    |max_sim : {round(max_sim.item(), 2)}    |REJECT')

    eer = compute_eer(preds, labels)
    print(f'Equal Error Rate(%) : {eer * 100:.4f}')

    # # emb1 = torch.cat(embeddings, dim=0) # 20, 1, 192
    # # emb2 = torch.cat(embeddings, dim=1) # 1, 20, 192
    # # cos_sim = compute_cossim(emb1, emb2)

    # # plt.figure(figsize=(30,25))
    # # sns.heatmap(cos_sim, 
    # #             annot=True,
    # #             annot_kws={'fontsize' : 9},
    # #             fmt='.2f',
    # #             cmap="YlOrRd", # YlGnBu, Spectral, YlOrRd
    # #             linewidths=.5,
    # #             vmin=0, 
    # #             vmax=0.5) 

    # # plt.xticks(np.arange(0.5, len(spks), 1), spks, fontsize=9, rotation=90)
    # # plt.yticks(np.arange(0.5, len(spks), 1), spks, fontsize=9, rotation=0)

    # # plt.ylabel('Speaker ID')
    # # # plt.xlabel('Speaker ID')
    # # plt.savefig(f'cos_matrix.png', format='png')

    # # T-SNE
    # # print(f'enrolled_embs : {enrolled_embs.shape}')
    # # embeddings = np.array(torch.stack(enrolled_embs.squeeze(0))) # embeddings.shape=(40, 192)
    # tsne = TSNE(n_components=2, verbose=1) #, random_state=785)
    # transformed = tsne.fit_transform(np.array(enrolled_embs.squeeze(0)))
    # print(f'tsne shape:{transformed.shape}')

    # data = {
    #         "dim_X": transformed[:, 0],
    #         "dim_Y": transformed[:, 1],
    #         "label": enrolled_spks
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
    #              s=enrolled_spks[i], fontsize=8, 
    #              horizontalalignment='left',
    #              verticalalignment='center')

    # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # plt.tight_layout()
    # plt.savefig(f'tsne.png', format='png')