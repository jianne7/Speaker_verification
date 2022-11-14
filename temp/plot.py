import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

embeddings = np.array(torch.stack(embeddings).squeeze(1)) # embeddings.shape=(40, 192)
tsne = TSNE(n_components=2, verbose=1) #, random_state=785)
transformed = tsne.fit_transform(embeddings)
print(f'tsne shape:{transformed.shape}')

data = {
        "dim_X": transformed[:, 0],
        "dim_Y": transformed[:, 1],
        "label": spk_ids
        }

# plt.figure(figsize=(15,12))
plt.figure()
sns.scatterplot(x="dim_X",
                y="dim_Y",
                hue="label",
                # palette=sns.color_palette(n_colors=20),
                palette='pastel',
                data=data,
                legend="full",
                s=200)

# spk_ids 표시
for i in range(len(transformed)):
    plt.text(x=transformed[:,0][i], 
                y=transformed[:,1][i], 
                s=spk_ids[i], fontsize=8, 
                horizontalalignment='left',
                verticalalignment='center')

plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(f'tsne.png', format='png')