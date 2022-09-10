import string
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from plate_classifier import *

# alfabeto
alf = [i for i in string.ascii_uppercase]

# números de 0 a 9
nums = [str(i) for i in range(10)]

# index da base completa
alf.extend(nums)

def plot_sidebyside(img_list,
                    titles,
                    colormap=None,
                    figsize=(12,6)):
  n = len(img_list)
  figure, axis = plt.subplots(1, n, figsize=figsize)
  
  for i in range(n):  
    axis[i].imshow(img_list[i], cmap=colormap)
    axis[i].set_title(titles[i])
    axis[i].axis('off')
  plt.show()

def prediction_results(preds, plates_texts):
    df_output = pd.DataFrame()

    df_output['true'] = plates_texts
    df_output['pred'] = preds

    df_output['hits'] = df_output.apply(
        lambda x: sum([1 if a == b else 0 for a, b in zip(x['true'], x['pred'])]),
        axis=1
    )

    df_output['len'] = df_output['true'].map(lambda x: len(x))
    df_output['score'] = df_output['hits'] / df_output['len']

    df_output['equal'] = df_output.apply(
        lambda x: 1 if x['pred'] == x['true'] else 0, axis=1
    )

    return df_output

def summary_results(df_output):
    # contagem de acertos por caractere
    char_hits = {i: 0 for i in alf}

    # contagem total de caracteres
    char_count = {i: 0 for i in alf}

    for i in df_output.iterrows():
        row = i[1]
        
        for ct, cp in zip(row['true'], row['pred']):
            if ct == cp:
                char_hits[ct] += 1
            char_count[ct] += 1

    # acertos de cada caractere em relação ao total
    hits_of_total = {key: str(hit) + ' of ' + str(count) \
                    for key, hit, count in zip(alf, list(char_hits.values()),
                                                    list(char_count.values()))}

    # acurácia por caractere
    # a letra 'Z' está como 'np.nan' pois não ocorreu
    # nenhuma vez nas placas

    accuracies = np.array(list(char_hits.values())) \
                / np.array(list(char_count.values()))

    char_acc = {key: acc for key, acc in zip(alf, accuracies)}

    # exibindo os resultados
    df_acc = pd.DataFrame(index=alf)
    df_acc['hits of total'] = list(hits_of_total.values())
    df_acc['accuracy'] = accuracies
    return df_acc

def summary_plot(df_acc, palette='Set2'):
    # gráfico de barras da acurácia por caractere

    x = df_acc['accuracy'].sort_values(ascending=False).index
    y = df_acc['accuracy'].sort_values(ascending=False).values

    plt.figure(figsize=(12,6));
    sns.barplot(x, y, palette=palette);
    plt.xticks(fontsize=14);
    plt.title('Acurácias por Caractere', fontsize=14);
    plt.ylabel('Acurácia', fontsize=14);
    plt.show();

def get_image_stats(c,
                    return_as='list',
                    feature_names=None,
                    templates=None):
    stats = []
    stats.append(len(c[c == 0]))
    stats.append(len(c[c == 0])/len(c.ravel()))
    stats.append(len(c[c == 255]))
    stats.append(len(c[c == 255])/len(c.ravel()))
    stats.append(max([len(i) for i in [k[k == 0] for k in c]]))
    stats.append(np.argmax(([len(i) for i in [k[k == 0] for k in c]])))
    stats.append(min([len(i) for i in [k[k == 0] for k in c]]))
    stats.append(np.argmin(([len(i) for i in [k[k == 0] for k in c]])))
    stats.append(max([len(i) for i in [k[k == 0] for k in c.T]]))
    stats.append(np.argmax(([len(i) for i in [k[k == 0] for k in c.T]])))
    stats.append(min([len(i) for i in [k[k == 0] for k in c.T]]))
    stats.append(np.argmin(([len(i) for i in [k[k == 0] for k in c.T]])))
    stats.append(len(np.diag(c)[np.diag(c) == 0]))
    stats.append(len(np.diag(np.fliplr(c))[np.diag(np.fliplr(c)) == 0]))
    stats.append(len(c[len(c)//2][c[len(c)//2] == 0]))
    stats.append(len(c[len(c)//2][c[len(c)//2] == 255]))
    stats.append(len(c.T[len(c.T)//2][c.T[len(c.T)//2] == 0]))
    stats.append(len(c.T[len(c.T)//2][c.T[len(c.T)//2] == 255]))

    stats.append(c.ravel().mean())
    stats.append(c.ravel().std())
    stats.append(len(c[0:len(c)//2, 0:len(c)//2][c[0:len(c)//2, 0:len(c)//2] == 0]))
    stats.append(len(c[0:len(c)//2, len(c)//2:len(c)][c[0:len(c)//2, len(c)//2:len(c)] == 0]))
    stats.append(len(c[len(c)//2:len(c), 0:len(c)//2][c[len(c)//2:len(c), 0:len(c)//2] == 0]))
    stats.append(len(c[len(c)//2:len(c), len(c)//2:len(c)][c[len(c)//2:len(c), len(c)//2:len(c)] == 0]))
    stats.append(len(c[0:len(c)//2, 0:len(c)//2][c[0:len(c)//2, 0:len(c)//2] == 255]))
    stats.append(len(c[0:len(c)//2, len(c)//2:len(c)][c[0:len(c)//2, len(c)//2:len(c)] == 255]))
    stats.append(len(c[len(c)//2:len(c), 0:len(c)//2][c[len(c)//2:len(c), 0:len(c)//2] == 255]))
    stats.append(len(c[len(c)//2:len(c), len(c)//2:len(c)][c[len(c)//2:len(c), len(c)//2:len(c)] == 255]))
    stats.append(len(c[:, 0:len(c)//2][c[:, 0:len(c)//2] == 0]))
    stats.append(len(c[:, len(c)//2:len(c)][c[:, len(c)//2:len(c)] == 0]))
    stats.append(len(c[0:len(c)//2, :][c[0:len(c)//2, :] == 0]))
    stats.append(len(c[len(c)//2:len(c), :][c[len(c)//2:len(c), :] == 0]))
    stats.append(len(c[:, 0:len(c)//2][c[:, 0:len(c)//2] == 255]))
    stats.append(len(c[:, len(c)//2:len(c)][c[:, len(c)//2:len(c)] == 255]))
    stats.append(len(c[0:len(c)//2, :][c[0:len(c)//2, :] == 255]))
    stats.append(len(c[len(c)//2:len(c), :][c[len(c)//2:len(c), :] == 255]))
    stats.append(np.array([len(row[row == 0]) for row in c]).mean())
    stats.append(np.array([len(row[row == 0]) for row in c.T]).mean())

    stats.append(max([len((c-temp)[(c-temp) == 0]) for temp in templates]))
    stats.append(np.argmax([len((c-temp)[(c-temp) == 0]) for temp in templates]))
    stats.append(max([len((temp-c)[(temp-c) == 0]) for temp in templates]))
    stats.append(np.argmax([len((temp-c)[(temp-c) == 0]) for temp in templates]))

    stats.append(max([len((c-temp)[(c-temp) == 255]) for temp in templates]))
    stats.append(np.argmax([len((c-temp)[(c-temp) == 255]) for temp in templates]))
    stats.append(max([len((temp-c)[(temp-c) == 255]) for temp in templates]))
    stats.append(np.argmax([len((temp-c)[(temp-c) == 255]) for temp in templates]))

    stats.append(max([len((c-temp)[(c-temp) == -255]) for temp in templates]))
    stats.append(np.argmax([len((c-temp)[(c-temp) == -255]) for temp in templates]))
    stats.append(max([len((temp-c)[(temp-c) == -255]) for temp in templates]))
    stats.append(np.argmax([len((temp-c)[(temp-c) == -255]) for temp in templates]))

    if return_as == 'list':
        return stats
    elif return_as == 'json':
        return {feature_names[i]: stats[i] for i in range(len(stats))}