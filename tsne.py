import argparse
import os
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd 
import json
import pdb
import torch.nn.functional as F
from einops import rearrange

def plot_TNSE(s, X_2d_tr, tr_labels, label_target_names, filename):
    colors = ["red", "green", "blue", "black", "brown", "grey", "orange", "yellow", "pink", "cyan", "magenta"]
    plt.figure(figsize=(5, 5))
    #import pdb; pdb.set_trace()

    for i, label in zip(range(len(label_target_names[:-2])), label_target_names[:-2]):
        #plt.scatter(X_2d_tr[tr_labels == i, 0], X_2d_tr[tr_labels == i, 1], c=colors[i], marker=".", label=label)
        plt.scatter(X_2d_tr[tr_labels == i, 0], X_2d_tr[tr_labels == i, 1], marker=".", label=label,s=s)


    length = len(label_target_names)
    plt.scatter(X_2d_tr[tr_labels == length-2, 0], X_2d_tr[tr_labels == length-2, 1], c='black', marker=".", label=length-2,s=s)
    plt.scatter(X_2d_tr[tr_labels == length-1, 0], X_2d_tr[tr_labels == length-1, 1], c='yellow', marker=".", label=length-1,s=s)

    plt.savefig(filename)

def plot_TNSE_noT(s, X_2d_tr, tr_labels, label_target_names, filename):
    colors = ["red", "green", "blue", "black", "brown", "grey", "orange", "yellow", "pink", "cyan", "magenta"]
    plt.figure(figsize=(5, 5))
    #import pdb; pdb.set_trace()

    for i, label in zip(range(len(label_target_names[:-2])), label_target_names[:-2]):
        #plt.scatter(X_2d_tr[tr_labels == i, 0], X_2d_tr[tr_labels == i, 1], c=colors[i], marker=".", label=label)
        plt.scatter(X_2d_tr[tr_labels == i, 0], X_2d_tr[tr_labels == i, 1], marker=".", label=label, s=s)


    # length = len(label_target_names)
    # plt.scatter(X_2d_tr[tr_labels == length-2, 0], X_2d_tr[tr_labels == length-2, 1], c='black', marker=".", label=length-2)
    # plt.scatter(X_2d_tr[tr_labels == length-1, 0], X_2d_tr[tr_labels == length-1, 1], c='yellow', marker=".", label=length-1)

    plt.savefig(filename)

def plot_TNSE3(X_2d_tr, tr_labels, label_target_names, filename):
    colors = ["red", "green", "blue", "black", "brown", "grey", "orange", "yellow", "pink", "cyan", "magenta"]
    plt.figure(figsize=(16, 16)).add_subplot(projection='3d')

    for i, label in zip(range(len(label_target_names[:-2])), label_target_names[:-2]):
        #plt.scatter(X_2d_tr[tr_labels == i, 0], X_2d_tr[tr_labels == i, 1], c=colors[i], marker=".", label=label)
        plt.scatter(X_2d_tr[tr_labels == i, 0], X_2d_tr[tr_labels == i, 1], X_2d_tr[tr_labels == i, 2], marker=".", label=label)


    length = len(label_target_names)
    plt.scatter(X_2d_tr[tr_labels == length-2, 0], X_2d_tr[tr_labels == length-2, 1], X_2d_tr[tr_labels == length-2, 2], c='black', marker=".", label=length-2)
    plt.scatter(X_2d_tr[tr_labels == length-1, 0], X_2d_tr[tr_labels == length-1, 1], X_2d_tr[tr_labels == length-1, 2], c='yellow', marker=".", label=length-1)

    plt.savefig(filename)




def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

from numpy.random import default_rng


def cosine_similarity( features, prototype_feature):
    normed_features = F.normalize(features, dim=1)
    normed_prototype = F.normalize(prototype_feature, dim=1)
    pair_wises = torch.einsum('bd,dn->bn', normed_features, rearrange(normed_prototype, 'n d -> d n'))
    # label_matrix = (domain_class_labels.unsqueeze(1) == self.prototype_net.labels.unsqueeze(0)).int()
    # pair_wises[label_matrix==0] = -1.0
    similarity = 1 - pair_wises
    return similarity
# METRIC for WS ----------------------------------------------
    


def tsne_compress(X):
    tsne_model = TSNE(n_components=2, init="pca")
    featn = tsne_model.fit_transform(X)
    return featn

def plot(x, labels, markers=None, colors=None, sizes=None, edgecolors=None, save_files=None, title=''):
    
    # if markers is None:
    #     markers = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    # Create a scatter plot.
    plt.figure(figsize=(10, 10))
    # for data, label, mk, color in zip(x, labels, markers, colors):
    unique_labels = np.unique(labels)
    if colors is None:
        colors = labels
    if edgecolors is None:
        edgecolors = labels

    palette = np.random.uniform(0, 1, (max(max(edgecolors).astype(np.int), 5) + 3, 3))
    np.random.shuffle(palette)
    palette = np.append(palette, [[1,1,1]], axis=0)
    # print(palette)

    for cls_c in unique_labels:
        x_c = x[labels==cls_c]
        if markers is None:
            mk = '.'
        else:
            mk = markers[labels==cls_c][0]
        color = colors[labels==cls_c]
        ed_col = edgecolors[labels==cls_c]
        # sizes
        plt.scatter(x_c[:,0], x_c[:,1], s=100, label=cls_c, c=palette[color.astype(np.int)], marker=mk, edgecolors=palette[ed_col.astype(np.int)])
    # plt.legend()
    plt.title(title)
    if save_files is not None:
        plt.savefig(save_files)

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--plotdir", help="Path to configuration file")
    bash_args = parser.parse_args()
    dir_name = bash_args.plotdir

    file_list = []
    name_list = []
    for root, dirs, files in os.walk(dir_name):
        for f_n in files:
            if ".pkl" in f_n:
                file_list.append(os.path.join(root, f_n))
                name_list.append(f_n)

    data_dict = {}

    for file_link, file_name in zip(file_list, name_list):
        # pdb.set_trace()
        with open(file_link, "rb") as fp:
            data_loaded = np.asarray(pickle.load(fp))
            data_dict[file_name] = data_loaded
    print(data_dict.keys())
    

    # import pdb; pdb.set_trace()
    
    domains = ['Art', 'Clipart', 'Product', 'RealWorld']
    for domain in domains:
        feature = data_dict["{}_feature_train.pkl".format(domain)] 
        label =  data_dict["{}_Y_train.pkl".format(domain)] 
        len_data = len(feature)
        
        # all_feature = all_feature/np.linalg.norm(all_feature, axis=-1)[..., None]
        compress = tsne_compress(feature)
        class_unique = np.unique(label)
        
        end_f = len(label)
        max_label = class_unique.max()
        markers = np.asarray(['.']*len_data)

        plot(compress, label, colors=None, markers=markers, title="Label with class", 
        save_files=os.path.join(bash_args.plotdir, "{}_tsne_class_label.png".format(domain)))
 