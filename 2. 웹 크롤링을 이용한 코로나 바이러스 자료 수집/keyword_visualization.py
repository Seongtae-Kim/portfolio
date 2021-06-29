import matplotlib.font_manager as fm
from matplotlib.pyplot import figure
from gensim.models import Word2Vec
from gensim.models import FastText
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from lxml import etree as ET
from konlpy.tag import Mecab
from lxml.builder import E
import matplotlib as mpl
import networkx as nx
import pandas as pd
tagger = Mecab()

# Seongtae Kim 2020-06-14
# https://github.com/Seongtae-Kim/WordEmbeddingVisualization


def load_model(path="./", model_type="w2v"):
    if model_type == "w2v":
        return Word2Vec.load(path)
    else:
        return FastText.load(path)

def keyword_visualization(model, keyword, num_nodes=10, font_size=17):
    words = []
    vectors = []
    edges = []
    sizes = [ 5000 ] # center word size: 5000

    w2vs = model.wv.most_similar(positive=keyword, topn=num_nodes)
    
    i=1
    for w2v in w2vs:
        word = str(i) + "\n" + w2v[0] #+ "\n"+ str(round(w2v[1], 3))
        rank = i
        words.append(word)
        vectors.append(w2v[1])
        edges.append((keyword, word, rank))
        i += 1
    G = nx.Graph()
    G.add_node(keyword, weight=1, rank=0)
    
    i=1
    for word, vector in zip(words, vectors):
        G.add_nodes_from(words, weight=vector)
        
        if int(i/10) in list(range(1, 10)):
            size = 5000 * round(vector, 2)
        else: # The first ten sequence
            size = 5000 * round(vector, 2)
        sizes.append(size)
        i+=1
    for (key, word, rank) in edges:
        G.add_edge(key, word, length=rank)

   #pos = nx.spring_layout()
    pos = nx.spiral_layout(G, resolution=1, equidistant=True) 
    colors= range(num_nodes)
    #print(G.nodes(data="weight"))
    plt.figure(figsize=(16,12))
    plt.axis('off')
    path_nanum = "NanumMyeongjo-Bold.ttf"
    prop = fm.FontProperties(fname=path_nanum)
    plt.title(keyword + "_" + str(num_nodes) + "_" + str(model), fontproperties=prop, fontsize=30)
    nx.draw(G, pos, node_color="yellow", node_size=sizes, width=4, with_labels=True,
           font_family='NanumGothic', font_size=font_size)
    plt.savefig("./vis/" + keyword + "_" + str(num_nodes) + "_" + str(model) + ".png", format="PNG")