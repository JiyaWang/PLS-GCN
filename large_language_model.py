import pandas as pd
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import dot_score
import torch

def embeddings(sentences):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    query_embeddings = model.encode(sentences)
    return query_embeddings


def similarity(target, query):
    return dot_score(target, query)


if __name__ == '__main__':
    with open('./data/graph_data+node_mapping(select_label_4).pkl', 'rb') as f:
        data = pickle.load(f)
        node_mapping = pickle.load(f)

    df = pd.read_csv("./data/nodeidx2paperid.csv", header=0)
    paper_mapping = dict(zip(df.values[:, 0].tolist(), df.values[:, [1]]))

    titleabs = pd.read_table("./data/titleabs.tsv", header=None)
    title_mapping = dict(zip(tuple(titleabs.values[:, 0].tolist()), tuple(titleabs.values[:, [1, 2]].tolist())))

    nodes = list(node_mapping.values())
    paper = [paper_mapping[key] for key in nodes]

    title_abstract = []
    for paper_id in paper:
        title_abstract.append('《{}》,{}'.format(title_mapping[paper_id.item()][0], title_mapping[paper_id.item()][1]))


    sentences_embeddings=embeddings(title_abstract)
    result = torch.tensor(sentences_embeddings) @ torch.tensor(sentences_embeddings).T
    target_covariance = []
    for target in tqdm(sentences_embeddings):
        row = similarity(target, sentences_embeddings).squeeze().tolist()
        target_covariance.append(row)

    with open('./data/similarity_matrix_raw3.pkl', 'wb') as f:
        pickle.dump(result, f)
