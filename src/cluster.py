import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from data import LoadDossierEmbeddings

N_CLUSTERS = 50


def main():
    embeddings = LoadDossierEmbeddings(Path('data') / 'json' / 'bundestag', 'gte', limit=-1)
    question_embeddings = []
    urls = []
    for path, i, qa, question_embedding, answer_embedding in tqdm(embeddings, desc='loading data'):
        if question_embedding is not None:
            question_embeddings.append(question_embedding)
            urls.append(qa.url)
    question_embeddings = np.array(question_embeddings)

    print('running clustering')
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=10000, verbose=True)
    kmeans.fit(question_embeddings)

    prediction = kmeans.predict(question_embeddings)
    url_to_cluster = {url: int(p) for url, p in zip(urls, prediction)}
    with open(f'data/embeddings/cluster/bundestag/cluster{N_CLUSTERS}.json', 'w') as f:
        json.dump(url_to_cluster, f, indent=2)


if __name__ == '__main__':
    main()
