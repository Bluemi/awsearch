from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from data import LoadDossierEmbeddings

N_CLUSTERS = 25


def main():
    embeddings = LoadDossierEmbeddings(Path('data') / 'json' / 'bundestag', 'gte', limit=-1)
    question_embeddings = []
    urls = []
    for qa, question_embedding, answer_embedding in embeddings:
        if question_embedding is not None:
            question_embeddings.append(question_embedding)
            urls.append(qa.url)
    question_embeddings = np.array(question_embeddings)

    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=10000, verbose=False)
    kmeans.fit(question_embeddings)

    clusters = defaultdict(list)
    prediction = kmeans.predict(question_embeddings)
    for p, url in zip(prediction, urls):
        clusters[p].append(url)

    for c_index, cluster in enumerate(clusters.values()):
        print(f'cluster {c_index}:')
        for i, url in enumerate(cluster):
            print('  ', url, sep='')
            if i == 20:
                break


if __name__ == '__main__':
    main()
