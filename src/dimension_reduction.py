from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE
# from cuml import TSNE

from src.data import LoadDossierEmbeddings


def create_embeddings():
    embeddings = LoadDossierEmbeddings(Path('data') / 'json' / 'bundestag', 'gte', limit=-1)
    question_embeddings = []
    urls = []
    for qa, question_embedding, answer_embedding in embeddings:
        if question_embedding is not None:
            question_embeddings.append(question_embedding)
            urls.append(qa.url)
    question_embeddings = np.array(question_embeddings)

    tsne = TSNE(n_components=2)
    embeddings_2d = tsne.fit_transform(question_embeddings)

    save_path = Path('data') / 'embeddings' / 'tsne' / 'embedding'
    np.savez(str(save_path), embeddings_2d=embeddings_2d, urls=urls)


if __name__ == '__main__':
    create_embeddings()
