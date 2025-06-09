from pathlib import Path

import numpy as np
import pygame as pg
from sklearn.manifold import TSNE
from viztools.drawable import Points
from viztools.viewer import Viewer

# from cuml import TSNE

from data import LoadDossierEmbeddings

N_CLUSTERS = 25


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


class EmbeddingViewer(Viewer):
    def __init__(self, points: np.ndarray, urls: list[str]):
        super().__init__()
        self.points = Points(points, colors=pg.Color(0, 255, 0, 50))
        self.urls = urls

    def tick(self, delta_time: float):
        pass

    def render(self):
        self.render_coordinate_system()
        self.render_drawables([self.points])

    def handle_event(self, event: pg.event.Event):
        super().handle_event(event)
        for p_index in self.points.clicked_points(event, self.coordinate_system):
            print(self.urls[p_index])


def show_embeddings():
    embeddings_path = Path('data') / 'embeddings' / 'tsne' / 'embedding.npz'
    embedding = np.load(str(embeddings_path))
    embeddings_2d = embedding['embeddings_2d']
    urls = embedding['urls']

    viewer = EmbeddingViewer(embeddings_2d, urls)
    viewer.run()


if __name__ == '__main__':
    # create_embeddings()
    show_embeddings()
