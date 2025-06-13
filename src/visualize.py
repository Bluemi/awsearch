from pathlib import Path

import numpy as np
import pygame as pg
from viztools.drawable.points import Points
from viztools.viewer import Viewer


class EmbeddingViewer(Viewer):
    def __init__(self, points: np.ndarray, urls: list[str]):
        super().__init__()
        self.points = Points(points, color=np.array([0, 255, 0, 50]))
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
    show_embeddings()
