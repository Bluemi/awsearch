from pathlib import Path
from typing import List

import numpy as np
import pygame as pg
from viztools.drawable.points import Points
from viztools.viewer import Viewer

from data import load_qa_id


class EmbeddingViewer(Viewer):
    def __init__(self, points: np.ndarray, urls: list[str], qa_ids: List[str]):
        super().__init__(screen_size=(0, 0))
        self.points = Points(points, color=np.array([0, 255, 0, 50]))
        self.urls = urls
        self.qa_ids = qa_ids
        self.cache = {}

    def tick(self, delta_time: float):
        pass

    def render(self):
        self.render_coordinate_system()
        self.render_drawables([self.points])

    def handle_event(self, event: pg.event.Event):
        super().handle_event(event)
        for p_index in self.points.hovered_points(self.mouse_pos, self.coordinate_system):
            qa = load_qa_id(self.qa_ids[p_index], self.cache)
            print(qa.question[:20])
        for p_index in self.points.clicked_points(event, self.coordinate_system):
            print(self.urls[p_index])


def show_embeddings():
    embeddings_path = Path('data') / 'embeddings' / 'tsne' / 'embedding.npz'
    embedding = np.load(str(embeddings_path))
    embeddings_2d = embedding['embeddings_2d']
    urls = embedding['urls']
    qa_ids = embedding['qa_ids']

    viewer = EmbeddingViewer(embeddings_2d, urls, qa_ids)
    viewer.run()


if __name__ == '__main__':
    show_embeddings()
