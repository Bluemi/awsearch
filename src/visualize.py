import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pygame as pg
from viztools.drawable.overlay_text import OverlayText, OverlayPosition
from viztools.drawable.points import Points
from viztools.viewer import Viewer

from data import load_qa_id


class EmbeddingViewer(Viewer):
    def __init__(self, points: np.ndarray, urls: list[str], qa_ids: List[str], color_ids: np.ndarray):
        super().__init__(screen_size=(0, 0))
        random_colors = np.random.randint(50, 255, (len(set(color_ids)), 3))
        random_colors = np.concatenate((random_colors, np.full((len(random_colors), 1), 50)), axis=1)
        colors = random_colors[color_ids]
        self.points = Points(
            points,
            color=colors,
            size=4
        )
        self.question_text: OverlayText | None = None
        self.urls = urls
        self.qa_ids = qa_ids
        self.cache = {}

    def render(self):
        self.render_coordinate_system()
        self.render_drawables([self.points])
        if self.question_text:
            self.render_drawables([self.question_text])

    def handle_event(self, event: pg.event.Event):
        super().handle_event(event)
        if event.type == pg.MOUSEMOTION:
            hovered_ids = self.points.hovered_points(self.mouse_pos, self.coordinate_system)
            if len(hovered_ids) > 0:
                qa = load_qa_id(self.qa_ids[hovered_ids[0]], self.cache)
                text_parts = ['\nFrage:']
                max_length = 190
                text_parts.extend(cut_text(qa.question, max_length))
                if qa.question_addition:
                    text_parts.extend(cut_text(qa.question_addition, max_length))
                if qa.answer:
                    text_parts.append('\nAntwort:')
                    text_parts.extend(cut_text(qa.answer, max_length))
                else:
                    text_parts.append('\n<keine Antwort>')
                self.question_text = OverlayText(
                    '\n'.join(text_parts),
                    OverlayPosition.BOT,
                    background_color=(30, 30, 30, 200),
                    border_color=(50, 50, 50, 200),
                    font_name='liberationmono',
                )
            else:
                self.question_text = None
            self.render_needed = True


def cut_text(text: str, max_length: int):
    if not text:
        return []
    parts = []
    words = text.split()
    current_part = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_length:
            current_part.append(word)
            current_length += len(word) + 1
        else:
            parts.append(' '.join(current_part))
            current_part = [word]
            current_length = len(word) + 1

    if current_part:
        parts.append(' '.join(current_part))

    return parts


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_clusters', '-c', type=int, default=35)
    return parser.parse_args()


def main():
    args = get_args()
    embeddings_path = Path('data') / 'embeddings' / 'tsne' / 'embedding.npz'
    embedding = np.load(str(embeddings_path))
    embeddings_2d = embedding['embeddings_2d']
    urls = embedding['urls']
    qa_ids = embedding['qa_ids']

    with open(f'data/embeddings/cluster/bundestag/cluster{args.n_clusters}.json', 'r') as f:
        cluster = json.load(f)

    color_ids = np.array([cluster[url] for url in urls])

    assert len(cluster) == len(embeddings_2d) == len(urls) == len(qa_ids)
    viewer = EmbeddingViewer(embeddings_2d, urls, qa_ids, color_ids)
    viewer.run()


if __name__ == '__main__':
    main()
