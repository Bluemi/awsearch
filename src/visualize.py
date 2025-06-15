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


def random_colors(n):
    colors = np.random.randint(0, 256, size=(n, 3), dtype=np.uint8)
    mask = colors.max(axis=1) < 100
    count = mask.sum()
    if count:
        idx = np.random.randint(0, 3, size=count)
        colors[mask, idx] = np.random.randint(100, 256, size=count, dtype=np.uint8)

    return np.concatenate((colors, np.full((n, 1), 50)), axis=1)


class EmbeddingViewer(Viewer):
    def __init__(
            self, points: np.ndarray, urls: list[str], qa_ids: List[str], color_ids: np.ndarray,
            cluster_centers: np.ndarray, topics: List[str]
    ):
        super().__init__(screen_size=(0, 0))
        n_colors = np.max(color_ids) + 1
        all_colors = random_colors(n_colors)
        colors = all_colors[color_ids]
        self.points = Points(
            points,
            color=colors,
            size=2
        )
        self.cluster_labels = [
            OverlayText(
                t, p, font_size=16, color=c, background_color=np.array([0, 0, 0, 180]), font_name='liberationmono',
            ) for t, p, c in zip(topics, cluster_centers, all_colors)
        ]
        self.question_text: OverlayText | None = None
        self.urls = urls
        self.qa_ids = qa_ids
        self.cache = {}

    def render(self):
        self.render_coordinate_system(draw_numbers=False)
        self.render_drawables([self.points])
        self.render_drawables(self.cluster_labels)
        if self.question_text:
            self.render_drawables([self.question_text])

    def handle_event(self, event: pg.event.Event):
        super().handle_event(event)
        if event.type == pg.MOUSEMOTION:
            point_index, dist = self.points.closest_point(self.mouse_pos, self.coordinate_system)
            if dist < 10:
                qa = load_qa_id(self.qa_ids[point_index], self.cache)
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
    parser.add_argument('--d2', action='store_true', help='use 2d embeddings')
    return parser.parse_args()


def main():
    args = get_args()
    embeddings_path = Path('data') / 'embeddings' / 'tsne' / 'embedding.npz'
    embedding = np.load(str(embeddings_path))
    embeddings_2d = embedding['embeddings_2d']
    urls = embedding['urls']
    qa_ids = embedding['qa_ids']
    d2_str = '_2d' if args.d2 else ''

    with open(f'data/embeddings/cluster/bundestag/cluster{args.n_clusters}{d2_str}.json', 'r') as f:
        cluster = json.load(f)

    with open(f'data/embeddings/cluster/bundestag/topics{args.n_clusters}{d2_str}.json', 'r') as f:
        topics_dict = json.load(f)

    cluster_ids = np.array([cluster[url] for url in urls])

    # calculate 2d cluster centers
    cluster_centers = []
    topics = []
    for c_id in range(np.max(cluster_ids) + 1):
        indices = np.equal(cluster_ids, c_id)
        emb_2d = embeddings_2d[indices]
        if len(emb_2d) == 0:
            cluster_center = np.zeros(2)
        else:
            cluster_center = np.mean(emb_2d, axis=0)
        cluster_centers.append(cluster_center)
        topics.append(topics_dict.get(str(c_id), ' '))
    cluster_centers = np.array(cluster_centers)

    assert len(cluster) == len(embeddings_2d) == len(urls) == len(qa_ids)
    assert len(cluster_centers) == len(topics), f'{len(cluster_centers)} != {len(topics)}'
    viewer = EmbeddingViewer(embeddings_2d, urls, qa_ids, cluster_ids, cluster_centers, topics)
    viewer.run()


if __name__ == '__main__':
    main()
