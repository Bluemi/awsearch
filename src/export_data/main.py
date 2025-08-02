import argparse
import json
from typing import Dict
from pathlib import Path

import numpy as np
from abgeordnetenwatch_python.models.questions_answers import QuestionAnswerResult
from tqdm import tqdm

from data import LoadDossiers
import questionbase_pb2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_clusters', '-c', type=int, default=35)
    return parser.parse_args()


def main():
    args = get_args()

    # load url to question
    url_to_question = load_url_to_question()

    # load question data
    embeddings_path = Path('data') / 'embeddings' / 'tsne' / 'embedding.npz'
    embedding = np.load(str(embeddings_path))
    embeddings_2d = embedding['embeddings_2d']
    urls = embedding['urls']

    # load cluster data
    with open(f'data/embeddings/cluster/bundestag/cluster{args.n_clusters}.json', 'r') as f:
        cluster = json.load(f)

    cluster_ids = np.array([cluster[url] for url in urls])

    questionbase = questionbase_pb2.QuestionBase()

    # export questions
    for pos, url, cluster_id in zip(embeddings_2d, urls, cluster_ids):
        question = questionbase_pb2.Question()
        question.x = float(pos[0])
        question.y = float(pos[1])
        question.cluster_id = int(cluster_id)
        questionbase.questions.append(question)

    # export clusters
    cluster_centers, topics = load_cluster_data(args, cluster_ids, embeddings_2d)
    for index, (center, topic) in enumerate(zip(cluster_centers, topics)):
        cluster = questionbase_pb2.Cluster()
        cluster.topic = topic
        cluster.center_x = float(center[0])
        cluster.center_y = float(center[1])
        questionbase.clusters.append(cluster)

    buf = questionbase.SerializeToString()

    out_file = 'data/export/export.bin'
    with open(out_file, 'wb') as f:
        f.write(buf)
        buflen = len(buf)
    print(f'Wrote {buflen} bytes to {out_file}')


def load_cluster_data(args, cluster_ids, embeddings_2d):
    cluster_centers = []
    topics = []
    with open(f'data/embeddings/cluster/bundestag/topics{args.n_clusters}.json', 'r') as f:
        topics_dict = json.load(f)
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
    return cluster_centers, topics


def load_url_to_question() -> Dict[str, QuestionAnswerResult]:
    data_dir = Path('data/json/bundestag')

    url_to_question = {}
    for path, dossier in tqdm(LoadDossiers(data_dir), desc='load questions'):
        for qa in dossier.questions_answers.questions_answers:
            url_to_question[qa.url] = qa
    return url_to_question


if __name__ == '__main__':
    main()
