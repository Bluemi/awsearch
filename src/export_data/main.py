import argparse
import json
from typing import Dict

import numpy as np
from pathlib import Path

import flatbuffers
from abgeordnetenwatch_python.models.questions_answers import QuestionAnswerResult
from tqdm import tqdm

import export_data.models.Point as Point
import export_data.models.Cluster as Cluster
import export_data.models.Question as Question
import export_data.models.QuestionBase as QuestionBase
from data import LoadDossiers


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

    # export questions
    builder = flatbuffers.Builder(1024)
    questions = []
    for pos, url, cluster_id in zip(embeddings_2d, urls, cluster_ids):
        qa = url_to_question[url]
        url = builder.CreateString(url)
        question_date = builder.CreateString(qa.get_question_date())
        answer_date = None
        if qa.answer_date:
            answer_date = builder.CreateString(qa.get_answer_date())

        Question.Start(builder)
        Question.AddPos(builder, Point.CreatePoint(builder, float(pos[0]), float(pos[1])))
        Question.AddUrl(builder, url)
        Question.AddQuestionDate(builder, question_date)
        if answer_date:
            Question.AddAnswerDate(builder, answer_date)
        Question.AddClusterId(builder, int(cluster_id))
        questions.append(Question.End(builder))
    QuestionBase.StartQuestionsVector(builder, len(questions))
    for q in questions[::-1]:
        builder.PrependUOffsetTRelative(q)
    questions = builder.EndVector()

    # export clusters
    cluster_centers, topics = load_cluster_data(args, cluster_ids, embeddings_2d)
    clusters = []
    for index, (center, topic) in enumerate(zip(cluster_centers, topics)):
        topic = builder.CreateString(topic)
        Cluster.Start(builder)
        Cluster.AddCenter(builder, Point.CreatePoint(builder, float(center[0]), float(center[1])))
        Cluster.AddTopic(builder, topic)
        Cluster.AddId(builder, index)
        clusters.append(Cluster.End(builder))

    QuestionBase.StartClustersVector(builder, len(clusters))
    for c in clusters[::-1]:
        builder.PrependUOffsetTRelative(c)
    clusters = builder.EndVector()

    QuestionBase.Start(builder)
    QuestionBase.AddQuestions(builder, questions)
    QuestionBase.AddClusters(builder, clusters)
    builder.Finish(QuestionBase.End(builder))
    out_file = 'data/export/export.bin'
    with open(out_file, 'wb') as f:
        buf = builder.Output()
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
