import argparse
import json
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
from abgeordnetenwatch_python.models.questions_answers import QuestionAnswerResult
from pydantic import BaseModel
from tqdm import tqdm

from data import LoadDossiers
import questionbase_pb2


class CompleteQuestion(BaseModel):
    id: int
    url: str
    question: str
    question_date: str
    answer: Optional[str] = None
    answer_date: Optional[str] = None


class CompleteQuestionBase(BaseModel):
    questions: List[CompleteQuestion]


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

    preview_questionbase = questionbase_pb2.PreviewQuestionBase()

    questions = []
    # export questions
    for index, (pos, url, cluster_id) in enumerate(zip(embeddings_2d, urls, cluster_ids)):
        # preview questions
        preview_question = questionbase_pb2.PreviewQuestion()
        preview_question.x = float(pos[0])
        preview_question.y = float(pos[1])
        preview_question.cluster_id = int(cluster_id)
        preview_questionbase.questions.append(preview_question)

        # complete questions
        qa = url_to_question[url]
        question_text = qa.question or ''
        if qa.question_addition:
            question_text = question_text + '\n' + qa.question_addition

        answer_text = None
        answer_date = None
        if qa.answer:
            answer_text = qa.answer
            answer_date = qa.get_answer_date()

        complete_question = {
            'id': index, 'url': url, 'question': question_text, 'question_date': qa.get_question_date()
        }

        if answer_text:
            complete_question['answer'] = answer_text
            complete_question['answer_date'] = answer_date
        questions.append(CompleteQuestion(**complete_question))

    complete_questionbase = CompleteQuestionBase(questions=questions)

    # export clusters
    cluster_centers, topics = load_cluster_data(args, cluster_ids, embeddings_2d)
    for index, (center, topic) in enumerate(zip(cluster_centers, topics)):
        cluster = questionbase_pb2.PreviewCluster()
        cluster.topic = topic
        cluster.center_x = float(center[0])
        cluster.center_y = float(center[1])
        preview_questionbase.clusters.append(cluster)

    buf = preview_questionbase.SerializeToString()

    export_dir = Path('data/export')
    export_dir.mkdir(parents=True, exist_ok=True)

    preview_out_file = export_dir / 'preview_export.bin'
    with open(preview_out_file, 'wb') as f:
        f.write(buf)
        buffer_length = len(buf)
    print(f'Wrote {buffer_length} bytes to {preview_out_file}')

    complete_out_file = export_dir / 'complete_export.json'
    with open(complete_out_file, 'w') as f:
        buf = complete_questionbase.model_dump_json()
        f.write(buf)
        buffer_length = len(buf)
    print(f'Wrote {buffer_length} bytes to {complete_out_file}')


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
