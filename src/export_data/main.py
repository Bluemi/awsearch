import argparse
import json
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
from abgeordnetenwatch_python.models.politicians import Politician
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
    questioner: Optional[str] = None
    politician: Optional[str] = None
    subject_area: Optional[str] = None
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
    for index, (pos, url, cluster_id) in tqdm(
            enumerate(zip(embeddings_2d, urls, cluster_ids)), desc='export', total=len(urls)
    ):
        # preview questions
        preview_question = questionbase_pb2.PreviewQuestion()
        preview_question.x = float(pos[0])
        preview_question.y = float(pos[1])
        preview_question.cluster_id = int(cluster_id)
        preview_questionbase.questions.append(preview_question)

        # complete questions
        qa, politician = url_to_question[url]
        question_text = qa.question or ''
        extra_info = parse_question_string(question_text, politician.get_full_name())
        if qa.question_addition:
            if extra_info is None:
                question_text = question_text + '\n' + qa.question_addition
            else:
                question_text = qa.question_addition

        answer_text = None
        answer_date = None
        if qa.answer:
            answer_text = qa.answer
            answer_date = qa.get_answer_date()

        complete_question = {
            'id': index, 'url': url, 'question': question_text, 'question_date': qa.get_question_date(),
            'politician': politician.get_full_name(), 'politician_url': politician.abgeordnetenwatch_url
        }
        if extra_info is not None:
            complete_question['questioner'] = extra_info['questioner']
            complete_question['subject_area'] = extra_info['subject_area']

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


def load_url_to_question() -> Dict[str, Tuple[QuestionAnswerResult, Politician]]:
    data_dir = Path('data/json/bundestag')

    url_to_question = {}
    for path, dossier in tqdm(LoadDossiers(data_dir), desc='load questions'):
        for qa in dossier.questions_answers.questions_answers:
            url_to_question[qa.url] = qa, dossier.politician
    return url_to_question

def parse_question_string(text: str, politician_name: str) -> dict | None:
    """
    Parses a string of the format "Frage an <respondent> von <questioner>
    bezüglich <subject_area>..." and extracts the relevant information.

    :param text: The input string to parse.

    :returns: A dictionary with keys 'respondent', 'questioner', and 'subject_area' if the text matches the pattern and
    the subject is valid. Returns None otherwise.
    """
    valid_subjects = [
        'Familie', 'Klima', 'Außenwirtschaft', 'Reaktorsicherheit', 'Geschäftsordnung', 'Immunität', 'Kultur', 'Medien',
        'Medien, Kommunikation und Informationstechnik', 'Soziale Sicherung', 'Verbraucherschutz',
        'Bildung und Erziehung', 'Technologiefolgenabschätzung', 'Forschung', 'Bundestag', 'Wahlprüfung', 'Verkehr',
        'Innere Angelegenheiten', 'Verteidigung', 'Tourismus', 'Sport, Freizeit und Tourismus',
        'Europapolitik und Europäische Union', 'Energie', 'Deutsche Einheit / Innerdeutsche Beziehungen (bis 1990)',
        'Recht', 'Jugend', 'Landwirtschaft und Ernährung', 'Arbeit und Beschäftigung', 'Wirtschaft', 'Finanzen',
        'Politisches Leben, Parteien', 'Senioren', 'Öffentliche Finanzen, Steuern und Abgaben',
        'Gesellschaftspolitik, soziale Gruppen', 'Migration und Aufenthaltsrecht', 'Haushalt', 'Frauen',
        'Entwicklungspolitik', 'Umwelt', 'Wissenschaft, Forschung und Technologie',
        'Außenpolitik und internationale Beziehungen', 'Petitionen', 'Menschenrechte', 'Digitale Agenda',
        'Innere Sicherheit', 'Lobbyismus & Transparenz', 'Staat und Verwaltung', 'Gesundheit', 'Humanitäre Hilfe',
        'Naturschutz', 'digitale Infrastruktur', 'Sport', 'Raumordnung, Bau- und Wohnungswesen'
    ]

    pattern = fr"Frage an {politician_name} von (.*?) bezüglich (.*)"
    match = re.match(pattern, text)

    # Case 1: The overall pattern does not match the text.
    if match:
        questioner = match.group(1).strip()
        subject_area = match.group(2).strip()
    else:
        pattern = fr"Frage an {politician_name} von (.*)"
        match = re.match(pattern, text)
        if match:
            questioner = match.group(1).strip()
            subject_area = None
        else:
            return None

    if subject_area is not None:
        found = False
        for possible_area in valid_subjects:
            if subject_area.startswith(possible_area):
                subject_area = possible_area
                found = True
                break
    else:
        found = True

    # Case 2: The pattern matched, but the subject area is not in the valid list.
    if not found:
        return None

    # Case 3: A full, valid match.
    return {
        "questioner": questioner,
        "subject_area": subject_area,
    }


def testy():
    text = "Frage an Viola von Cramon-Taubadel von Drei französische S. bezüglich Europapolitik und Europäische Union"
    parse_question_string(text, 'Viola von Cramon-Taubadel')


if __name__ == '__main__':
    main()
