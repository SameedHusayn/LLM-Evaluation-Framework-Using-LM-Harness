"""
CoQA: A Conversational Question Answering Challenge
https://arxiv.org/pdf/1808.07042.pdf

CoQA is a large-scale dataset for building Conversational Question Answering 
systems. The goal of the CoQA challenge is to measure the ability of machines to 
understand a text passage and answer a series of interconnected questions that 
appear in a conversation.

Homepage: https://stanfordnlp.github.io/coqa/

@misc{reddy2018coqa,
    title={CoQA: A Conversational Question Answering Challenge},
    author={Siva Reddy and Danqi Chen and Christopher D. Manning},
    year={2018},
    eprint={1808.07042},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""
import os
import json
import transformers.data.metrics.squad_metrics as squad_metrics
from lm_eval.base import Task, rf, mean
from ..utils import sh
from itertools import zip_longest
from best_download import download_file


class CoQA(Task):
    VERSION = 1

    def download(self):
        coqa_train_filepath = 'data/coqa/coqa-train-v1.0.json'
        coqa_dev_filepath = 'data/coqa/coqa-dev-v1.0.json'

        sh ("""mkdir -p data/coqa""")

        download_file("http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json", local_file=coqa_train_filepath, expected_checksum="b0fdb2bc1bd38dd3ca2ce5fa2ac3e02c6288ac914f241ac409a655ffb6619fa6")
        download_file("http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json", local_file=coqa_dev_filepath, expected_checksum="dfa367a9733ce53222918d0231d9b3bedc2b8ee831a2845f62dfc70701f2540a")

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return json.load(open('data/coqa/coqa-train-v1.0.json'))['data']

    def validation_docs(self):
        return json.load(open('data/coqa/coqa-dev-v1.0.json'))['data']

    def test_docs(self):
        pass

    def doc_to_text(self, doc):
        # Given a passage p, the conversation history {q1, a1, . . . qi−1, ai−1} 
        # and a question qi, the task is to predict the answer ai
        doc_text = doc["story"] + '\n\n'
        for (q, a) in zip_longest(doc["questions"], doc["answers"][:-1]):   # omit target answer ai
            question = f"Q: {q['input_text']}" + '\n\n'
            answer = f"A: {a['input_text']}" + '\n\n' if a is not None else "A:"
            doc_text += question + answer
        return doc_text
        
    @classmethod
    def get_answers(cls, doc, turn_id):
        # Returns unique answers and valid alternatives (Some questions in CoQA have multiple valid answers).
        answers = []
        answer_forturn = doc["answers"][turn_id - 1]["input_text"]
        answers.append(answer_forturn)
        
        additional_answers = doc.get("additional_answers")
        if additional_answers:
            for key in additional_answers:
                additional_answer_for_turn = additional_answers[key][turn_id - 1]["input_text"]
                if additional_answer_for_turn.lower() not in map(str.lower, answers):
                    answers.append(additional_answer_for_turn)
        return answers
    
    @classmethod
    def get_answer_choice(self, raw_text):
        # Function maps answers to CoQA answer categories
        # ~ 1/5 of the CoQA answers are Yes/No 
        # ~ 2/3 of the CoQA answers are span-based
        # (answers overlap with the passage ignoring punctuation and case mismatch)
        if raw_text == "unknown":
            return '0'
        if squad_metrics.normalize_answer(raw_text) == "yes":
            return '1'
        if squad_metrics.normalize_answer(raw_text) == "no":
            return '2'
        return '3' # Not a yes/no question

    @staticmethod
    def compute_scores(gold_list, pred):
        # tests for exact match and on the normalised answer (compute_exact)
        # test for overlap (compute_f1)
        f1_sum = 0.0
        em_sum = 0.0
        if len(gold_list) > 1:
            for i in range(len(gold_list)):
                gold_answers = gold_list[0:i] + gold_list[i + 1:]
                # predictions compared against (n) golds and take maximum
                em_sum += max(squad_metrics.compute_exact(a, pred) for a in gold_answers)
                f1_sum += max(squad_metrics.compute_f1(a, pred) for a in gold_answers)
        else:
            em_sum += max(squad_metrics.compute_exact(a, pred) for a in gold_list)
            f1_sum += max(squad_metrics.compute_f1(a, pred) for a in gold_list)

        return {'em': em_sum / max(1, len(gold_list)), 'f1': f1_sum / max(1, len(gold_list))}

    def doc_to_target(self, doc, turnid=None):
        # Default to prediction of last turn.
        if turnid is None:
            turnid = len(doc["questions"])
        raw_text = doc['answers'][turnid - 1]["input_text"]
        return " " + raw_text

    def construct_requests(self, doc, ctx):
        """ Uses RequestFactory to construct Requests and returns an iterable of 
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural 
            language description, as well as the few shot examples, and the question
            part of the document for `doc`. 
        """
        cont_request = rf.greedy_until(ctx, ['\nQ:'])
        return cont_request

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a 
        dict where keys are the names of submetrics and values are the values of 
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        turn_id = len(doc["questions"])
        gold_list = self.get_answers(doc, turn_id)
        pred = results[0].strip().split('\n')[0]

        scores = self.compute_scores(gold_list, pred)

        return {
            "f1": scores['f1'],
            "em": scores['em'],
        }

    def higher_is_better(self):
        return {
            "f1": True,
            "em": True,
        }

    def aggregation(self):
        return {
            "f1": mean,
            "em": mean,
        }
