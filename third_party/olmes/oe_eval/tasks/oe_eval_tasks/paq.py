"""
TriviaQA
"""

from typing import List, Union

from oe_eval.components.instances import RequestInstance
from oe_eval.components.requests import RequestType
from oe_eval.metrics.metric import GenericMetric
from oe_eval.tasks.base_task import Task
from oe_eval.tasks.utils import make_cloze_prompt
from functools import cached_property

_CITATION = """
@article{lewis2021paq,
      title={PAQ: 65 Million Probably-Asked Questions and What You Can Do With Them}, 
      author={Patrick Lewis and Yuxiang Wu and Linqing Liu and Pasquale Minervini and Heinrich Küttler and Aleksandra Piktus and Pontus Stenetorp and Sebastian Riedel},
      year={2021},
      eprint={2102.07033},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


class PAQ(Task):
    REQUEST_TYPE = RequestType.GENERATE_UNTIL
    TASK_CONFIG_DEFAULTS: dict = {
        "dataset_path": "dhgottesman/paq_sample",
        "split": "train",
        "native_id_field": "id",
        "primary_metric": "exact_match",
        "fewshot_source": None,
        "context_kwargs": {
            "description": None,  # "Answer the following question."
        },
        "num_shots": 10,
        "generation_kwargs": {
            "max_gen_toks": 50,
            "temperature": 0.0,
            "do_sample": False,
            "stop_sequences": ["\n\n"],
        },
    }

    def make_metrics(self):
        self._metrics = [
            GenericMetric(
                process_results_fn=self.process_results,
                metric_names=["exact_match", "exact_match_s_100", "exact_match_s_1", "exact_match_o_100", "exact_match_o_1"],
            )
        ]
        return self._metrics

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return [self._process_doc(doc, idx) for idx, doc in enumerate(self.dataset["train"])]

    def _process_doc(self, doc, idx):
        # Question: what is the door on bute house made of
        # Answer: polished black oak
        return {
            "id": idx,
            "query": doc["question"],
            "answer": doc["answer"],
            "sitelink_count": doc["sitelink_count"]
        }

    def doc_to_text(self, doc):
        return "Q: " + doc["query"] + " A:"

    def doc_to_target(self, doc):
        return " " + doc["answer"][0]

    def construct_requests(
        self, doc: dict, ctx: Union[str, list, dict], doc_id: int
    ) -> List[RequestInstance]:
        return self.construct_basic_generation_requests(doc, ctx, doc_id, label=doc["answer"])

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        preds, labels = results, doc["answer"]

        # Filter out examples where the answer is in the question
        if any(
            [
                label in doc["query"] or label.lower() in doc["query"] or label.capitalize() in doc["query"]
                for label in labels
            ]
        ):
            return {}
        
        is_correct = 0.0
        for label in labels:
            if any(
                [
                    label in pred or label.lower() in pred or label.capitalize() in pred
                    for pred in preds
                ]
            ):
                is_correct = 1.0

        results = {"exact_match": is_correct}

        # Calculate popularity sliced metric
        s_sitelinks = doc["sitelink_count"]
        # FIX o_sitelinks = doc["o_uri"]]


        if s_sitelinks >= 100:
            results["exact_match_s_100"] = is_correct
        else:
            results["exact_match_s_1"] = is_correct
        # if o_sitelinks >= 100:
        #     results["exact_match_o_100"] = is_correct
        # else:
        #     results["exact_match_o_1"] = is_correct
        return results