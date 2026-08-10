import os
import json
import numpy as np
from typing import Callable, Dict, Any, TypeVar, List
from tqdm import tqdm
from collections import Counter

T = TypeVar("T")  # Generic type for accumulated result

class MetricsLogger:
    def __init__(self, run_dir: str):
        """
        Initializes the logger.

        :param log_dir: Directory where metric logs will be stored.
        """
        self.log_dir = os.path.join(run_dir, "metrics")
        self.metrics = {}
        self.batch_index = 0
        os.makedirs(self.log_dir, exist_ok=True)
        self.batch_log_file = os.path.join(self.log_dir, "batch_metrics.jsonl")
        self.dataset_log_file = os.path.join(self.log_dir, "dataset_metrics.json")

    def log_batch_metrics(self, batch_metrics: Dict[str, Any]):
        """
        Logs per-batch metrics and writes them to a file.

        :param batch_metrics: Dictionary containing metric names and values.
        """
        batch_metrics["batch_index"] = self.batch_index

        # Store metrics in memory
        for key, value in batch_metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

        # Append batch metrics to the log file
        with open(self.batch_log_file, "a") as f:
            f.write(json.dumps(batch_metrics) + "\n")

        self.batch_index += 1

    def log_dataset_metrics(self, dataset_metrics: Dict[str, Any]):
        """
        Logs dataset-wide metrics.

        :param dataset_metrics: Aggregated dataset metrics.
        """
        with open(self.dataset_log_file, "w") as f:
            json.dump(dataset_metrics, f, indent=4)

    def load_batch_metrics(self) -> List[Dict[str, Any]]:
        """
        Loads recorded batch metrics.

        :return: List of batch metric dictionaries.
        """
        if not os.path.exists(self.batch_log_file):
            return []

        with open(self.batch_log_file, "r") as f:
            return [json.loads(line) for line in f]

    def load_dataset_metrics(self) -> Dict[str, Any]:
        """
        Loads recorded dataset metrics.

        :return: Dataset metric dictionary.
        """
        if not os.path.exists(self.dataset_log_file):
            return {}

        with open(self.dataset_log_file, "r") as f:
            return json.load(f)

    def get_metric_array(self, metric_name: str) -> np.ndarray:
        """
        Returns metric values as a NumPy array.

        :param metric_name: The name of the metric to retrieve.
        :return: NumPy array of the metric values.
        """
        if metric_name not in self.metrics:
            raise ValueError(f"Metric '{metric_name}' has not been logged yet.")

        return np.array(self.metrics[metric_name])


class BaseMetrics:
    def __init__(self, data_loader, logger: MetricsLogger):
        """
        Initializes the class for computing batch and dataset metrics.

        Args:
            data_loader: Iterable that yields batches of data.
            logger: Logger instance for logging metrics.
        """
        self.data_loader = data_loader
        self.logger = logger
        self.dataset_results = {}
    
    def compute_iterative_metrics(
        self,
        iterative_fn: Callable[[Dict[str, Any], T], T],
        metric_fn: Callable[[T], Dict[str, Any]],
        initial_result: T
    ):
        """
        Computes metrics iteratively, where each batch updates an accumulated result.
    
        Args:
            iterative_metric_fn: A function that takes the current batch and the accumulated result so far,
                                 returning an updated accumulated result.
            initial_state: The initial state of the accumulated metric result.
        """
        result = initial_result

        # Process each batch iteratively
        batch_metric = {}
        for batch in tqdm(self.data_loader, total=len(self.data_loader)):
            result = iterative_fn(batch, result)
            batch_metric = metric_fn(result)
            self.logger.log_batch_metrics(batch_metric)

        # Final result after all batches processed
        self.logger.log_dataset_metrics(batch_metric)

def _entity_coverage(batch, entities):
    """
    Iteratively computes entity coverage by maintaining a running set of encountered entities.

    Args:
        batch: The current batch of data.
        accumulated_result: The accumulated result from previous batches.

    Returns:
        Updated accumulated result.
    """
    metadata_list = batch.get("metadata", [])
    for metadata in metadata_list:
        for entity in metadata.get("entities", []):
            entities.add(entity["name"])

    return entities

def compute_entity_coverage(entities):
    return {"entity_coverage": len(entities)}

# Example usage: 
# logger = MetricsLogger(run_name)
# metrics = BaseMetrics(data_loader, logger)
# metrics.compute_iterative_metrics(_entity_coverage, compute_entity_coverage, set())