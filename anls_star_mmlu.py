from deepeval.benchmarks.mmlu.mmlu import MMLU
from deepeval.benchmarks.mmlu.mmlu import Golden
from deepeval.benchmarks.mmlu.task import MMLUTask


class GroundTruthGetterMMLU:
    def __init__(self):
        self.mmlu_obj = MMLU(n_shots=5)

    def get_ground_truths_task(self, task: MMLUTask) -> list:
        """returns ground truth data structure for MMLU subtask"""
        return self.mmlu_obj.load_benchmark_dataset(task)

    def extract_ground_truth(self, golden: Golden) -> str:
        """extracts the correct answer from a Golden (Deepeval datastructure which stores correct answer"""
        pass


class AnlsStarMMLUEvaluator:
    def __init__(self, predictions: dict):
        self.predictions = predictions

    def compute_anls_star_subtask(self, task: MMLUTask):
        """computes anls_star metric for a subtask of MMLU using ground truth string and predicted answer string"""
        pass

    def compute_anls_star_average(self):
        """computes anls_star average over all subtasks"""
        pass
