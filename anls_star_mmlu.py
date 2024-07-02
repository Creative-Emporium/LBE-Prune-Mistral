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
        """extracts the correct label and answer as a single str from a Golden
        (Deepeval datastructure which stores correct answer)"""
        parsed_result: list = golden.input.split("\n")
        assert parsed_result[-1] == "Answer:"
        answers_with_labels: list = parsed_result[-5:-1]
        import re

        regex_answer_format = re.compile(r"[A-D]. [a-zA-Z\d ]+")
        assert all(regex_answer_format.match(answer) for answer in answers_with_labels)
        correct_label = golden.expected_output
        index: int = self.__get_list_index_from_label(correct_label)
        return answers_with_labels[index]

    def __get_list_index_from_label(self, label: str) -> int:
        """helper function for extract_ground_truth;
        gets index for answer subarray depending on correct label"""
        match label:
            case "A":
                return 0
            case "B":
                return 1
            case "C":
                return 2
            case "D":
                return 3
            case _:
                raise IndexError("label must be A, B, C or D")


class AnlsStarMMLUEvaluator:
    def __init__(self, predictions: dict):
        self.predictions = predictions

    def compute_anls_star_subtask(self, task: MMLUTask):
        """computes anls_star metric for a subtask of MMLU using ground truth string and predicted answer string"""
        pass

    def compute_anls_star_average(self):
        """computes anls_star average over all subtasks"""
        pass
