from deepeval.benchmarks.mmlu.mmlu import MMLU
from deepeval.benchmarks.mmlu.mmlu import Golden
from deepeval.benchmarks.mmlu.task import MMLUTask
from anls_star.anls_star import anls_score


class GroundTruthGetterMMLU:
    def __init__(self):
        self.mmlu_obj = MMLU(n_shots=5)

    def get_ground_truths_task(self, task: MMLUTask) -> list:
        """returns ground truth data structure for MMLU subtask"""
        return self.mmlu_obj.load_benchmark_dataset(task)

    def extract_ground_truth(self, golden: Golden) -> str:
        """extracts the correct label and answer as a single str from a Golden
        (Deepeval datastructure which stores correct answer);
        Examples of format: A. This is the description of answer A
                  C. This is another different description for answer C
        """
        parsed_result: list = golden.input.split("\n")
        assert parsed_result[-1] == "Answer:"
        answers_with_labels: list = parsed_result[-5:-1]
        import re

        regex_answer_format = re.compile(r"[A-D]" + re.escape(r".") + r".+")
        assert all(regex_answer_format.match(answer) for answer in answers_with_labels)
        correct_label = golden.expected_output
        index: int = self.__get_list_index_from_label(correct_label)
        return answers_with_labels[index]

    def get_extracted_ground_truths(self, tasks: list) -> dict:
        """input: list of MMLUTask: tasks for which the ground truths answers should be extracted
        returns: dict with tasks as keys, list of formatted ground truth answers as value
        """
        gt_per_task: dict = {}
        for task in tasks:
            ground_truths: list = self.get_ground_truths_task(task)
            extracted_gts: list = [
                self.extract_ground_truth(golden=gt) for gt in ground_truths
            ]
            gt_per_task[task] = extracted_gts

        return gt_per_task

    def __get_list_index_from_label(self, label: str) -> int:
        """helper function for extract_ground_truth;
        gets index for answer subarray depending on expected output label of Golden"""
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
        self.gt_getter = GroundTruthGetterMMLU()
        self.anls_scores_task = {}

    def compute_anls_star_subtask(self, task: MMLUTask):
        """computes anls_star metric for a subtask of MMLU using ground truth strings and predicted answer strings"""
        ground_truths_dict: dict = self.gt_getter.get_extracted_ground_truths([task])
        ground_truths_task: list = ground_truths_dict[task]
        predictions_task: list = self.predictions[task]
        assert len(ground_truths_task) == len(predictions_task)
        self.anls_scores_task[task] = anls_score(
            gt=ground_truths_task, pred=predictions_task
        )  # no need to loop, anls_score supports evaluation on lists directly (avg computed as well)
        return self.anls_scores_task[task]

    def compute_anls_star_average(self) -> (int, dict):
        """computes anls_star average over all subtasks"""
        subtasks: list = list(self.predictions.keys())
        n_subtasks: int = len(subtasks)
        added_anls = 0
        for task in subtasks:
            added_anls += self.compute_anls_star_subtask(task=task)
        avg_anls = added_anls / n_subtasks
        assert len(self.anls_scores_task.keys()) == len(self.predictions.keys())
        return avg_anls, self.anls_scores_task
