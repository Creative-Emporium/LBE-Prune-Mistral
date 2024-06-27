from typing import List, Dict, Optional
import pandas as pd
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.models import DeepEvalBaseLLM
from tqdm import tqdm


class MyMMLU(MMLU):
    """
    we override the MMLU class within DeepEval to extract the ground truth responses ("goldens") for use in ANSL* metric
    the evaluate method is identical to the parent class, except that we extract the goldens_per_task into a class variable
    """

    def __init__(self, tasks: List[MMLUTask] = None, n_shots: int = 5):
        super().__init__(tasks, n_shots)
        self.goldens_per_task = {}

    def evaluate(self, model: DeepEvalBaseLLM) -> Dict:
        overall_correct_predictions = 0
        overall_total_predictions = 0
        predictions_row = []
        scores_row = []

        for task in self.tasks:
            goldens = self.load_benchmark_dataset(task)
            self.goldens_per_task[task] = goldens
            task_correct_predictions = 0
            task_total_predictions = len(goldens)
            overall_total_predictions += len(goldens)

            # Calculate task accuracy
            for golden in tqdm(goldens, desc=f"Processing {task.value}"):
                prediction, score = self.predict(model, task, golden).values()
                if score:
                    task_correct_predictions += 1
                    overall_correct_predictions += 1
                predictions_row.append((task.value, golden.input, prediction, score))
            task_accuracy = task_correct_predictions / task_total_predictions
            print(f"MMLU Task Accuracy (task={task.value}): {task_accuracy}")
            scores_row.append((task.value, task_accuracy))

        # Calculate overall accuracy
        overall_accuracy = overall_correct_predictions / overall_total_predictions
        print(f"Overall MMLU Accuracy: {overall_accuracy}")

        # Create a DataFrame from task_results_data
        # Columns: 'Task', 'Input', 'Prediction', 'Score'
        self.predictions = pd.DataFrame(
            predictions_row, columns=["Task", "Input", "Prediction", "Correct"]
        )
        self.task_scores = pd.DataFrame(scores_row, columns=["Task", "Score"])
        self.overall_score = overall_accuracy

        return overall_accuracy
