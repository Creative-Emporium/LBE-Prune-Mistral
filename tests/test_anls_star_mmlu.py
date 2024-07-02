import pytest

from anls_star_mmlu import AnlsStarMMLUEvaluator, GroundTruthGetterMMLU
from deepeval.benchmarks.mmlu.mmlu import MMLU
from deepeval.benchmarks.mmlu.mmlu import Golden
from deepeval.benchmarks.mmlu.task import MMLUTask


@pytest.fixture()
def task() -> MMLUTask:
    return MMLUTask.HIGH_SCHOOL_EUROPEAN_HISTORY


@pytest.fixture()
def ground_truth_getter() -> GroundTruthGetterMMLU:
    return GroundTruthGetterMMLU()


def test_get_ground_truths_task(
    ground_truth_getter: GroundTruthGetterMMLU, task: MMLUTask
):
    ground_truths: list = ground_truth_getter.get_ground_truths_task(task)
    assert all(type(gt) is Golden for gt in ground_truths)
    assert (
        len(ground_truths) == 165
    )  # size of test split for HIGH_SCHOOL_EUROPEAN_HISTORY
