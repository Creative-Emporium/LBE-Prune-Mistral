import pytest

from anls_star_mmlu import AnlsStarMMLUEvaluator, GroundTruthGetterMMLU
from deepeval.benchmarks.mmlu.mmlu import Golden
from deepeval.benchmarks.mmlu.task import MMLUTask


@pytest.fixture()
def task() -> MMLUTask:
    return MMLUTask.HIGH_SCHOOL_EUROPEAN_HISTORY


HIGH_SCHOOL_EUROPEAN_TEST_SET_SIZE = 165


@pytest.fixture()
def ground_truth_getter() -> GroundTruthGetterMMLU:
    return GroundTruthGetterMMLU()


@pytest.fixture()
def all_ground_truths(
    task: MMLUTask, ground_truth_getter: GroundTruthGetterMMLU
) -> list:
    ground_truths: list = ground_truth_getter.get_ground_truths_task(task)
    return ground_truths


def test_get_ground_truths_task(
    ground_truth_getter: GroundTruthGetterMMLU, task: MMLUTask
):
    ground_truths: list = ground_truth_getter.get_ground_truths_task(task)
    assert all(type(gt) is Golden for gt in ground_truths)
    assert (
        len(ground_truths) == HIGH_SCHOOL_EUROPEAN_TEST_SET_SIZE
    )  # size of test split for HIGH_SCHOOL_EUROPEAN_HISTORY


def test_extract_ground_truth(
    ground_truth_getter: GroundTruthGetterMMLU, all_ground_truths: list
):
    for single_ground_truth in all_ground_truths:
        expected_label: str = single_ground_truth.expected_output
        gt_label_with_answer: str = ground_truth_getter.extract_ground_truth(
            golden=single_ground_truth
        )
        import re

        regex_expr = re.compile(
            r"([A-D]). ([a-zA-Z\d ]+)"
        )  # first group should match answer label, 2nd group should match answer text
        regex_result = regex_expr.search(gt_label_with_answer)
        extracted_label = regex_result.group(1)
        assert extracted_label == expected_label

        extracted_answer = regex_result.group(2)
        assert len(extracted_answer) > 0


def test_get_extracted_ground_truths(
    ground_truth_getter: GroundTruthGetterMMLU, task: MMLUTask
):
    ground_truths_per_task = ground_truth_getter.get_extracted_ground_truths(
        tasks=[task]
    )
    assert len(ground_truths_per_task[task]) == HIGH_SCHOOL_EUROPEAN_TEST_SET_SIZE
    import re

    regex_answer_format = re.compile(r"[A-D]. [a-zA-Z\d ]+")
    assert all(
        regex_answer_format.match(answer) for answer in ground_truths_per_task[task]
    )
