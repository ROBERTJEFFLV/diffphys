"""Explicit test-only scales; these numbers are not production recommendations."""
from response_task import TaskLossConfig
from tools.train_response_control import parse_args

LOSS_FLAGS = ('--epsilon-p', '.5', '--epsilon-a', '.25', '--lambda-R', '.4')


def test_loss(**overrides):
    values = dict(epsilon_p=.5, epsilon_a=.25, lambda_R=.4)
    values.update(overrides)
    return TaskLossConfig(**values)


test_loss.__test__ = False


def parse_loss_args(argv=None):
    return parse_args([*LOSS_FLAGS, *(argv or [])])
