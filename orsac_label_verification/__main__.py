# Filename: __main__.py
import abc
import json

from orsac_label_verification.orsac_evaluate import orsac_evaluate
from orsac_label_verification.orsac_test import orsac_test
from orsac_label_verification.train import train
from orsac_label_verification.train_config import (  # , PathsConfig
    ExperimentationConfig,
)


class Executor:
    def __init__(self, args):
        if args.config is not None:
            self.config = []
            for i, config in enumerate(args.config):
                with open(config) as json_file:
                    self.config.append(json.load(json_file))

                if "mode" in args and (
                    "mode" not in self.config[i] or self.config[i]["mode"] != args.mode
                ):
                    self.config[i]["mode"] = args.mode

                if args.batch_size:
                    self.config[i]["batch_size"] = args.batch_size
        else:
            self.config = None
        self.args = args

    @abc.abstractmethod
    def execute(self):
        return NotImplemented


class TrainingExecutor(Executor):
    def __init__(self, args):
        super(TrainingExecutor, self).__init__(args)
        assert self.config is not None, "need configuration file for training provided"
        for i in range(len(self.config)):
            self.config[i][
                "mode"
            ] = args.mode  # Add this line to update the mode value in the configuration

    def execute(self):
        for config in self.config:
            train_settings = ExperimentationConfig.parse_obj(config)
            train(train_settings)


class TestingExecutor(Executor):
    def __init__(self, args):
        super(TestingExecutor, self).__init__(args)
        assert self.config is None, "No configuration needed for testing"

    def execute(self):
        for exp in self.args.exp_dir:
            orsac_test(self.args.exp_dir)


class EvaluationExecutor(Executor):
    def __init__(self, args):
        super(EvaluationExecutor, self).__init__(args)
        assert self.config is None, "No configuration needed for evaluation"

    def execute(self):
        orsac_evaluate(self.args.exp_dir)


EXECUTORS = {
    "train": TrainingExecutor,
    "test": TestingExecutor,
    "eval": EvaluationExecutor,
}


def get_executor(mode: str) -> Executor:
    return EXECUTORS[mode]


def run(args=None):
    from argparse import ArgumentParser

    parser = ArgumentParser("orsac_label_verification")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=sorted(EXECUTORS.keys()),
        help="Overwrite mode from the configuration file",
    )

    parser.add_argument(
        "--config", type=str, help="The configuration file for training",action='append'
    )

    parser.add_argument(
        "--exp-dir", type=str, help="The experiment directory for tests",action='append'
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Overwrite the batch size from configuration",
        default=None,
    )

    args = parser.parse_args(args)
    print(args)  # Add this line to print the parsed arguments

    try:
        ExecutorClass = get_executor(args.mode)
        print(
            "Executor class:", ExecutorClass
        )  # Add this line to print the Executor class
        executor = ExecutorClass(args)
        print(
            "executor object:", executor
        )  # Add this line to print the executor object
        executor.execute()
    except KeyError:
        parser.print_help()
        

if __name__ == "__main__":
    run()
