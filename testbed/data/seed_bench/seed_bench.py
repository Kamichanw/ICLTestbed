import json
import os
from pathlib import Path
import pdb

import datasets

from testbed.data.common import split_generators


_URLS = {
    "annotations": {
        "test": "https://huggingface.co/datasets/AILab-CVC/SEED-Bench/raw/main/SEED-Bench.json",
    },
    "images": {
        "test": "https://huggingface.co/datasets/AILab-CVC/SEED-Bench/raw/main/SEED-Bench-image.zip"
    },
}

_SUB_FOLDER_OR_FILE_NAME = {
    "annotations": {
        "test": "SEED-Bench.json",
    },
    "images": {
        "test": "SEED-Bench-image",
    },
}


class SEEDBenchConfig(datasets.BuilderConfig):

    def __init__(self, images_dir=None, verbose=True, **kwargs):
        self.verbose = verbose
        self.images_dir = images_dir
        super().__init__(**kwargs)


class SEEDBenchDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = SEEDBenchConfig

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "answer": datasets.Value("string"),
                    "choice_a": datasets.Value("string"),
                    "choice_b": datasets.Value("string"),
                    "choice_c": datasets.Value("string"),
                    "choice_d": datasets.Value("string"),
                    "data_id": datasets.Value("string"),
                    "data_type": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "question_id": datasets.Value("string"),
                    "question_type_id": datasets.Value("string"),
                    "image": datasets.Image(),
                    "segment": datasets.Value("string", id=None),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        if self.config.data_dir is None:
            raise ValueError("Missing arguments for data_dir")

        if self.config.images_dir is None:
            # we don't use the exact images root dir here
            # the actual image_path should be data_dir / _SUB_FOLDER_OR_FILE_NAME["images"]["test"]
            # we will handle this in expand_path_fn
            self.config.images_dir = os.path.join(
                self.config.data_dir, _SUB_FOLDER_OR_FILE_NAME["images"]["test"]
            )

        def expand_path_fn(file_type, split_name):
            if file_type == "images":
                return Path(self.config.images_dir).resolve()
            return (
                Path(self.config.data_dir).resolve()
                / _SUB_FOLDER_OR_FILE_NAME[file_type][split_name]
            )

        return split_generators(
            expand_path_fn,
            _SUB_FOLDER_OR_FILE_NAME,
            self.config.verbose,
        )

    def _generate_examples(self, split, annotations_path, images_path):
        with open(annotations_path, "r") as f:
            data = json.load(f)

        for item in data["questions"]:
            answer = item["answer"]
            choice_a = item["choice_a"]
            choice_b = item["choice_b"]
            choice_c = item["choice_c"]
            choice_d = item["choice_d"]
            data_id = item["data_id"]
            data_type = item["data_type"]
            question = item["question"]
            question_id = item["question_id"]
            question_type_id = item["question_type_id"]
            image = str(images_path.resolve() / data_id)
            segment = item.get("segment", None)

            if os.path.exists(image):
                yield question_id, {
                    "answer": answer,
                    "choice_a": choice_a,
                    "choice_b": choice_b,
                    "choice_c": choice_c,
                    "choice_d": choice_d,
                    "data_id": data_id,
                    "data_type": data_type,
                    "question": question,
                    "question_id": question_id,
                    "question_type_id": question_type_id,
                    "image": image,
                    "segment": segment,
                }
