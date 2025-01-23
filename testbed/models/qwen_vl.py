from functools import lru_cache
import itertools
from typing import Any, Dict, List, Optional, Union
from PIL.Image import Image
import warnings
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)

from .model_base import ModelBase


HF_QWEN_VL = {
    "": ""
}


class QwenVL(ModelBase):
    def __init__(
        self,
        model_root,
        processor_class=AutoProcessor,
        model_class=Qwen2VLForConditionalGeneration,
        processor_args=None,
        model_args=None,
        **common_args,
    ):
        super().__init__(
            model_root=model_root,
            processor_class=processor_class,
            model_class=model_class,
            support_models=list(name for v in HF_QWEN_VL.values() for name in v),
            processor_args=processor_args,
            model_args=model_args,
            **common_args,
        )
    @property
    def default_prompt_template(self):
        # see https://huggingface.co/docs/transformers/main/model_doc/llava
        # make sure you download from hf official llava, otherwise you should use customize your own prompt template,
        @lru_cache
        def warn_once(msg):
            warnings.warn(msg)

        