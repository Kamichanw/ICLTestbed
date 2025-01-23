from typing import List, Union
from testbed.data.common import register_dataset_retriever, register_postprocess


@register_dataset_retriever(__name__.split(".")[-1])
def retriever(item, is_last: bool):
    return (
        [
            {"role": "image", "content": [{"type": "image"}]},
            {
                "role": "question",
                "content": item["question"],
            },
            (
                {"role": "answer"}
                if is_last
                else {
                    "role": "answer",
                    "content": item["answer"],
                }
            ),
        ],
        item["image"],
    )


# mroe post process will be done in evaluate procedure
@register_postprocess(__name__.split(".")[-1])
def postprocess(text: Union[str, List[str]]) -> Union[str, List[str]]:
    return text
