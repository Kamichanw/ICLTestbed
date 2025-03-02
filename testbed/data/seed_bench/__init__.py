from typing import List, Union
from testbed.data.common import register_dataset_retriever, register_postprocess


@register_dataset_retriever(__name__.split(".")[-1])
def retriever(item, is_last: bool):
    num_images = len(item["image"]) if isinstance(item["image"], list) else 1
    return (
        [
            {"role": "image", "content": [{"type": "image"}] * num_images},
            {
                "role": "question",
                "content": [{"type": "text", "text": item["question"]}],
            },
            {
                "role": "choices",
                "content": [
                    {
                        "type": "text",
                        "text": f"A. {item['choice_a']} B. {item['choice_b']} C. {item['choice_c']} D. {item['choice_d']}",
                    },
                    {"type": "text", "text": "Answer with the letter."},
                ],
            },
            (
                {"role": "answer"}
                if is_last
                else {
                    "role": "answer",
                    "content": {"type": "text", "text": item["answer"]},
                }
            ),
        ],
        item["image"],
    )


@register_postprocess(__name__.split(".")[-1])
def postprocess(text: Union[str, List[str]]) -> Union[str, List[str]]:
    return text
