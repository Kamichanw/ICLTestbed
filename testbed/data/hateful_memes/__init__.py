import nltk
from testbed.data import register_postprocess, register_dataset_retriever


@register_dataset_retriever(__name__.split(".")[-1])
def retriever(item, is_last: bool):
    return (
        [
            {
                "role": "image",
                "content": [{"type": "image"}],
            },
            {
                "role": "question",
                "content": [
                    {
                        "type": "text",
                        "text": f'is an image with written "{item["text"]}" on it. Is it hateful?',
                    }
                ],
            },
            (
                {"role": "answer"}
                if is_last
                else {
                    "role": "answer",
                    "content": "yes" if item["label"] == 1 else "no",
                }
            ),
        ],
        item["img"],
    )


@register_postprocess(__name__.split(".")[-1])
def postprocess(pred):
    hateful_keywords = ["yes", "y", "hateful", "hate"]
    non_hateful_keywords = ["no", "n", "non-hateful", "not hateful", "benign"]

    pred = pred.lower()
    tokens = nltk.word_tokenize(pred)

    for token in tokens:
        if token in hateful_keywords:
            return 1
        elif token in non_hateful_keywords:
            return 0

    return 0
