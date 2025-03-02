## How to customize a new dataset?
There are two ways to load datasets using the Hugging Face `datasets` library:

1. **Using a Loading Script**  
   Please refer to the Hugging Face [official documentation](https://huggingface.co/docs/datasets/v2.20.0/en/dataset_script#create-a-dataset-loading-script) for details on creating a dataset loading script.  
   If you need to write your own loading script, place it in:  
   ```
   testbed/data/<dataset_name>/<dataset_name>.py
   ```
   Additionally, register the *dataset retriever* and *post-processing* functions in:  
   ```
   testbed/data/<dataset_name>/__init__.py
   ```

2. **Loading Directly from Hugging Face or Locally**  
   You can also load datasets directly from Hugging Face or a local source.  
   In this case, simply register the *dataset retriever* and *post-processing* functions **before** using  `testbed.data.prepare_dataloader` and `testbed.data.prepare_input`.

### Dataset Retriever

A **dataset retriever** is a function responsible for extracting data from a single row of the dataset and converting it into a Hugging Face conversation dictionary. For more details on Hugging Face conversation dictionaries, refer to [Chatting with Transformer](https://huggingface.co/docs/transformers/main/en/conversations).

For example, given a dataset with a row structured as follows:

```json
{
    "text": "example text",
    "label": "example label",
}
```
You can register a dataset retriever using the following approach:

```python
@register_dataset_retriever("dataset_name")
def retriever(item, is_last: bool):
    return [
        {"role": "user", "content": item["text"]},
        (
            {"role": "assistant"}
            if is_last
            else {
                "role": "assistant",
                "content": item["label"],
            }
        ),
    ]
```

After registration, when calling `testbed.data.prepare_input`, a batch of dataset rows will be transformed into actual conversation dictionaries.

For more details on customization, please refer to the [tutorial](../examples/tutorial_custom.ipynb).

### Post-Processing Function

A **post-processing function** is responsible for converting the model's output into a format that is easier to evaluate. For example, it can map `"yes"`/`"no"` responses to `1` and `0`, respectively.



## How to customize a new metric?

None of my business, see hugging face [official document](https://huggingface.co/docs/evaluate/creating_and_sharing). Just place your new metric into `testbed/evaluate/<metric_name>/<metric_name>.py`.

🚨 If you want replace official evaluation, you should add a test script in `test/` to prove that your code is consistent with official code.

## How to customize a new model?
You need to do follows:
1. Inherit from `ModelBase` that placed at `testbed/models/model_base.py`. It is just a simple wrapper for pretrained model and processor.
2. Implement `default_prompt_template` which is used in `apply_prompt_tempalte` to transform raw texts and images to a model-specific prompt.

The `default_prompt_template` I implemented for each model generally aligns with the template defined in the model's `tokenizer_config.json` (`chat_template`) or `chat_template.json`. However, I have extended its functionality in the following ways:

1. **Automatic Determination of the Generation Role**  
   In the official implementation, when `add_generation_prompt=True` is passed to `apply_chat_template`, the resulting output typically appends `"assistant:"` at the end (depending on the specific model). This behavior limits template flexibility.  

   To address this, I automatically determine the **generation role** based on the last role in a conversation turn. Consider a typical in-context learning scenario where the context follows this format:

   ```
   Question: {question}
   Answer: {answer}
   Question: {question}
   ...
   ```

   In this case, the last role in a conversation turn (e.g., `"Answer"`) is used as the **generation role** in the template.

2. **Allowing Items in the Conversation Dictionary to Omit `"content"`**  
   This feature enables more flexible customization of the **generation role**. If an item in the conversation dictionary lacks a `"content"` field, it signifies that the item serves as a query in an in-context learning setup.  
   
   As a result:
   - It will be placed **at the very end** of the input sequence.
   - No special token (such as `<im_end>`) will be appended to mark the end of a conversation turn.
