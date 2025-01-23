from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from testbed.models.model_base import ModelBase

class Mistral(ModelBase):
    def __init__(
        self,
        model_root,
        processor_class=AutoTokenizer,
        model_class=AutoModelForCausalLM,
        processor_args=None,
        model_args=None,
        **common_args,
    ):
        processor_args = (
            processor_args
            if processor_args
            else dict(chat_template=self.default_prompt_template)
        )

        super().__init__(
            model_root=model_root,
            processor_class=processor_class,
            model_class=model_class,
            processor_args=processor_args,
            model_args=model_args,
            **common_args,
        )

    @property
    def default_prompt_template(self):
        # fmt: off
        return (
            "{% if messages[0]['role'].lower() in ['instruction', 'system'] %}"
                "{{ messages[0]['role'].capitalize() + '\n' + messages[0]['content'] + '\n'}}"
                "{% set messages = messages[1:] %}"
            "{% endif %}"
            "{% for message in messages %}"
                "{% if message['role'] != '' %}"
                    "{{ message['role'].capitalize() }}: "
                "{%+ endif %}"
                "{% if 'content' in message %}"
                    "{% if message['content'] is string %}"
                        "{{ message['content'] }}\n"
                    "{% else %}"
                        "{% for line in message['content'] %}"
                            "{% if line['type'] == 'text' %}"
                                "{{ line['text'] }}"
                            "{% endif %}"
                            "{% if loop.last %}"
                                "\n\n"
                            "{% endif %}"
                        "{% endfor %}"
                    "{% endif %}"
                "{% endif %}"
            "{% endfor %}"
        )
        # fmt: on
