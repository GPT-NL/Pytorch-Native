import json
import os
from typing import List

from tqdm import tqdm
from typer import Option, Typer
from typing_extensions import Annotated
from vllm import LLM, SamplingParams

app = Typer(pretty_exceptions_enable=False)


class LLMProcessing:
    prompt: str
    model_name: str
    model: LLM

    def __init__(self, model_name: str, prompt: str):
        """
        The prompt should only have a template variable called {text}
        """
        self.model_name = model_name
        self.prompt = prompt
        self.vllm_model_params = self.get_vllm_model_params()
        self.model = LLM(**self.vllm_model_params)

    def process_texts(
        self,
        texts: List[str],
        max_tokens: int,
        temperature: float,
        use_tqdm: bool = True,
        print_prompt: bool = False,
        use_chat_template: bool = False,
        max_context_size: int = 16384,
    ):
        # cut texts that are too long
        char_max_size = max_context_size // 3  # leave some room for prompt and chat template, so don't do //4
        if any(len(t) > char_max_size for t in texts):
            print("WARN: some texts will be trimmed because of max_context_size!")
        texts = [t[:char_max_size] for t in texts]

        # now format according to prompt
        input_texts = [self.prompt.format(text=text) for text in texts]
        if print_prompt:
            print(input_texts)
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            # repetition_penalty=1.5,
            # frequency_penalty=0.5,
            # presence_penalty=0.3,
            # min_p=0.1,
            # top_p=0.9,
            # seed=40,
            temperature=temperature,
            skip_special_tokens=False,
        )
        if use_chat_template:
            conversations = [[{"role": "user", "content": t}] for t in input_texts]
            raw_outputs = self.model.chat(
                messages=conversations,
                sampling_params=sampling_params,
                use_tqdm=use_tqdm,
            )
        else:
            raw_outputs = self.model.generate(input_texts, sampling_params, use_tqdm=use_tqdm)
        outputs = []
        for o in raw_outputs:
            print(len(o.outputs))
            print(o.outputs)
            first = o.outputs[0]
            # outputs.append("".join(chunk.text for chunk in o.outputs))
            outputs.append({"text": first.text, "tokens": first.token_ids})

        return outputs

    def get_vllm_model_params(self) -> dict:
        # manage revision
        if os.path.exists(self.model_name):
            return {"model": self.model_name}
        else:
            # try with HF Hub model@revision
            parts = self.model_name.split("@")
            if len(parts) == 2:
                return {"model": parts[0], "revision": parts[1]}
            else:
                raise ValueError(self.model_name)


@app.command()
def main(
    model: Annotated[str, Option()],
    input_file: Annotated[str, Option()],
    output_file: Annotated[str, Option()],
    max_tokens: int = 20,
    temperature: float = 0.9,
):
    with open(input_file) as f:
        input_data = json.load(f)
    processor = LLMProcessing(model_name=model, prompt="{text}")
    final_outputs = {}
    for group, input_texts in tqdm(input_data.items()):
        outputs = processor.process_texts(texts=input_texts, max_tokens=max_tokens, temperature=temperature)
        output_content = [
            {
                "input_text": input_text,
                "output_text": output["text"],
                "output_tokens": output["tokens"],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            for input_text, output in zip(input_texts, outputs)
        ]
        final_outputs[group] = output_content
    with open(output_file, "w") as f:
        json.dump(final_outputs, f, indent=2)


if __name__ == "__main__":
    app()
