from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from typing import Generator


class Llm:
    def __init__(self, model_path, temperature, max_new_tokens=256, context_window=3900, verbose=False) -> None:
        self.model = self.init_model(model_path=model_path,
                                     temperature=temperature,
                                     max_new_tokens=max_new_tokens,
                                     context_window=context_window,
                                     verbose=verbose)

    def init_model(self, model_path, temperature, max_new_tokens, context_window, verbose):
        return LlamaCPP(
            model_path=model_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            context_window=context_window,
            generate_kwargs={},
            # set to at least 1 to use GPU
            model_kwargs={"n_gpu_layers": 1},
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            verbose=verbose,
        )

    def generate(self, prompt: str) -> str:
        response = self.model.complete(prompt)
        return response.text

    def generate_stream(self: str, prompt: str) -> Generator:
        response = self.model.stream_complete(prompt)
        return response