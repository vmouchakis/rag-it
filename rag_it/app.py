from rag_it.model import Llm
from rag_it.config import LLAMA2_7B


def main():
    llm = Llm(model_path=LLAMA2_7B, temperature=0.0)
    response = llm.generate("Hello! Can you tell me a poem about cats and dogs?")
    print(response)


if __name__ == "__main__":
    main()
