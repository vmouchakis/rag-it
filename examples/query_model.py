from rag_it.model import Llm, EmbedModel
from rag_it.query_engine import Engine
from rag_it.config import LLAMA2_7B_PATH
import logging


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Loading model")
    llm = Llm(model_path=LLAMA2_7B_PATH, temperature=0.0)
    logging.info("Model loaded")
    response = llm.generate("Write a poem about Panathinaikos.")
    logging.info(response)
