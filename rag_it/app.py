from rag_it.model import Llm, EmbedModel
from rag_it.query_engine import Engine
from rag_it.config import LLAMA2_7B_PATH
import logging


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Loading models")
    llm = Llm(model_path=LLAMA2_7B_PATH, temperature=0.0)
    embed_model = EmbedModel()
    logging.info("Models loaded")
    engine = Engine(llm, embed_model, 'lotr', 'samples')
    resp = engine.query("Who directed the Lord of the Rings trilogy?")
    logging.info(resp)
