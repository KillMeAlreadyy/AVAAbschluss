from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Extra, root_validator
from langchain.embeddings.base import Embeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.utils import get_from_dict_or_env
import tenacity
from tenacity import retry

openai = OpenAIEmbeddings(openai_api_key="")
class OpenAIEmbeddings(BaseModel, Embeddings):
    client = Any
    document_model_name: str = "text-embedding-ada-002"
    query_model_name: str = "text-embedding-ada-002"
    embedding_ctx_length: int = -1

    class Config:

        extra = Extra.forbid

    @root_validator(pre=True)
    def get_model_name(cls, values:Dict) -> Dict:
        if "model_name" in values:
            if "document_model_name" in values:
                raise ValueError(
                    "Both `model_name` and `document_model_name` were provided,"
                    "but only one should be."
                )
            if "query_model_name" in values:
                raise ValueError(
                    "Both `model_name` and `query_model_name` were provided,"
                    "but only one should be."
                )
            model_name = values.pop("model_name")
            values["document_model_name"] = f"text-search-{model_name}-doc-001"
            values["query_model_name"] = f"text-search-{model_name}-query-001"
        return values
    @root_validator()
    def validate_environment(cls, values:Dict) -> Dict:

        openai_api_key = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        try:
            import openai

            openai.api_key = openai_api_key
            values["client"] = openai.Embedding
        except ImportError:
            raise ValueError(
                "Could not import openai python package."
                "Please install it with `pip install openai`"
            )
        return values
    

    def _get_len_safe_embeddings(self, texts: List[str],*, engine:str, chunk_size: int = 1000) -> List[List[float]]:
        embeddings: List[List[float]] = [[]for i in range(len(texts))]
        try:
            import tiktoken

            tokens = []
            indices = []
            encoding = tiktoken.model.encoding_for_model(self.document_model_name)
            for i, text in enumerate(texts):
                text = text.replace("\n", " ")
                token = encoding.encode(text)
                for j in range(0, len(token), self.embedding_ctx_length):
                    tokens += [token[j : j +self.embedding_ctx_length]]
                    indices += [i]

            batched_embeddings = []
            for i in range(0, len(token), chunk_size):
                response = self.client.create(
                    input=tokens[i : i + chunk_size], engine= self.document_model_name
                )
                batched_embeddings += [r["embedding"] for r in response["data"]]

            results: List[List[List[float]]] = [[] for i in range(len(texts))]
            lens: List[List[int]] = [[] for i in range(len(texts))]
            for i in range(len(indices)):
                results[indices[i]].append(batched_embeddings[i])
                lens[indices[i]].append(len(batched_embeddings[i]))

            for i in range(len(texts)):
                average = np.average(results[i], axis=0, weights=lens[i])
                embeddings[i]= (average/ np.linalg.norm(average)).tolist()

            return embeddings
        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package"
                "This is needed in order to fo OpenAIEmbeddings."
                "Please install it with `pip install tiktoken`"
            )
    def _embedding_func(self, text:str,*, engine:str)-> List[float]:
        if self.embedding_ctx_length>0:
            return self._get_len_safe_embeddings([text], engine=engine)[0]
        else:
            text = text.replace("\n", " ")
            return self.client.create(input= [text], engine= engine)["data"][0]["embedding"]
        
        
    def embed_documents(self, texts: List[str], chunk_size: int = 1000) -> List[List[float]]:
        if self.embedding_ctx_length>0:
            return self._get_len_safe_embeddings(
                texts, engine= self.document_model_name, chunk_size= chunk_size
            )
        else:
            @retry(wait=tenacity.wait_fixed(10), stop=tenacity.stop_after_attempt(60))
            def addText(text):
                embedding = self._embedding_func(text, engine=self.document_model_name)
                return embedding
            
            responses = []
            i=0
            filename = "/home/gptbot/flask/logs/runcount.txt"
            for text in texts:
                with open(filename, "w") as f:
                    f.write("Current text number: " + str(i) + "\n")
                i += 1
                responses.append(addText(text))
            
            return responses
        
    def embed_query(self, text: str)-> List[float]:
        embedding = self._embedding_func(text, engine=self.query_model_name)
        return embedding  
