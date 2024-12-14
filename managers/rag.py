import os
import sys
from typing import Dict

# sys.path.append("../RAG/rainRAG")
# from config.ConfigLoader import ConfigLoader
# from retrievers.RetrieverCreator import RetrieverCreator
# from embeddings.TorchBgeEmbeddings import EmbeddingModelCreator

from RAG import ConfigLoader, EmbeddingModelCreator, RetrieverCreator


class RAGManager:
    def __init__(self, rag_config: Dict):
        self.rag_config = rag_config

        self.init_config()
        self.init_embedding()
        self.init_retriever()

    def init_config(self):
        self.config = ConfigLoader(self.rag_config)

    def init_embedding(self):
        embedding_creator = EmbeddingModelCreator(self.config)
        self.embedding_model = embedding_creator.create_embedding_model()

    def init_retriever(self):
        self.retri_creator = RetrieverCreator(
            self.config, self.embedding_model, self.config.get("vector_db.index_path")
        )

    # format RAG retrieval results
    def _format_docs(self, docs, wiki_docs=None):
        ans = "从古籍中检索到的信息如下：\n\n"
        for id, doc in enumerate(docs):
            ans += f"{id+1}. {doc.page_content}\n\n"
        if wiki_docs is not None:
            ans += "从维基百科中检索到的信息如下：\n\n"
            ans += f'{len(docs)+1}. {wiki_docs[0].metadata["summary"]}\n\n'
        # print(f'检索到的信息有：{ans}')
        return ans

    def get_RAG_prompt(
        self,
        source: str = "西游记",
        character: str = "孙悟空",
        query: str = "",
        **kwargs,
    ):
        if query is None and len(query) == 0:
            return None

        template_retrieved = "在{source}中, 针对{character}这个角色的提问：{query}"
        retrieved_dict = {"source": source, "character": character, "query": query}

        search_type = self.config.get(
            "langchain_modules.retrievers.vector_retriever.retrieval_type",
            "similarity",
        )
        search_kwargs = self.config.get(
            "langchain_modules.retrievers.vector_retriever.search_kwargs", {}
        )
        search_kwargs["k"] = kwargs.get("k", search_kwargs["k"])

        # reset retriever with new search_kwargs
        retriever = self.retri_creator.vector_store.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

        retrieved_docs = retriever.invoke(template_retrieved.format(**retrieved_dict))

        retrieved_info = self._format_docs(retrieved_docs, None)

        return retrieved_info
