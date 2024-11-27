import os
import sys

sys.path.append("../RAG/rain-rag")
from config.ConfigLoader import ConfigLoader
from retrievers.RetrieverCreator import RetrieverCreator
from embeddings.TorchBgeEmbeddings import EmbeddingModelCreator


class RAGManager:
    # TODO: update when env changed
    # TODO: change config info if necessary, such as embedding model PATH, retriever PATH, etc.

    def __init__(
        self,
        base_rag_path: str = "/root/work_dir/huawei-ict-2024/RAG/rain-rag",
        config_path: str = "config/config.yaml",
        embedding_path: str = "",
        collection_name: str = "four_famous",
    ):
        self.base_rag_path = base_rag_path
        self.config_path = config_path
        # TODO: add embedding_path
        self.embedding_path = embedding_path
        self.collection_name = collection_name

        self.init_config()
        self.init_embedding()
        self.init_retriever()

    def init_config(self):
        self.config = ConfigLoader(os.path.join(self.base_rag_path, self.config_path))

    def init_embedding(self):
        embedding_creator = EmbeddingModelCreator(self.config, self.embedding_path)
        self.embedding_model = embedding_creator.create_embedding_model()

    def init_retriever(self):
        vecDB_path = self.base_rag_path + self.config.get("vector_db.index_path")
        self.retriever = RetrieverCreator(
            self.config,
            self.embedding_model,
            vecDB_path,
            collection_name=self.collection_name,
        ).create_retriever()

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
        self, source: str = "西游记", role: str = "孙悟空", query: str = ""
    ):
        if query is None and len(query) == 0:
            return None

        retrieved_docs = self.retriever.invoke(query)
        retrieved_info = self._format_docs(retrieved_docs, None)

        return retrieved_info
