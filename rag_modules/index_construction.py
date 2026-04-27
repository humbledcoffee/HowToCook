# 构建向量库索引
# cspell: words FAISS OpenRouter OPENROUTER Qwen tiktoken vectorstore dotenv

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from config import DEFAULT_CONFIG, RAGConfig

# 日志对象实例化
logger = logging.getLogger(__name__)


class IndexConstructionModule:
    """
    索引构建模块。

    负责把上游的数据准备模块产出的 Document 列表向量化，
    并使用 FAISS 构建本地向量索引。
    """

    def __init__(self, config: RAGConfig = DEFAULT_CONFIG) -> None:
        """
        初始化索引构建模块。

        Args:
            config: RAG系统配置对象，包含embedding模型名、OpenRouter地址、索引保存路径等。
        """
        self.config: RAGConfig = config
        self.embeddings: OpenAIEmbeddings = self._create_embeddings()
        self.vector_store: FAISS | None = None

    def _create_embeddings(self) -> OpenAIEmbeddings:
        """
        创建OpenRouter Embedding客户端。

        OpenRouter提供OpenAI兼容接口，所以这里用LangChain的OpenAIEmbeddings，
        但把base_url切到OpenRouter，并使用.env里的OPENROUTER_API_KEY。
        """
        _ = load_dotenv(dotenv_path=".env")
        api_key = os.getenv(self.config.embedding_api_key_env)
        if not api_key:
            raise ValueError(
                f"未找到环境变量 {self.config.embedding_api_key_env}, 请先在.env中配置OpenRouter API Key"
            )

        return OpenAIEmbeddings(
            model=self.config.embedding_model,
            api_key=SecretStr(api_key),
            base_url=self.config.embedding_base_url,
            # Qwen embedding不是OpenAI官方模型，关闭tiktoken检查可避免本地tokenizer误判。
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
        )

    def build_index(self, documents: list[Document]) -> FAISS:
        """
        根据Document列表构建FAISS向量索引。

        Args:
            documents: 待向量化的文档列表，通常来自DataPreparationModule.split_documents()
        Returns:
            构建完成的FAISS向量库对象。
        """
        if not documents:
            raise ValueError("documents不能为空, 请先完成文档加载和分块")

        logger.info(f"开始构建FAISS向量索引, 文档块数量: {len(documents)}")
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        logger.info("FAISS向量索引构建完成")
        return self.vector_store

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """
        相似度搜索。

        Args:
            query: 查询文本，会先被embedding模型转换成查询向量。
            k: 返回结果数量。
        Returns:
            与query语义最相近的文档块列表。
        """
        if self.vector_store is None:
            raise ValueError("请先构建或加载向量索引")
        if not query.strip():
            raise ValueError("query不能为空")
        if k <= 0:
            raise ValueError("k必须大于0")

        return self.vector_store.similarity_search(query, k=k)

    def add_documents(self, new_chunks: list[Document]) -> list[str]:
        """
        向已有FAISS索引追加新的文档块。

        Args:
            new_chunks: 需要追加到索引里的新文档块列表。
        Returns:
            FAISS为新文档生成或记录的文档ID列表。
        """
        if self.vector_store is None:
            raise ValueError("请先构建向量索引, 再追加新文档")
        if not new_chunks:
            raise ValueError("new_chunks不能为空")

        logger.info(f"正在添加 {len(new_chunks)} 个新文档到索引")
        document_ids = self.vector_store.add_documents(new_chunks)
        logger.info("新文档添加完成")
        return document_ids

    def save_index(self, save_path: str | None = None) -> None:
        """
        将FAISS索引保存到本地目录。

        Args:
            save_path: 索引保存路径；不传则使用config.index_save_path。
        """
        if self.vector_store is None:
            raise ValueError("尚未构建向量索引, 请先调用build_index")

        target_path = Path(save_path or self.config.index_save_path)
        target_path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(target_path))
        logger.info(f"FAISS向量索引已保存到: {target_path}")

    def load_index(self, load_path: str | None = None) -> FAISS:
        """
        从本地目录加载FAISS索引。

        Args:
            load_path: 索引加载路径；不传则使用config.index_save_path。
        Returns:
            加载后的FAISS向量库对象。
        """
        source_path = Path(load_path or self.config.index_save_path)
        if not source_path.exists():
            raise FileNotFoundError(f"索引目录不存在: {source_path}")

        self.vector_store = FAISS.load_local(
            str(source_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"FAISS向量索引已从本地加载: {source_path}")
        return self.vector_store
