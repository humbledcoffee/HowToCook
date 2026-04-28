# 混合检索优化模块
# cspell: words FAISS BM25 RRF hybrid hygrid metadata rerank reranked top_k

import logging
from typing import cast

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever


logger = logging.getLogger(__name__)


class RetrievalOptimizationModule:
    """
    混合检索优化模块。

    负责在FAISS向量检索的基础上，结合相似度检索和BM25关键词检索结果，
    对检索到的文档使用rrf算法进行重新排序，提升最终返回给生成模块的文档相关性。
    """

    def __init__(self, vectorstores: FAISS, chunks: list[Document]) -> None:
        """
        初始化检索优化模块
        args:
            vectorstores:FAISS向量索引对象
            chunks:原始文档切分后的chunks列表
        """

        self.vectorstores: FAISS = vectorstores
        self.chunks: list[Document] = chunks
        self.setup_retrievers()

    def setup_retrievers(self) -> None:
        """
        初始化相似度检索器和BM25检索器
        """

        # 相似度检索器，基于FAISS向量索引
        self.similarity_retriever: VectorStoreRetriever = (
            self.vectorstores.as_retriever(
                search_type="similarity", search_kwargs={"k": 5}
            )
        )
        # BM25关键字检索器，基于原始文档切分后的chunks列表
        self.bm25_retriever: BM25Retriever = BM25Retriever.from_documents(
            self.chunks, k=5
        )

        logger.info("相似度检索器和BM25检索器初始化完成")

    def hybrid_retrieval(self, query: str, k: int = 5) -> list[Document]:
        """
        混合检索方法，结合相似度检索和BM25检索结果，使用rrf算法重新排序。

        Args:
            query: 用户查询文本
            k: 最终返回的文档数量
        Returns:
            经过混合检索优化后的文档列表
        """
        if not query.strip():
            raise ValueError("query不能为空")
        if k <= 0:
            raise ValueError("k必须大于0")

        # 获取相似度检索结果和BM25检索结果
        similarity_results = self.similarity_retriever.invoke(query)
        bm25_results = self.bm25_retriever.invoke(query)

        # 使用rrf算法对两个结果进行重新排序
        combined_results = self.rrf_rerank(
            similarity_results, bm25_results, top_k=k
        )
        return combined_results

    def hybrid_search(self, query: str, k: int = 5) -> list[Document]:
        """
        兼容老师示例里的方法名。

        当前项目主方法名是 hybrid_retrieval；这里提供 hybrid_search 作为同义入口，
        方便后续代码按“搜索 search”这个命名调用。
        """
        return self.hybrid_retrieval(query, k)

    def hygrid_retrieval(self, query: str, k: int = 5) -> list[Document]:
        """
        兼容旧方法名。

        hybrid 是“混合”的意思；hygrid 是之前的拼写误差。保留这个方法，
        可以避免外部如果已经调用旧名字时直接报错。
        """
        return self.hybrid_retrieval(query, k)

    def metadata_filtered_search(
        self, query: str, filters: dict[str, object], top_k: int = 5
    ) -> list[Document]:
        """
        基于metadata条件过滤的混合检索。

        先通过混合检索召回一批候选文档，再按metadata字段做二次过滤。
        例如 filters={"category": "荤菜", "difficulty": ["简单", "非常简单"]}。

        Args:
            query: 用户查询文本。
            filters: metadata过滤条件。单个值表示必须相等；列表表示命中任一值即可。
            top_k: 最终返回的文档数量。
        Returns:
            满足metadata过滤条件的文档列表。
        """
        if not query.strip():
            raise ValueError("query不能为空")
        if top_k <= 0:
            raise ValueError("top_k必须大于0")

        candidate_docs = self.hybrid_search(query, k=top_k * 2)
        if not filters:
            return candidate_docs[:top_k]

        filtered_docs: list[Document] = []
        for doc in candidate_docs:
            metadata = cast(dict[str, object], doc.metadata)
            if self._metadata_matches_filters(metadata, filters):
                filtered_docs.append(doc)
                if len(filtered_docs) >= top_k:
                    break

        return filtered_docs

    def rrf_rerank(
        self,
        similarity_results: list[Document],
        bm25_results: list[Document],
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> list[Document]:
        """
        Reciprocal Rank Fusion (RRF)算法实现。

        Args:
            similarity_results: 相似度检索结果列表
            bm25_results: BM25检索结果列表
            top_k: 最终返回的文档数量
            rrf_k: RRF算法平滑参数，通常取60，让排名靠后的文档分数衰减更平滑。
        Returns:
            经过RRF重新排序后的文档列表
        """
        if top_k <= 0:
            raise ValueError("top_k必须大于0")
        if rrf_k <= 0:
            raise ValueError("rrf_k必须大于0")

        doc_scores: dict[str, float] = {}
        doc_objects: dict[str, Document] = {}

        retrieval_results = [
            ("similarity", similarity_results),
            ("bm25", bm25_results),
        ]

        for retriever_name, documents in retrieval_results:
            # RRF的rank要在每一路检索结果内部单独计算，不能把两路结果拼接后统一排名。
            for rank, doc in enumerate(documents, start=1):
                doc_id = self._get_document_id(doc, f"{retriever_name}_{rank}")
                doc_objects[doc_id] = doc
                score = 1.0 / (rrf_k + rank)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score

        # 按最终的RRF分数排序
        ranked_doc_scores: list[tuple[str, float]] = sorted(
            doc_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        # 构建最终结果列表，去重并限制数量
        reranked_docs: list[Document] = []
        for doc_id, final_score in ranked_doc_scores:
            doc = doc_objects[doc_id]
            metadata = cast(dict[str, object], doc.metadata)
            metadata["rrf_score"] = final_score
            reranked_docs.append(doc)
            if len(reranked_docs) >= top_k:
                break
        return reranked_docs

    def _get_document_id(self, doc: Document, fallback: str) -> str:
        """
        生成用于去重和累计分数的文档ID。

        数据准备模块里主要生成的是chunk_id，不一定有id字段；如果没有稳定ID，
        就用兜底ID，保证字典key一定是str，类型检查也能明确知道它能作为key使用。
        """
        metadata = cast(dict[str, object], doc.metadata)

        raw_doc_id = (
            metadata.get("chunk_id")
            or metadata.get("id")
            or metadata.get("source")
            or fallback
        )
        return str(raw_doc_id)

    def _metadata_matches_filters(
        self, metadata: dict[str, object], filters: dict[str, object]
    ) -> bool:
        """
        判断一个文档的metadata是否满足所有过滤条件。

        filters里的每一项都是“并且”关系：
        - value是列表时：metadata[key] 在列表里就算匹配；
        - value是单个值时：metadata[key] 必须和 value 完全相等。
        """
        for key, expected_value in filters.items():
            if key not in metadata:
                return False

            actual_value = metadata[key]
            if isinstance(expected_value, list):
                if actual_value not in expected_value:
                    return False
            elif actual_value != expected_value:
                return False

        return True
