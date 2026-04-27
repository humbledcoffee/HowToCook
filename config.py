# rag系统配置文件

from dataclasses import dataclass
from typing import Any

@dataclass
class RAGConfig:
    """
    RAG系统配置类

    """
    data_path: str = "./data/dishes"  # 菜谱存储路径

    index_save_path: str = "./vector_index"  # 向量库索引保存路径
    embedding_model: str = "qwen/qwen3-embedding-4b"  # 向量化模型名称
    embedding_base_url: str = "https://openrouter.ai/api/v1"  # OpenRouter兼容OpenAI协议的API地址
    embedding_api_key_env: str = "OPENROUTER_API_KEY"  # 从.env中读取OpenRouter API Key的环境变量名
    retriever_top_k: int = 3  # 检索返回的top_k文档数量

    llm_model: str = "qwen/qwen3.6-plus"  # LLM模型名称
    temperature: float = 0.1  # LLM生成文本的温度参数
    max_tokens: int = 2048  # LLM生成文本的最大长度

    def __post_init__(self):
        """
        初始化后处理，可以添加一些验证逻辑
        """
        if self.retriever_top_k <= 0:
            raise ValueError("retriever_top_k必须大于0")
        if not (0.0 <= self.temperature <= 1.0):
            raise ValueError("temperature必须在0.0和1.0之间")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens必须大于0")
        if not self.embedding_model:
            raise ValueError("embedding_model不能为空")
        if not self.embedding_base_url:
            raise ValueError("embedding_base_url不能为空")
        if not self.embedding_api_key_env:
            raise ValueError("embedding_api_key_env不能为空")

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "RAGConfig":
        """
        从字典创建RAGConfig实例

        :param config_dict: 配置字典
        :return: RAGConfig实例
        """
        return cls(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        """
        将配置转换为字典格式

        :return: 配置字典
        """
        return {
            "data_path": self.data_path,
            "index_save_path": self.index_save_path,
            "embedding_model": self.embedding_model,
            "embedding_base_url": self.embedding_base_url,
            "embedding_api_key_env": self.embedding_api_key_env,
            "retriever_top_k": self.retriever_top_k,
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

DEFAULT_CONFIG   = RAGConfig()
