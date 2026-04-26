# RAG知识库数据准备模块

# 导入打印模块
import logging

# 根据给定的内容返回hash值
import hashlib

# 导入路径解析
from os import path
from pathlib import Path

# 导入数据类型
from pydoc import doc
from sre_compile import CATEGORY
from dataclasses import dataclass, field
from typing import ClassVar, cast

# 导入Markdown文档分割器
from langchain_text_splitters import MarkdownHeaderTextSplitter, MarkdownTextSplitter
from langchain_core.documents import Document

# 引入唯一ID
import uuid


# 创建日志打印对象
logger = logging.getLogger(__name__)


# 创建数据准备类
@dataclass
class DataPreparationModule:
    """
    数据准备模块 - 负责数据加载,清洗和预处理
    ARGS:
        data_path: str - 数据存储路径
    """

    # 统一维护的分类,供外部复用,避免关系词重复定义
    CATEGORY_MAPPING: ClassVar[dict[str, str]] = {
        "meat_dish": "荤菜",
        "vegetable_dish": "素菜",
        "soup": "汤品",
        "dessert": "甜品",
        "breakfast": "早点",
        "staple": "主食",
        "aquatic": "水产",
        "condiment": "调料",
        "drink": "饮品",
    }

    # 中文类别标签
    CATEGORY_LABELS: ClassVar[list[str]] = list(set(CATEGORY_MAPPING.values()))

    # 菜谱难度等级列表
    DIFFICULTY_LEVELS: ClassVar[list[str]] = [
        "非常简单",
        "简单",
        "中等",
        "困难",
        "非常困难",
    ]

    # 初始化数据准备模块
    data_path: str
    documents: list[Document] = field(default_factory=list)
    chunks: list[Document] = field(default_factory=list)
    parent_child_map: dict[str, str] = field(default_factory=dict)

    def load_documents(self) -> list[Document]:
        """
        加载文档数据库
        returns:
            加载的文档列表
        """
        logger.info(f"开始加载文档数据库,数据路径: {self.data_path}")

        # 加载 data/dishes 目录下所有的Markdown 格式菜谱文档
        documents: list[Document] = []
        data_path_obj = Path(self.data_path)
        for md_file in data_path_obj.rglob("*.md"):
            try:
                # 读取Markdown文件内容
                with md_file.open("r", encoding="utf-8") as f:
                    content = f.read()
                    # 为每个文档生成唯一ID
                    try:
                        # 获取数据的绝对路径
                        data_root_path = data_path_obj.resolve()
                        # 解析每个Markdown菜谱文件相对于数据根目录的相对路径
                        relative_path = (
                            md_file.resolve().relative_to(data_root_path).as_posix()
                        )
                        pass
                    except Exception:
                        relative_path = md_file.as_posix()
                        continue
                    parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()
                    # 创建document对象
                    document = Document(
                        page_content=content,
                        metadata={
                            "source": relative_path,
                            "parent_id": parent_id,
                            "doc_type": "parent",  # 标记为未分割的父文档,后续分块后会生成子文档,子文档会带上parent_id关联到父文档
                        },
                    )
                    document = self._enhance_metadata(document)
                    documents.append(document)
            except Exception as e:
                logger.error(f"加载文档失败: {md_file}, 错误信息: {e}")

        self.documents = documents
        logger.info(f"成功加载 {len(documents)} 个文档")
        return documents

    def _enhance_metadata(self, document: Document) -> Document:
        """
        增强文档的元数据,提取菜谱的类别和难度等级等信息
        args:
            document: Document - 需要增强元数据的文档对象
        """
        metadata = cast(dict[str, object], document.metadata)
        file_path = Path(str(metadata.get("source", "")))
        # 路径拆解
        path_parts = file_path.parts
        # 提取菜品分类
        metadata["category"] = "其他"
        for key, value in self.CATEGORY_MAPPING.items():
            if key in path_parts:
                metadata["category"] = value
                break
        # 提取菜谱名称
        metadata["dish_name"] = file_path.stem
        # 提取难度等级
        if "★★★★★" in document.page_content:
            metadata["difficulty"] = "非常困难"
        elif "★★★★" in document.page_content:
            metadata["difficulty"] = "困难"
        elif "★★★" in document.page_content:
            metadata["difficulty"] = "中等"
        elif "★★" in document.page_content:
            metadata["difficulty"] = "简单"
        elif "★" in document.page_content:
            metadata["difficulty"] = "非常简单"
        else:
            metadata["difficulty"] = "未知"
        return document

    def split_documents(self) -> list[Document]:
        """
        Markdown结构感知分块

        returns:
        分块后的文档列表
        """
        logger.info("开始进行Markdown结构感知分块")
        if not self.documents:
            raise ValueError(
                "没有加载的文档可供分块,请先调用 load_documents 方法加载文档"
            )
        chunks = self._markdown_header_split()
        for i,chunk in enumerate(chunks):
            metadata = cast(dict[str, object], chunk.metadata)
            if "chunk_id" not in metadata:
                # 如果没有chunk_id,说明这个chunk是分块失败的文档,我们为它生成一个唯一的chunk_id,并且parent_id指向自己
                chunk_id = str(uuid.uuid4())
                metadata["chunk_id"] = chunk_id
            metadata["batch_index"] = i
            metadata["chunk_size"] = len(chunk.page_content)
        self.chunks = chunks
        logger.info(f"Markdown结构感知分块完成,共生成 {len(self.chunks)} 个子文档")
        return self.chunks

    def _markdown_header_split(self) -> list[Document]:
        """
        根据Markdown标题进行分块
        returns:
            分块后的文档列表
        """
        # 定义要分割的标题层级
        headers_to_split_on = [
            ("#", "主标题"),
            ("##", "二级标题"),
            ("###", "三级标题"),
        ]
        # 创建MarkdownTextSplitter对象,指定分割规则
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,  # 是否在分块内容中去除标题文本
        )
        # 存储分割后的文档快
        all_chunks: list[Document] = []
        for document in self.documents:
            metadata = cast(dict[str, object], document.metadata)
            try:
                # 检查文档是否包含markdown标题
                content_preview = document.page_content[:100]  # 预览前100字符
                has_headers = any(
                    line.strip().startswith("#") for line in content_preview.split("\n")
                )
                if not has_headers:
                    logger.warning(
                        f"文档 {metadata.get('source', '未知')} 可能不包含Markdown标题,跳过分块"
                    )
                    logger.debug(f"文档内容预览: {content_preview}")
                    continue
                md_chunks: list[Document] = markdown_splitter.split_text(
                    document.page_content
                )
                if len(md_chunks) == 1:
                    logger.warning(
                        f"文档 {metadata.get('source', '未知')} 分块后只有一个chunk,可能分块效果不理想"
                    )
                else:
                    logger.debug(
                        f"正在分块文档: {metadata.get('dish_name', '未知')} 分割成{len(md_chunks)}"
                    )
                # 为每个子文档建立与付文档的关联关系
                for i, chunk in enumerate(md_chunks):
                    # 为每个chunk生成唯一ID
                    chunk_id = str(uuid.uuid4())
                    parent_id = str(metadata.get("parent_id", ""))
                    chunk_metadata = cast(dict[str, object], chunk.metadata)
                    chunk_metadata.update(
                        {
                            "chunk_id": chunk_id,  # 生成唯一的chunk_id
                            "parent_id": parent_id,  # 关联到父文档的ID
                            "doc_type": "child",
                            "chunk_index": i,  # 记录chunk在父文档中的顺序
                        }
                    )

                    # 更新父子映射表,后续检索到子块后可回溯父文档
                    self.parent_child_map[chunk_id] = parent_id
                    all_chunks.extend(md_chunks)
            except Exception:
                # 如果分块过程中发生错误,记录日志并将整个文档作为一个chunk保留,避免丢失数据
                logger.error(f"文档分块失败: {metadata.get('source', '未知')}")
                all_chunks.append(document)
        return all_chunks

    @classmethod
    def get_category_labels(cls) -> list[str]:
        """
        获取菜谱分类标签列表
        returns:
            菜谱分类标签列表
        """
        return cls.CATEGORY_LABELS

    @classmethod
    def get_difficulty_levels(cls) -> list[str]:
        """
        获取菜谱难度等级列表
        returns:
            菜谱难度等级列表
        """
        return cls.DIFFICULTY_LEVELS
