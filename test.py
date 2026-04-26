# 测试各个代码模块

# 导入日志打印模块
import logging
from venv import logger

#导入自定义RAG配置类和默认配置
from config import RAGConfig, DEFAULT_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

# 创建日志打印对象
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    # 测试配置类
    logger.info(f"测试RAGConfig配置类:{DEFAULT_CONFIG.data_path}")