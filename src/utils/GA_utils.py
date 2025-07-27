import logging
import os

def setup_logger(log_path='logs/global.log', level=logging.INFO):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logger = logging.getLogger('StructureGA')
    logger.setLevel(level)

    # 避免重复添加 handler
    if not logger.handlers:
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')

        # 文件输出
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)

        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
