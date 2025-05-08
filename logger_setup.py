import logging
import os
import datetime

def setup_logger(log_dir='logs', log_name=None):
    """
    配置日志系统，使其可以在所有模块间共享
    
    参数:
        log_dir: 日志保存目录
        log_name: 自定义日志文件名，如果不提供则使用时间戳
        
    返回:
        logger: 配置好的日志记录器
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建带有时间戳的日志文件名
    if log_name is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_name = f'pso_run_{timestamp}'
    log_file = os.path.join(log_dir, f'{log_name}.log')
    
    # 检查是否已经配置过日志记录器
    logger = logging.getLogger()
    
    # 如果已经有处理器，说明已经配置过，不需要重复配置
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info(f"日志已配置，将保存到: {log_file}")
    return logger 