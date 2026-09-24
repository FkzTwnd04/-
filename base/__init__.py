"""
初始化基础包的导入环境，并对外暴露配置与日志对象，供业务包复用。
该文件的路径设置用于兼容项目的包导入方式；
它不负责应用初始化，也不应创建数据库或模型实例。
"""

import os           
import sys          

module_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(module_dir)

if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config       
from logger import logger       