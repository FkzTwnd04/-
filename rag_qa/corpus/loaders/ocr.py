"""
封装 OCR 相关依赖和文字识别入口，为图片或扫描文档加载器提供共享能力。
输入是页面图像或图片文件，输出是识别出的文本；
它不决定文档切分粒度、不保存文档记录，也不访问向量库。
"""

from typing import TYPE_CHECKING
'''
paddleocr：解析图片中的文字，也可以进行表格识别
rapidocr_paddle 和 rapidocr_onnxruntime 两种导入方式
主要区别在于它们所使用的推理引擎和硬件支持
选择哪种方式最合适取决于你的硬件环境和性能需求。
当你有 GPU 且追求速度时：使用 rapidocr_paddle。PaddlePaddle 原生支持在 GPU 上推理 PaddleOCR 模型，速度更快。
当只有 CPU 且需要高效推理时：使用 rapidocr_onnxruntime。它在 CPU 上进行了优化，资源占用较低.

注意: OCR不是万能的, 如果比较模糊的图片, 它识别效果也不是特别OK, 可以借助一些多模态大模型来处理. 
'''

def get_ocr(use_cuda: bool = True) -> "RapidOCR":
    try:
        from rapidocr_paddle import RapidOCR
        '''
        det_use_cuda=True：启用检测模型的GPU加速。cls_use_cuda=True：启用分类模型的GPU加速。rec_use_cuda=True：启用识别模型的GPU加速。
        '''
        ocr = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
    except ImportError:
        
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    return ocr
