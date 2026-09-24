"""
使用文本分段模型辅助确定长文的语义边界，再把原始文档组织为下游检索可用的片段。
相比固定字符长度，模型切分可能保留更完整的主题边界，但会增加模型加载和处理成本，且需通过评测确认
切分质量。
"""

import warnings
warnings.filterwarnings("ignore")
from langchain_text_splitters import CharacterTextSplitter
import re
from typing import List
from modelscope.pipelines import pipeline

class AliTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        """
        函数功能: 初始化模型切分器，将其余参数传给 CharacterTextSplitter 并记录是否按 PDF 文本预处理.
        :param pdf: 为 True 时在切分前先清理 PDF 文本中的多余换行和空白
        :param kwargs: 传给 CharacterTextSplitter 的其他参数
        :return: None
        """
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        """
        函数功能: 使用本地 ModelScope 中文文档分段模型(CPU)对文本做语义分段，pdf 为 True 时先规范化换行和空白.
        :param text: 待切分的原始文本
        :return: 按模型输出的 "\\n\\t" 分隔并去除空串后的文本片段列表
        """
        
        
        
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)
        from base.config import Config
        p = pipeline(
            task="document-segmentation",
            model=f'{Config().MODELS_DIR}/nlp_bert_document-segmentation_chinese-base',
            device="cpu")
        result = p(documents=text)
        sent_list = [i for i in result["text"].split("\n\t") if i]
        return sent_list
if __name__ == '__main__':
    model_split = AliTextSplitter()
    result = model_split.split_text(text='移动端语音唤醒模型，检测关键词为“小云小云”。模型主体为4层FSMN结构，使用CTC训练准则，参数量750K，适用于移动端设备运行。模型输入为Fbank特征，输出为基于char建模的中文全集token预测，测试工具根据每一帧的预测数据进行后处理得到输入音频的实时检测结果。模型训练采用“basetrain + finetune”的模式，basetrain过程使用大量内部移动端数据，在此基础上，使用1万条设备端录制安静场景“小云小云”数据进行微调，得到最终面向业务的模型。后续用户可在basetrain模型基础上，使用其他关键词数据进行微调，得到新的语音唤醒模型，但暂时未开放模型finetune功能。')
    print(result)