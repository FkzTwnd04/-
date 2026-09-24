"""
加载本地中文 BERT 序列分类模型，区分通用知识与专业咨询，并提供数据预处理、训练、验证和单条
预测能力。
它属于“模型直接预测分类标签”的方式，不是基于历史代表问句的检索式意图识别。
后续可用同一标注集与 BM25+向量方案比较准确率、延迟和调用成本。
"""

import json
import os

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments

from base.config import config
from base.logger import logger

class QueryClassifier:
    
    def __init__(self, model_path=None):
        """
        :param model_path: 已训练模型的路径, 默认取 rag_qa/models/bert_query_classifier
        """
        
        self.model_path = model_path or os.path.join(config.MODELS_DIR, 'bert_query_classifier')

        
        bert_path = os.path.join(config.MODELS_DIR, 'bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        
        self.model = None

        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'使用设备: {self.device}')

        
        self.label_map = {"通用知识": 0, "专业咨询": 1}

        
        self.load_model()

    
    def load_model(self):
        """
        函数功能: 若 model_path 存在则加载已训练的分类模型, 否则基于 bert-base-chinese 初始化二分类新模型, 并移动到当前设备.
        :return: None
        """
        base_bert_path = os.path.join(config.MODELS_DIR, 'bert-base-chinese')
        if os.path.exists(self.model_path):
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            logger.info(f"加载模型: {self.model_path}")
        else:
            self.model = BertForSequenceClassification.from_pretrained(base_bert_path, num_labels=2)
            self.model.to(self.device)
            logger.info("初始化新 BERT 模型")

    
    def save_model(self):
        """
        函数功能: 将当前模型和分词器保存到 model_path.
        :return: None
        """
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info(f"保存模型至: {self.model_path}")

    
    def preprocess_data(self, texts, labels):
        """
        函数功能: 对文本批量分词编码(截断/填充, 最大长度 128), 并将中文标签映射为整数标签.
        :param texts: 文本列表
        :param labels: 中文标签列表('通用知识' 或 '专业咨询')
        :return: (分词编码结果, 整数标签列表)
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        return encodings, [self.label_map[label] for label in labels]

    
    def create_dataset(self, encodings, labels):
        """
        函数功能: 将分词编码和整数标签封装为可供 Trainer 使用的 torch Dataset.
        :param encodings: 分词器输出的编码结果
        :param labels: 整数标签列表
        :return: torch.utils.data.Dataset 实例
        """
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                """
                函数功能: 保存分词编码和标签.
                :param encodings: 分词器输出的编码结果
                :param labels: 整数标签列表
                :return: None
                """
                super().__init__()
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                """
                函数功能: 取出第 idx 条样本的各编码字段, 并附加 labels 张量.
                :param idx: 样本索引
                :return: 包含编码字段和 labels 的字典
                """
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                """
                函数功能: 返回样本数量.
                :return: 标签列表长度
                """
                return len(self.labels)

        return Dataset(encodings, labels)

    
    def train_model(self, data_file="training_dataset_hybrid_5000.json"):
        """
        函数功能: 读取 JSON Lines 数据集, 按 8:2 划分训练/验证集, 用 Trainer 训练 BERT 分类模型, 训练后保存模型并在验证集上评估.
        数据集文件不存在时记录错误并抛出 FileNotFoundError.
        :param data_file: 数据集文件路径, 每行一个含 query 和 label 字段的 JSON
        :return: None
        """
        if not os.path.exists(data_file):
            logger.error(f"数据集文件 {data_file} 不存在")
            raise FileNotFoundError(f"数据集文件 {data_file} 不存在")

        with open(data_file, "r", encoding="utf-8") as f:
            data = [json.loads(value) for value in f.readlines()]

        texts = [item["query"] for item in data]
        labels = [item["label"] for item in data]

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
        val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)

        train_dataset = self.create_dataset(train_encodings, train_labels)
        val_dataset = self.create_dataset(val_encodings, val_labels)

        training_args = TrainingArguments(
            output_dir="./bert_results",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=20,
            weight_decay=0.01,
            logging_dir="./bert_logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="eval_loss",
            fp16=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )
        logger.info("开始训练 BERT 模型...")
        trainer.train()
        self.save_model()
        self.evaluate_model(val_texts, val_labels)

    
    def compute_metrics(self, eval_pred):
        """
        函数功能: 根据 logits 取 argmax 得到预测标签, 计算准确率, 供 Trainer 评估使用.
        :param eval_pred: (logits, labels) 元组
        :return: {"accuracy": 准确率}
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    
    def evaluate_model(self, texts, labels):
        """
        函数功能: 对给定文本进行批量预测, 并通过日志输出分类报告和混淆矩阵.
        :param texts: 待评估文本列表
        :param labels: 对应的整数标签列表
        :return: None
        """
        encodings = self.tokenizer(
            texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
        )
        dataset = self.create_dataset(encodings, labels)
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(dataset)
        pred_labels = np.argmax(predictions.predictions, axis=-1)

        logger.info("分类报告:")
        logger.info(classification_report(true_labels := labels, pred_labels, target_names=["通用知识", "专业咨询"]))
        logger.info("混淆矩阵:")
        logger.info(confusion_matrix(true_labels, pred_labels))

    
    def predict_category(self, query):
        """
        函数功能: 判断单条查询属于'通用知识' 还是 '专业咨询'.
        :param query: 查询文本
        :return: '通用知识' 或 '专业咨询'
        """
        if self.model is None:
            logger.error("模型未训练或加载!")
            return '通用知识'

        encoding = self.tokenizer(
            query, truncation=True, padding=True, max_length=128, return_tensors="pt"
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        return '专业咨询' if prediction == 1 else '通用知识'

if __name__ == '__main__':
    query_classify = QueryClassifier()
    
    print(query_classify.predict_category('大数据分析师需要考取哪些证书'))
