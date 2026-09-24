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
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info(f"保存模型至: {self.model_path}")

    
    def preprocess_data(self, texts, labels):
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        return encodings, [self.label_map[label] for label in labels]

    
    def create_dataset(self, encodings, labels):
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                super().__init__()
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        return Dataset(encodings, labels)

    
    def train_model(self, data_file="training_dataset_hybrid_5000.json"):
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
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    
    def evaluate_model(self, texts, labels):
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
