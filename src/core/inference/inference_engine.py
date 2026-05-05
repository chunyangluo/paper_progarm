import torch
from transformers import AutoTokenizer
import logging
from src.core.models.model_definitions import TextCNN, instantiate_bert_textcnn_for_state_dict

logger = logging.getLogger(__name__)

class InferenceEngine:
    """
    推理引擎
    """
    def __init__(self, model_path, model_type, device=None):
        """
        初始化推理引擎
        
        参数:
            model_path: 模型路径
            model_type: 模型类型，可选 'bert_textcnn', 'textcnn'
            device: 运行设备
        """
        self.model_type = model_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.tokenizer = None
        
        if model_type in ['bert_textcnn']:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        
        logger.info(f"推理引擎初始化完成，模型类型: {model_type}，设备: {self.device}")
    
    def _load_model(self, model_path):
        """
        加载模型
        
        参数:
            model_path: 模型路径
        
        返回:
            加载的模型
        """
        try:
            logger.info(f"加载模型: {model_path}")
            
            if self.model_type == 'bert_textcnn':
                from transformers import AutoModel
                bert = AutoModel.from_pretrained("bert-base-chinese")
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                model = instantiate_bert_textcnn_for_state_dict(bert, state_dict).to(self.device)
            elif self.model_type == 'textcnn':
                model = TextCNN(5000).to(self.device)
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
            
            # 加载模型权重
            if self.model_type == 'textcnn':
                model.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict, strict=True)
            model.eval()
            
            logger.info("模型加载成功")
            return model
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def infer(self, input_data, network_features=None):
        """
        单次推理
        
        参数:
            input_data: 输入数据
            network_features: 保留参数，用于兼容旧调用；当前核心识别模型不使用该特征
        
        返回:
            dict: 推理结果
        """
        try:
            with torch.no_grad():
                if self.model_type == 'bert_textcnn':
                    # 处理文本输入
                    inputs = self.tokenizer(input_data, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                elif self.model_type == 'textcnn':
                    # 处理文本特征输入
                    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(self.device)
                    outputs = self.model(input_tensor)
                else:
                    raise ValueError(f"不支持的模型类型: {self.model_type}")
                
                # 计算概率
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                
                # 生成结果
                result = {
                    "probabilities": probabilities.tolist(),
                    "prediction": int(torch.argmax(outputs, dim=1).cpu().numpy()[0]),
                    "confidence": float(probabilities.max())
                }
                
                logger.debug(f"推理完成: {result}")
                return result
        except Exception as e:
            logger.error(f"推理失败: {e}")
            raise
    
    def batch_infer(self, input_data_list, network_features_list=None):
        """
        批量推理
        
        参数:
            input_data_list: 输入数据列表
            network_features_list: 网络特征列表（仅用于多模态模型）
        
        返回:
            list: 推理结果列表
        """
        try:
            results = []
            
            with torch.no_grad():
                if self.model_type == 'bert_textcnn':
                    # 批量处理文本输入
                    inputs = self.tokenizer(input_data_list, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                    
                    for i, prob in enumerate(probabilities):
                        results.append({
                            "probabilities": prob.tolist(),
                            "prediction": int(torch.argmax(outputs[i], dim=0).cpu().numpy()),
                            "confidence": float(prob.max())
                        })
                elif self.model_type == 'multimodal':
                    # 批量处理多模态输入
                    inputs = self.tokenizer(input_data_list, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
                    
                    if network_features_list is None:
                        network_features = torch.zeros(len(input_data_list), 8, device=self.device)
                    else:
                        network_features = torch.tensor(network_features_list, dtype=torch.float32).to(self.device)
                    
                    outputs = self.model(**inputs, network_features=network_features)
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                    
                    for i, prob in enumerate(probabilities):
                        results.append({
                            "probabilities": prob.tolist(),
                            "prediction": int(torch.argmax(outputs[i], dim=0).cpu().numpy()),
                            "confidence": float(prob.max())
                        })
                elif self.model_type == 'textcnn':
                    # 批量处理文本特征输入
                    input_tensor = torch.tensor(input_data_list, dtype=torch.float32).to(self.device)
                    outputs = self.model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                    
                    for i, prob in enumerate(probabilities):
                        results.append({
                            "probabilities": prob.tolist(),
                            "prediction": int(torch.argmax(outputs[i], dim=0).cpu().numpy()),
                            "confidence": float(prob.max())
                        })
                else:
                    raise ValueError(f"不支持的模型类型: {self.model_type}")
            
            logger.info(f"批量推理完成，处理了 {len(input_data_list)} 个样本")
            return results
        except Exception as e:
            logger.error(f"批量推理失败: {e}")
            raise
