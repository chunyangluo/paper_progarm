import torch
import torch.nn as nn

class TextCNN(nn.Module):
    """
    TextCNN模型
    """
    def __init__(self, vocab_size, embed_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Linear(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(1, 32, 2)
        self.conv2 = nn.Conv1d(1, 32, 3)
        self.conv3 = nn.Conv1d(1, 32, 4)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x1 = self.pool(torch.relu(self.conv1(x))).squeeze(-1)
        x2 = self.pool(torch.relu(self.conv2(x))).squeeze(-1)
        x3 = self.pool(torch.relu(self.conv3(x))).squeeze(-1)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.dropout(x)
        return self.fc(x)

class BERTTextCNN(nn.Module):
    """
    BERT-TextCNN混合模型
    """
    def __init__(self, bert_model):
        super().__init__()
        # 特征提取层
        # BERT语义特征提取（全局语义）
        self.bert = bert_model
        # TextCNN局部特征提取
        self.conv1 = nn.Conv1d(768, 64, 2)  # 2-gram
        self.conv2 = nn.Conv1d(768, 64, 3)  # 3-gram
        self.conv3 = nn.Conv1d(768, 64, 4)  # 4-gram
        self.conv_bn = nn.BatchNorm1d(64)
        self.cls_proj = nn.Sequential(
            nn.Linear(768, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.cnn_proj = nn.Sequential(
            nn.Linear(64 * 3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # 分类层
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # BERT语义特征提取
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = out.last_hidden_state  # [batch_size, seq_len, 768]
        cls_features = self.cls_proj(x[:, 0, :])
        
        # TextCNN局部特征提取
        x = x.transpose(1, 2)  # [batch_size, 768, seq_len]
        x1 = torch.relu(self.conv_bn(self.conv1(x))).max(dim=2)[0]  # 2-gram特征
        x2 = torch.relu(self.conv_bn(self.conv2(x))).max(dim=2)[0]  # 3-gram特征
        x3 = torch.relu(self.conv_bn(self.conv3(x))).max(dim=2)[0]  # 4-gram特征
        
        # 特征融合
        cnn_features = self.cnn_proj(torch.cat([x1, x2, x3], dim=1))
        out = torch.cat([cls_features, cnn_features], dim=1)
        out = self.dropout(out)
        
        # 分类
        return self.fc(out)


class BERTTextCNNLegacy(nn.Module):
    """
    与早期 `bert_textcnn_best.pth`（仅 TextCNN 池化 + fc(192→2)）权重兼容的结构。
    部署目录中的旧检查点仍使用该布局；新训练产物为 `BERTTextCNN`（[CLS]+CNN 融合，fc 256→2）。
    """
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.conv1 = nn.Conv1d(768, 64, 2)
        self.conv2 = nn.Conv1d(768, 64, 3)
        self.conv3 = nn.Conv1d(768, 64, 4)
        self.conv_bn = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(64 * 3, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = out.last_hidden_state
        x = x.transpose(1, 2)
        x1 = torch.relu(self.conv_bn(self.conv1(x))).max(dim=2)[0]
        x2 = torch.relu(self.conv_bn(self.conv2(x))).max(dim=2)[0]
        x3 = torch.relu(self.conv_bn(self.conv3(x))).max(dim=2)[0]
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.dropout(out)
        return self.fc(out)


def instantiate_bert_textcnn_for_state_dict(bert_model, state_dict: dict) -> nn.Module:
    fc_w = state_dict.get("fc.weight")
    if fc_w is None:
        raise ValueError("state_dict missing fc.weight")
    shape = tuple(fc_w.shape)
    if shape == (2, 256):
        return BERTTextCNN(bert_model)
    if shape == (2, 192):
        return BERTTextCNNLegacy(bert_model)
    raise ValueError(f"Unsupported BERT-TextCNN fc.weight shape: {shape}")


class CrossModalAttention(nn.Module):
    """
    跨模态注意力融合模块
    """
    def __init__(self, text_dim, network_dim, hidden_dim):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.network_proj = nn.Linear(network_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, text_feat, network_feat):
        # 维度适配：(batch, seq_len, dim)
        text_feat = self.text_proj(text_feat).unsqueeze(1)  # (B,1,H)
        network_feat = self.network_proj(network_feat).unsqueeze(1)  # (B,1,H)
        
        # 交叉注意力：文本作为Query，网络特征作为Key/Value
        attn_output, _ = self.attention(text_feat, network_feat, network_feat)
        
        # 残差连接
        text_feat = self.layer_norm(text_feat + attn_output)
        
        # 融合输出
        fusion_feat = torch.cat([text_feat.squeeze(1), network_feat.squeeze(1)], dim=1)
        return fusion_feat

class MultimodalModel(nn.Module):
    """
    多模态融合模型
    """
    def __init__(self, bert_model, url_feature_dim=16, network_feature_dim=8):
        super().__init__()
        # 特征提取层
        # 1. BERT语义特征提取（全局语义）
        self.bert = bert_model
        # 2. TextCNN局部特征提取（局部关键特征）
        self.conv1 = nn.Conv1d(768, 64, 2)  # 2-gram
        self.conv2 = nn.Conv1d(768, 64, 3)  # 3-gram
        self.conv3 = nn.Conv1d(768, 64, 4)  # 4-gram
        self.conv_bn = nn.BatchNorm1d(64)
        
        # 3. URL特征处理
        self.url_encoder = nn.Sequential(
            nn.Linear(url_feature_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # 4. 网络行为特征处理（使用真实网络特征）
        # 轻量级子网络，提升特征表征能力
        self.network_encoder = nn.Sequential(
            nn.Linear(network_feature_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # 5. 将 URL+网络特征融合成统一模态表示
        self.modal_encoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 6. 文本特征编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(64 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # 7. 跨模态注意力融合
        self.cross_attention = CrossModalAttention(128, 128, 256)

        # 8. 模态自适应权重（文本、URL+网络融合模态、跨模态交互）
        self.modal_weights = nn.Parameter(torch.ones(3))

        # 9. 多模态融合层
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128 + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # 10. 分类层：采用全连接层+Softmax激活函数
        self.classification = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
            # 注意：在训练时使用CrossEntropyLoss，它会自动应用Softmax
        )

    def forward(self, input_ids, attention_mask, url_features=None, network_features=None, token_type_ids=None):
        # 1. 文本特征提取
        # BERT语义特征提取（全局语义）
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state  # [batch_size, seq_len, 768]
        
        # TextCNN局部特征提取（局部关键特征）
        x = x.transpose(1, 2)  # [batch_size, 768, seq_len]
        x1 = torch.relu(self.conv_bn(self.conv1(x))).max(dim=2)[0]  # 2-gram特征
        x2 = torch.relu(self.conv_bn(self.conv2(x))).max(dim=2)[0]  # 3-gram特征
        x3 = torch.relu(self.conv_bn(self.conv3(x))).max(dim=2)[0]  # 4-gram特征
        text_features = torch.cat([x1, x2, x3], dim=1)  # 文本特征向量 [batch_size, 192]
        
        # 文本特征编码
        text_encoded = self.text_encoder(text_features)  # [batch_size, 128]
        
        # 2. URL特征处理
        if url_features is not None:
            url_encoded = self.url_encoder(url_features)  # [batch_size, 128]
        else:
            url_encoded = torch.zeros(text_encoded.size(0), 128, device=text_encoded.device)

        # 3. 网络行为特征处理
        if network_features is not None:
            network_encoded = self.network_encoder(network_features)  # 网络行为特征向量 [batch_size, 128]
        else:
            # 如果没有网络特征，使用零向量
            network_encoded = torch.zeros(text_encoded.size(0), 128, device=text_encoded.device)

        # 4. URL+网络融合模态
        modal_encoded = self.modal_encoder(torch.cat([url_encoded, network_encoded], dim=1))

        # 5. 跨模态注意力融合
        cross_attention_features = self.cross_attention(text_encoded, modal_encoded)

        # 6. 模态自适应权重融合
        weights = torch.softmax(self.modal_weights, dim=0)
        weighted_text = weights[0] * text_encoded
        weighted_modal = weights[1] * modal_encoded
        weighted_cross = weights[2] * cross_attention_features

        # 7. 多模态特征融合
        all_features = torch.cat([weighted_text, weighted_modal, weighted_cross], dim=1)
        fused_features = self.fusion(all_features)

        # 8. 分类
        return self.classification(fused_features)
