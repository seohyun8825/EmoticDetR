#Models

import torch
import torch.nn as nn
import timm
from torchvision.models import resnet50

#Without Contextual Understanding(for ablaton study)

class EmoticModel(nn.Module):
    def __init__(self, body_name, context_name, hidden_dim=512, pretrained=True):
        super(EmoticModel, self).__init__()
        # body_name: "resnet18"
        # context_name: "resnet18"

        self.model_body = timm.create_model(body_name, pretrained=pretrained, num_classes=0, global_pool='')
        self.model_context = timm.create_model(context_name, pretrained=True, num_classes=0, global_pool='')
        # Nx512

        num_context_features = 512
        num_body_features = 512
        self.fusion = nn.Sequential(
                        nn.Linear((num_context_features + num_body_features + 512 + 4 + 4), hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.Dropout(0.5),
                        nn.ReLU(),
                        )
        self.fc_cat = nn.Linear(hidden_dim, 26)
        self.fc_cont = nn.Linear(hidden_dim, 3)


    def forward(self, x_context, x_body, x_clip, bbox_body, bbox_face):
        # Backbone
        x_context = self.model_context(x_context)
        x_body = self.model_body(x_body)

        # Global Average Pooling
        x_context = torch.mean(x_context, [2, 3])
        x_body = torch.mean(x_body, [2, 3])

        # Concat with CLIP features
        # Case 1. X_clip: Nx512x7x7
        # Case 2. X_clip: Nx512

        if len(x_clip.shape) > 2:
            x_clip = torch.mean(x_clip, [2, 3]) # x_clip: Nx512

        out = torch.cat((x_context, x_body, x_clip, bbox_body, bbox_face), 1)
        out = self.fusion(out)

        out_cat = self.fc_cat(out)
        out_cont = self.fc_cont(out)

        return out_cat, out_cont

#With Contextual Understanding


class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # creat 기본 파이토치 transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # baseline DETR linear_bbox layer -> 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        h_pos = pos + 0.1 * h.flatten(2).permute(2, 0, 1)

        # propagate through the transformer
        h = self.transformer(h_pos,
                             self.query_pos.unsqueeze(1).repeat(1, inputs.shape[0], 1)).transpose(0, 1)
        return {'pred_logits': self.linear_class(h),
                'pred_boxes': self.linear_bbox(h).sigmoid(),
                'feats': h_pos}

import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, layer, rank):
        super().__init__()
        self.layer = layer  # 원래의 MultiheadAttention 레이어
        self.rank = rank

        # LoRA 매트릭스 초기화
        self.A_q = nn.Parameter(torch.randn(layer.embed_dim, self.rank))
        self.B_q = nn.Parameter(torch.randn(self.rank, layer.embed_dim))
        self.A_k = nn.Parameter(torch.randn(layer.embed_dim, self.rank))
        self.B_k = nn.Parameter(torch.randn(self.rank, layer.embed_dim))
        self.A_v = nn.Parameter(torch.randn(layer.embed_dim, self.rank))
        self.B_v = nn.Parameter(torch.randn(self.rank, layer.embed_dim))

    def forward(self, *args, **kwargs):
        # 원래 레이어의 forward 메서드 호출 전에 Q, K, V에 LoRA 적용
        q, k, v = args[:3]  # Q, K, V 행렬 추출

        # LoRA 변형 적용
        q = q + torch.matmul(q, self.A_q).matmul(self.B_q)
        k = k + torch.matmul(k, self.A_k).matmul(self.B_k)
        v = v + torch.matmul(v, self.A_v).matmul(self.B_v)

        # 수정된 Q, K, V를 사용하여 원래 레이어의 forward 메서드 호출
        return self.layer.forward(q, k, v, *args[3:], **kwargs)
    def __getattr__(self, name):
        # super()를 사용하여 상위 클래스의 __getattr__ 호출
        try:
            return super().__getattr__(name)
        except AttributeError:
            # 상위 클래스에서 속성을 찾지 못한 경우 원본 레이어에서 속성을 찾음
            return getattr(self.layer, name)


import torch.nn as nn
import torch.nn.functional as F

class EmoticDetModel(nn.Module):
    def __init__(self, body_name, hidden_dim=256, pretrained=True):
        super(EmoticDetModel, self).__init__()
        feature_size = 128
        self.model_body = timm.create_model(body_name, pretrained=pretrained, num_classes=0, global_pool='')
        self.linear_body = nn.Linear(512+4+4, 256)

        self.model_context = DETR(num_classes=91)
        state_dict = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
            map_location='cpu', check_hash=True)
        self.model_context.load_state_dict(state_dict)
        self.model_context.eval()
        self.hidden_dim = hidden_dim
        self.context_feature_transform = nn.Linear(1024, hidden_dim)
        self.scenario_transform = nn.Linear(768, hidden_dim)  # Adjust input dimension

        hidden_dim = hidden_dim
        nheads = 8
        num_encoder_layers = 3
        num_decoder_layers = 3

        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        self.fc_cat = nn.Linear(hidden_dim + 512, 26)
        self.fc_cont = nn.Linear(hidden_dim + 512, 3)
    def forward(self, x_context, x_body, x_clip, bbox_body, bbox_face, x_scenario):
        x_body = self.model_body(x_body)
        x_body = torch.mean(x_body, [2, 3])  # Global average pooling
        x_body = torch.cat((x_body, bbox_body, bbox_face), 1)
        x_body = self.linear_body(x_body)  # Shape: [batch_size, 256]

        # Process x_context
        x_context = self.model_context(x_context)['feats']
        x_context = x_context.transpose(0, 1)  # Shape: [49, batch_size, 1024]

        # Transform x_context to have the correct feature dimension
        if x_context.size(-1) != self.hidden_dim:
            x_context = self.context_feature_transform(x_context)  # Shape: [49, batch_size, hidden_dim]

        # Transform x_scenario
        x_scenario = self.scenario_transform(x_scenario)  # Shape: [batch_size, hidden_dim]

        # Transformer
        x_body_transformer = x_body.unsqueeze(1).repeat(1, 49, 1)  # Repeat x_body to match sequence length

        out = self.transformer(x_context, x_body_transformer)

        x_clip = x_clip.squeeze(1)
        x_clip = x_clip.unsqueeze(1).repeat(1, out.size(1), 1)  # Repeat x_clip along sequence length

        out = torch.cat((out, x_clip), 2)
        out_cat = self.fc_cat(out)
        out_cont = self.fc_cont(out)
        out_cat = out_cat.mean(dim=1)
        out_cont = out_cont.mean(dim=1)

        return out_cat, out_cont
    def apply_lora(self, rank=4):
        # 첫 번째 Transformer 인코더 레이어의 MultiheadAttention에 LoRA 적용
        original_attention_layer = model.transformer.encoder.layers[0].self_attn

        # LoRA 레이어 생성 및 적용
        lora_layer = LoRALayer(original_attention_layer, rank=4)
        model.transformer.encoder.layers[0].self_attn = lora_layer

