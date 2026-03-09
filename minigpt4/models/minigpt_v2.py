import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from minigpt4.common.registry import registry
from minigpt4.models.base_model import disabled_train
from minigpt4.models.minigpt_base import MiniGPTBase
import logging
from peft import LoraConfig, inject_adapter_in_model
import warnings
import math

class EnhancedResampler3D(nn.Module):
    def __init__(self, num_global_queries=6, num_detail_queries=2, 
                 num_preserved_details=48, embed_dim=2560, 
                 num_heads=8, kv_dim=None, norm_layer=nn.LayerNorm):
        """
        Enhanced 3D Resampler with two-stage detail compression
        
        Args:
            num_global_queries: number of global compression queries
            num_detail_queries: number of final detail queries (after compression)
            num_preserved_details: number of detail features to preserve before compression
            embed_dim: embedding dimension
            num_heads: attention heads
            kv_dim: key-value dimension (if different from embed_dim)
            norm_layer: normalization layer
        """
        super().__init__()
        self.num_global_queries = num_global_queries
        self.num_detail_queries = num_detail_queries
        self.num_preserved_details = num_preserved_details
        self.total_queries = num_global_queries + num_detail_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 全局压缩查询
        self.global_query = nn.Parameter(torch.zeros(num_global_queries, embed_dim))
        self.global_query.data.normal_(mean=0.0, std=0.02)
        
        # 细节压缩查询（用于将preserved details压缩为最终的detail queries）
        self.detail_query = nn.Parameter(torch.zeros(num_detail_queries, embed_dim))
        self.detail_query.data.normal_(mean=0.0, std=0.02)
        
        # 灵活的位置编码
        self.global_pos_embed = nn.Parameter(
            self._generate_flexible_pos_embed(num_global_queries, embed_dim)
        )
        self.detail_pos_embed = nn.Parameter(
            self._generate_flexible_pos_embed(num_detail_queries, embed_dim)
        )
        
        # KV投影层
        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()
        
        # 注意力层
        self.global_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.detail_selection_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.detail_compression_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # 归一化层
        self.ln_global_q = norm_layer(embed_dim)
        self.ln_global_kv = norm_layer(embed_dim)
        self.ln_detail_q = norm_layer(embed_dim)
        self.ln_detail_kv = norm_layer(embed_dim)
        self.ln_detail_compress = norm_layer(embed_dim)
        
        # 注意力互补机制
        self.attention_gate = AttentionGate(embed_dim)
        
        # 细节选择机制
        self.detail_selector = DetailSelector(embed_dim, num_preserved_details)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            norm_layer(embed_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def _generate_flexible_pos_embed(self, num_queries, embed_dim):
        """生成灵活的位置编码"""
        pos = torch.arange(num_queries, dtype=torch.float32)
        pe = torch.zeros(num_queries, embed_dim)
        
        position = pos.unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                            (-np.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, x):
        """
        Args:
            x: [bsz, 513, D] input features
        Returns:
            compressed: [bsz, total_queries, D] compressed features
            attention_info: dict with attention analysis
        """
        bsz = x.size(0)
        
        # 基础特征处理
        voxel_feats = self.kv_proj(x)  # [bsz, 513, D]
        
        # === 全局特征压缩 ===
        global_kv = self.ln_global_kv(voxel_feats)
        
        # 准备全局查询
        global_Q = self.ln_global_q(self.global_query)  # [num_global, D]
        global_Q = global_Q + self.global_pos_embed
        global_Q = global_Q.unsqueeze(0).expand(bsz, -1, -1)  # [bsz, num_global, D]
        
        # 全局注意力
        global_compressed, global_attn_weights = self.global_attn(
            query=global_Q,
            key=global_kv,
            value=global_kv
        )  # [bsz, num_global, D], [bsz, num_global, 513]
        
        # === 细节特征选择和压缩 ===
        detail_kv = self.ln_detail_kv(voxel_feats)
        
        # Step 1: 选择重要的细节特征
        # 使用attention gate找到全局注意力较低但重要的区域
        attention_mask = self.attention_gate(global_attn_weights, voxel_feats)
        
        # 使用detail_selector选择top-k个细节特征
        selected_details, detail_indices, detail_scores = self.detail_selector(
            detail_kv, attention_mask
        )  # [bsz, num_preserved_details, D], [bsz, num_preserved_details], [bsz, num_preserved_details]
        
        # Step 2: 使用细节查询压缩选中的细节特征
        detail_Q = self.ln_detail_q(self.detail_query)  # [num_detail_queries, D]
        detail_Q = detail_Q + self.detail_pos_embed
        detail_Q = detail_Q.unsqueeze(0).expand(bsz, -1, -1)  # [bsz, num_detail_queries, D]
        
        # 对选中的细节特征进行归一化
        selected_details = self.ln_detail_compress(selected_details)
        
        # 细节压缩注意力
        detail_compressed, detail_attn_weights = self.detail_compression_attn(
            query=detail_Q,
            key=selected_details,
            value=selected_details
        )  # [bsz, num_detail_queries, D]
        
        # === 直接拼接全局和细节特征 ===
        compressed_features = torch.cat([global_compressed, detail_compressed], dim=1)
        # [bsz, num_global + num_detail, D]
        
        # 输出投影
        output_features = self.output_proj(compressed_features)
        
        # 返回注意力分析信息
        attention_info = {
            'global_attention': global_attn_weights,
            'detail_selection_scores': detail_scores,
            'detail_indices': detail_indices,
            'detail_compression_attention': detail_attn_weights,
            'attention_mask': attention_mask,
            'num_global_queries': self.num_global_queries,
            'num_detail_queries': self.num_detail_queries,
            'num_preserved_details': self.num_preserved_details
        }
        
        return output_features, attention_info


class AttentionGate(nn.Module):
    """生成互补的注意力掩码"""
    
    def __init__(self, embed_dim, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.importance_scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1)
        )
    
    def forward(self, global_attention, features):
        """
        Args:
            global_attention: [bsz, num_global, seq_len] 全局注意力权重
            features: [bsz, seq_len, D] 输入特征
        Returns:
            attention_scores: [bsz, seq_len] 互补注意力分数
        """
        bsz, num_global, seq_len = global_attention.shape
        
        # 计算每个位置被全局查询关注的总权重
        global_coverage = global_attention.sum(dim=1)  # [bsz, seq_len]
        
        # 计算特征重要性分数
        importance_scores = self.importance_scorer(features).squeeze(-1)  # [bsz, seq_len]
        importance_scores = torch.sigmoid(importance_scores)
        
        # 生成互补分数：低全局覆盖但高重要性的区域
        complementary_scores = importance_scores * (1 - torch.sigmoid(global_coverage * 10))
        
        return complementary_scores


class DetailSelector(nn.Module):
    """选择最重要的细节特征"""
    
    def __init__(self, embed_dim, num_preserved_details):
        super().__init__()
        self.num_preserved_details = num_preserved_details
        self.score_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1)
        )
    
    def forward(self, features, attention_scores):
        """
        Args:
            features: [bsz, seq_len, D] 输入特征
            attention_scores: [bsz, seq_len] 互补注意力分数
        Returns:
            selected_features: [bsz, num_preserved_details, D] 选中的特征
            indices: [bsz, num_preserved_details] 选中特征的索引
            scores: [bsz, num_preserved_details] 选中特征的分数
        """
        bsz, seq_len, embed_dim = features.shape
        
        # 结合attention scores和特征自身的重要性
        feature_importance = self.score_projector(features).squeeze(-1)  # [bsz, seq_len]
        feature_importance = torch.sigmoid(feature_importance)
        
        # 综合分数
        combined_scores = attention_scores * feature_importance  # [bsz, seq_len]
        
        # 选择top-k个特征
        topk_scores, topk_indices = torch.topk(
            combined_scores, 
            k=min(self.num_preserved_details, seq_len), 
            dim=-1,
            largest=True
        )  # [bsz, num_preserved_details]
        
        # 提取对应的特征
        batch_indices = torch.arange(bsz, device=features.device).unsqueeze(1)
        selected_features = features[batch_indices, topk_indices]  # [bsz, num_preserved_details, D]
        
        return selected_features, topk_indices, topk_scores
    
    
@registry.register_model("minigpt_3d")
class MiniGPT_3D(MiniGPTBase):
    """
    MiniGPT_3D model
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/minigpt_3d.yaml",
    }

    def __init__(
            self,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            pc_precision="fp16",
            freeze_pc=True,
            llama_model="",
            prompt_template='###Human: {} ###Assistant: ',
            max_txt_len=300,
            end_sym='\n',
            lora_r=64,
            lora_target_modules=['query_key_value', 'dense'],
            lora_alpha=16,
            lora_dropout=0.05,
            chat_template=False,
            use_grad_checkpoint_llm=False,
            max_context_len=3800,
            low_resource=False,  # use 8 bit and put pc in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
            pc_linear_layer=2,
            only_train_pc_linear=False,
            only_train_Q=False,
            embed_dim=2560
    ):
        super().__init__(
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            max_context_len=max_context_len,
            end_sym=end_sym,
            prompt_template=prompt_template,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        
        print('Init pc encoder: pc_encoder')
        self.pc_encoder = self.init_pc_encoder(pc_precision, freeze_pc)

        print('Init PointCloud Resampler')
        self.resampler = EnhancedResampler3D(
                num_global_queries=8,        # 6个全局压缩查询
                num_detail_queries=4,        # 2个最终细节查询
                num_preserved_details=96,    # 先保留48个细节特征
                embed_dim=2560,
                num_heads=8
            )

        print('Create the MLP: mlp_projector')
        if pc_linear_layer == 1:
            projection_layers = []
            projection_layers.append(nn.Linear(self.pc_encoder.trans_dim, self.llama_model.config.hidden_size))
            self.mlp_projector = nn.Sequential(*projection_layers)
        elif pc_linear_layer == 2:
            self.mlp_projector = nn.Sequential(
                nn.Linear(self.pc_encoder.trans_dim, 4096),
                nn.GELU(),
                nn.Linear(4096, self.llama_model.config.hidden_size),
            )
        elif pc_linear_layer == 3:
            self.mlp_projector = nn.Sequential(
                nn.Linear(self.pc_encoder.trans_dim, 768),
                nn.GELU(),
                nn.Linear(768, 4096),
                nn.GELU(),
                nn.Linear(4096, self.llama_model.config.hidden_size),
            )
      

        print("Freeze pc encoder")
        # fix the pc encoder in all training stages
        self.freeze_pc = freeze_pc
        if self.freeze_pc:
            for name, param in self.pc_encoder.named_parameters():
                param.requires_grad = False
            self.point_encoder = self.pc_encoder.eval()
            self.point_encoder.train = disabled_train

        
        
        ############### Stage I - Global Compression Training ###
        self.only_train_pc_linear = only_train_pc_linear
        if self.only_train_pc_linear:

            # PC Encoder in Stage I of paper figure 3
            for name, param in self.pc_encoder.named_parameters():
                param.requires_grad = False

            # MLP in Stage I of paper figure 3
            for name, param in self.mlp_projector.named_parameters():
                param.requires_grad = True

            # LLM (Phi-2) with LoRA in Stage I of paper figure 3
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False

            # Resampler parameters configuration for Stage I
            for param in self.resampler.parameters():
                param.requires_grad = False  # 默认关闭所有参数

            # Stage I: 仅训练全局压缩相关参数

            # 1. 全局查询和位置编码
            self.resampler.global_query.requires_grad = True
            self.resampler.global_pos_embed.requires_grad = True

            # 2. 全局注意力层
            for param in self.resampler.global_attn.parameters():
                param.requires_grad = True

            # 3. 全局归一化层
            for param in self.resampler.ln_global_q.parameters():
                param.requires_grad = True
            for param in self.resampler.ln_global_kv.parameters():
                param.requires_grad = True

            # 4. KV投影层（如果存在且不是Identity）
            if not isinstance(self.resampler.kv_proj, nn.Identity):
                for param in self.resampler.kv_proj.parameters():
                    param.requires_grad = True

            # 5. 输出投影层（需要处理全局特征）
            for param in self.resampler.output_proj.parameters():
                param.requires_grad = True

            print("Stage I: only train global compression and MLP projector")

        ############### Stage I - Global Compression Training ###


        ############### Stage IV - Detail Compression Training ###
        self.only_train_Q = only_train_Q
        if self.only_train_Q:

            # PC Encoder in Stage IV of paper figure 3
            for name, param in self.pc_encoder.named_parameters():
                param.requires_grad = False

            # MLP in Stage IV of paper figure 3
            for name, param in self.mlp_projector.named_parameters():
                param.requires_grad = False

            # LLM (Phi-2) with LoRA in Stage IV of paper figure 3
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False

            # Resampler parameters configuration for Stage IV
            for param in self.resampler.parameters():
                param.requires_grad = False  # 默认关闭所有参数

            # Stage IV: 仅训练细节压缩相关参数

            # 1. 细节查询和位置编码
            self.resampler.detail_query.requires_grad = True
            self.resampler.detail_pos_embed.requires_grad = True

            # 2. 细节选择和压缩注意力层
            for param in self.resampler.detail_selection_attn.parameters():
                param.requires_grad = True
            for param in self.resampler.detail_compression_attn.parameters():
                param.requires_grad = True

            # 3. 细节归一化层
            for param in self.resampler.ln_detail_q.parameters():
                param.requires_grad = True
            for param in self.resampler.ln_detail_kv.parameters():
                param.requires_grad = True
            for param in self.resampler.ln_detail_compress.parameters():
                param.requires_grad = True

            # 4. 注意力门控机制（生成互补掩码）
            for param in self.resampler.attention_gate.parameters():
                param.requires_grad = True

            # 5. 细节选择器（选择重要特征）
            for param in self.resampler.detail_selector.parameters():
                param.requires_grad = True

            # 6. 输出投影层（需要处理融合后的特征）
            for param in self.resampler.output_proj.parameters():
                param.requires_grad = True

            print("Stage IV: only train detail selection and compression")

       
    
    def encode_pc(self, pc):
        device = pc.device

        with self.maybe_autocast():

                pc_features = self.pc_encoder(pc).to(device)# [bsz, 513, 384]

                pc_embeds = self.mlp_projector(pc_features)  # [bsz,513,llama_hidden_size(2560)]
                # print(f"pc_embeds: {pc_embeds.shape}")   
                
                compressed_inputs_llama, _ = self.resampler(pc_embeds)# [bsz,size^3,llama_hidden_size(2560)]
                # print(f"compressed_inputs_llama: {compressed_inputs_llama.shape}")

                atts_llama = torch.ones(compressed_inputs_llama.shape[:-1], dtype=torch.long, device=pc.device).to(device)

        return compressed_inputs_llama, atts_llama
            


    @classmethod
    def from_config(cls, cfg):
        pc_model = cfg.get("pc_model", "pointbert")

        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        pc_precision = cfg.get("pc_precision", "fp16")
        freeze_pc = cfg.get("freeze_pc", True)
        low_resource = cfg.get("low_resource", False)

        prompt_template = cfg.get("prompt_template", '[INST] {} [/INST]')
        max_txt_len = cfg.get("max_txt_len", 300)
        end_sym = cfg.get("end_sym", '\n')

        lora_r = cfg.get("lora_r", 64)
        lora_alpha = cfg.get("lora_alpha", 16)
        chat_template = cfg.get("chat_template", False)

        use_grad_checkpoint_llm = cfg.get("use_grad_checkpoint_llm", False)
        max_context_len = cfg.get("max_context_len", 3800)


        pc_linear_layer = cfg.get("pc_linear_layer", 2)

        only_train_pc_linear = cfg.get("only_train_pc_linear", False)

        only_train_Q = cfg.get("only_train_Q", False)

        model = cls(
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            pc_precision=pc_precision,
            freeze_pc=freeze_pc,
            llama_model=llama_model,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            low_resource=low_resource,
            end_sym=end_sym,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            chat_template=chat_template,
            use_grad_checkpoint_llm=use_grad_checkpoint_llm,
            max_context_len=max_context_len,
            pc_linear_layer=pc_linear_layer,
            only_train_pc_linear=only_train_pc_linear,
            only_train_Q=only_train_Q,
        )

        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            print("Load MiniGPT-3D first Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        else:
            print("No load first ckpt!!!")

        stage_3_ckpt = cfg.get("second_ckpt", "")
        if stage_3_ckpt:
            print("Load MiniGPT-3D second_ckpt Checkpoint: {}".format(stage_3_ckpt))
            ckpt = torch.load(stage_3_ckpt, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        else:
            print("No load second_ckpt!!!")

        return model