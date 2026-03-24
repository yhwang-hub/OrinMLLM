# EAGLE-3 投机解码方案深度分析报告

## 目录

1. [概述](#概述)
2. [从EAGLE-1到EAGLE-3的演进](#从eagle-1到eagle-3的演进)
3. [EAGLE-3核心创新](#eagle-3核心创新)
4. [EAGLE-3工程架构分析](#eagle-3工程架构分析)
5. [Draft模型（草稿模型）结构详解](#draft模型草稿模型结构详解)
6. [推理阶段完整流程](#推理阶段完整流程)
7. [训练阶段完整流程](#训练阶段完整流程)
8. [树状投机解码机制](#树状投机解码机制)
9. [Draft词汇表压缩机制](#draft词汇表压缩机制)
10. [草稿Token验证的完整流程详解](#草稿token验证的完整流程详解)
11. [性能分析](#性能分析)
12. [与OrinMLLM集成分析](#与orinmllm集成分析)

---

## 概述

EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）是一种面向大语言模型（LLM）的投机解码（Speculative Decoding）加速方案。其核心思想是：使用一个轻量级的**草稿模型（Draft Model）**并行生成多个候选token，再由**目标模型（Target Model）**一次性验证，从而将原本逐token的自回归解码加速为批量验证解码。

EAGLE-3是该系列的最新版本，于2025年3月发布，已被NeurIPS'25接收。相比前代版本：

| 版本 | 相对原始解码的加速 | 核心技术 |
|------|-------------------|----------|
| EAGLE-1 | ~3x (13B) | 次顶层特征外推 |
| EAGLE-2 | ~4x (13B) | 动态调整草稿树结构 |
| **EAGLE-3** | **~5.6x (13B)** | 多层特征融合 + Training-Time Testing |

## 从EAGLE-1到EAGLE-3的演进

### EAGLE-1: 特征外推

EAGLE-1的核心观察是：LLM第二顶层（倒数第二层）的特征向量在时间步之间的变化比token嵌入更加平稳和可预测。因此，EAGLE-1训练一个轻量级草稿模型来**预测下一时间步的次顶层特征**，然后通过目标模型的LM Head将其映射为token预测。

**输入**: 上一步的次顶层隐藏状态 + 上一步的token嵌入
**输出**: 下一步的次顶层隐藏状态预测

### EAGLE-2: 动态树结构

EAGLE-2引入了**基于置信度的动态树结构调整**。草稿模型的预测置信度被用来近似接受率，从而动态决定树形草稿的结构——高置信度的分支被扩展，低置信度的被剪枝。

### EAGLE-3: 多层语义融合 + Training-Time Testing

EAGLE-3做出了两个关键改变：

1. **移除特征预测约束**: EAGLE-1/2的草稿模型被限制去预测目标模型的次顶层特征，这是一个间接优化目标。EAGLE-3移除了这个约束，直接优化token预测的准确性。

2. **多层特征融合替代单层特征**: EAGLE-1/2仅使用目标模型的次顶层特征。EAGLE-3认为顶层特征主要服务于下一个token的预测，信息有限。因此，EAGLE-3将**低层、中层和高层**三个层次的特征拼接，提供更丰富的语义信息：
   - 低层特征（如第2层）: 包含基础语法和词法信息
   - 中层特征（如中间层）: 包含句法和浅层语义
   - 高层特征（如倒数第3层）: 包含深层语义但保留了更多通用性

3. **Training-Time Testing**: 训练过程模拟推理时的自回归生成，让草稿模型在训练阶段就体验到推理时的特征"平移（shift）"模式。

---

## EAGLE-3核心创新

### 创新1: 三层特征融合

在目标模型的前向传播中，EAGLE-3修改了基座模型（如Qwen3、LLaMA等），从三个特定层提取隐藏状态并拼接：

```python
# 来自 modeling_qwen3_kv.py 的关键修改
for idx, decoder_layer in enumerate(self.layers):
    # 提取三个层的隐藏状态: 低层(idx=2)、中层(中间层)、高层(倒数第3层)
    if idx == len(self.layers) - 3 or idx == len(self.layers) // 2 or idx == 2:
        all_hidden_states += (hidden_states,)
```

对于一个32层的模型（如LLaMA-3 8B / Qwen3-8B），提取的三个层分别为：
- **第2层** (`idx == 2`): 低层语义特征
- **第16层** (`idx == len(self.layers) // 2`): 中层语义特征
- **第29层** (`idx == len(self.layers) - 3`): 高层语义特征

这三个隐藏状态在特征维度上拼接：

```python
# 来自 utils.py 的 initialize_tree 和 tree_decoding 函数
hidden_states = torch.cat(outputs["hidden_states"], dim=-1)
# 结果维度: [batch, seq_len, hidden_size * 3]
```

### 创新2: 多层特征投影网络 (FC层)

融合后的特征通过一个全连接投影层压缩回原始维度：

```python
# 来自 cnets.py 的 Model 类
self.fc = nn.Linear(config.hidden_size * 3, self.hidden_size, bias=False)
# 对于 hidden_size=4096 的模型: Linear(12288, 4096)
```

如果目标模型和草稿模型的hidden size不同（跨模型的情况），还支持配置 `target_hidden_size`：

```python
if hasattr(config, "target_hidden_size"):
    self.fc = nn.Linear(config.target_hidden_size * 3, self.hidden_size, bias=False)
else:
    self.fc = nn.Linear(config.hidden_size * 3, self.hidden_size, bias=False)
```

### 创新3: Training-Time Testing

传统的草稿模型训练将其视为独立的序列预测任务。EAGLE-3的训练过程模拟了推理时的**自回归草稿生成**：

```python
# 来自 traineagle3/cnets.py 的训练 forward
for idx in range(self.length):  # self.length = 7，模拟7步自回归
    # 获取当前token的嵌入
    inputs_embeds = self.embed_tokens(input_ids)
    
    # 草稿模型前向（使用缓存的隐藏状态历史）
    layer_outputs, cache_hidden = self.midlayer(
        input_emb=inputs_embeds,
        hidden_states=hidden_states,
        cache_hidden=cache_hidden,  # 累积的历史KV缓存
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    
    # 计算损失（KL散度）
    logits = self.lm_head(self.norm(hidden_states_out))
    out_logp = nn.LogSoftmax(dim=2)(logits)
    plogp = target_p * out_logp
    loss = -torch.sum(position_mask * plogp, 2).mean()
    
    # 特征左移（shift）: 模拟推理时的自回归行为
    input_ids = padding(input_ids, left=False)   # token序列左移
    target = padding(target, left=False)          # 目标分布左移
    loss_mask = padding(loss_mask, left=False)    # 损失掩码左移
```

关键点在于`padding(tensor, left=False)`函数——它将张量向左移位一个位置，模拟了推理时每一步草稿模型只能访问"前一步"特征的情况。

---

## EAGLE-3工程架构分析

### 项目结构

```
EAGLE/
├── eagle/
│   ├── model/              # 推理模型
│   │   ├── ea_model.py     # 主入口：EaModel类，封装base_model + draft_model
│   │   ├── cnets.py        # EAGLE-3草稿模型 (Model类)
│   │   ├── cnets1.py       # EAGLE-1/2草稿模型 (Model类)
│   │   ├── configs.py      # 草稿模型配置 (EConfig)
│   │   ├── utils.py        # 推理utils: 树构建、验证、更新
│   │   ├── utils_c.py      # 优化的utils版本
│   │   ├── kv_cache.py     # KV缓存管理
│   │   ├── choices.py      # 树结构定义
│   │   ├── modeling_llama_kv.py    # 修改版LLaMA (输出多层隐藏状态)
│   │   ├── modeling_qwen2_kv.py    # 修改版Qwen2
│   │   ├── modeling_qwen3_kv.py    # 修改版Qwen3
│   │   └── modeling_mixtral_kv.py  # 修改版Mixtral
│   ├── traineagle3/        # EAGLE-3训练代码
│   │   ├── main.py         # 训练入口
│   │   ├── cnets.py        # 训练版草稿模型（含Training-Time Testing逻辑）
│   │   ├── configs.py      # 训练配置
│   │   └── modeling_llama_kv.py    # 训练用基座模型
│   └── train/              # EAGLE-1/2训练代码
├── inference.py             # 推理脚本示例
└── setup.py
```

### 核心类关系

```
EaModel (ea_model.py)
├── base_model: Qwen3ForCausalLM / LlamaForCausalLM  # 目标模型(冻结)
│   └── model.layers[]                                # Transformer层
│       └── 输出 all_hidden_states (3层拼接)
├── ea_layer: Model (cnets.py)                         # 草稿模型(可训练)
│   ├── fc: Linear(hidden*3, hidden)                   # 三层特征投影
│   ├── embed_tokens: Embedding(vocab, hidden)         # Token嵌入(从base复制,冻结)
│   ├── midlayer: LlamaDecoderLayeremb                 # 单层Decoder
│   │   ├── self_attn: LlamaAttention                  # 注意力(input_dim=hidden*2)
│   │   ├── mlp: LlamaMLP                              # FFN
│   │   ├── hidden_norm: LlamaRMSNorm                  # 隐藏状态输入规范化
│   │   ├── input_layernorm: LlamaRMSNorm              # 嵌入输入规范化
│   │   └── post_attention_layernorm: LlamaRMSNorm     # FFN前规范化
│   ├── norm: LlamaRMSNorm                             # 输出规范化
│   ├── lm_head: Linear(hidden, draft_vocab)           # 草稿词汇表预测头
│   ├── d2t: Tensor[draft_vocab]                       # Draft→Target词汇映射
│   └── t2d: Tensor[vocab]                             # Target→Draft词汇映射(bool)
└── tokenizer: AutoTokenizer
```

---

## Draft模型（草稿模型）结构详解

EAGLE-3的草稿模型由以下核心组件构成：

### 1. 特征投影层 (FC)

```python
self.fc = nn.Linear(config.hidden_size * 3, self.hidden_size, bias=False)
```

将三层隐藏状态的拼接 `[batch, seq, hidden*3]` 投影为 `[batch, seq, hidden]`。

### 2. 单层Decoder (`LlamaDecoderLayeremb`)

这是整个草稿模型的核心计算单元，**仅有一层Decoder Layer**。其特殊之处在于：

#### 双流输入设计

```python
def forward(self, input_emb, hidden_states, ...):
    residual = hidden_states
    
    # 分别规范化两个输入流
    hidden_states = self.hidden_norm(hidden_states)    # 隐藏状态流
    input_emb = self.input_layernorm(input_emb)        # 嵌入流
    
    # 拼接为双倍宽度: [batch, seq, hidden*2]
    hidden_states = torch.cat((input_emb, hidden_states), dim=-1)
    
    # 自注意力 (QKV投影接受 hidden*2 维度输入)
    hidden_states = self.self_attn(hidden_states=hidden_states, ...)
    hidden_states = residual + hidden_states
    
    # FFN (标准MLP，输入/输出均为 hidden 维度)
    residual = hidden_states
    hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
    hidden_states = residual + hidden_states
```

**两个输入**:
- `input_emb`: 上一步预测token的嵌入向量
- `hidden_states`: FC层投影后的特征（首次调用）或上一步Decoder输出（后续调用）

注意力层的QKV投影接受拼接后的 `hidden*2` 维度输入：
```python
self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
```

### 3. 输出头

```python
self.norm = LlamaRMSNorm(config.hidden_size)
self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)
```

通过RMSNorm规范化后，由一个线性层映射到**草稿词汇表**（通常小于目标模型词汇表，如32000 vs 128256）。

### 参数量分析

以hidden_size=4096, intermediate_size=14336, draft_vocab_size=32000为例:

| 组件 | 参数量 |
|------|--------|
| FC投影 | 4096 × 12288 = 50.3M |
| Q_proj | 4096 × 8192 = 33.6M |
| K_proj | 1024 × 8192 = 8.4M |
| V_proj | 1024 × 8192 = 8.4M |
| O_proj | 4096 × 4096 = 16.8M |
| Gate_proj | 14336 × 4096 = 58.7M |
| Up_proj | 14336 × 4096 = 58.7M |
| Down_proj | 4096 × 14336 = 58.7M |
| LM_head | 32000 × 4096 = 131.1M |
| Norm/LayerNorm | ~0.04M |
| **合计** | **~424M** |

草稿模型约424M参数，远小于8B目标模型，单层Decoder的前向计算成本约为目标模型的1/32。

---

## 推理阶段完整流程

### 整体流程图

```
输入 tokens
    │
    ▼
┌─────────────────────────────┐
│  Phase 1: Prefill (预填充)    │
│  目标模型前向 → 获取          │
│  ① 三层隐藏状态拼接           │
│  ② 最后一个token的logits      │
│  ③ 采样第一个新token          │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  Phase 2: Draft Generation (草稿生成) — 循环    │
│                                                   │
│  ┌───────────────────────────────────┐            │
│  │ Step 2a: 草稿模型 topK_genrate   │            │
│  │ 输入: 隐藏状态 + input_ids        │            │
│  │ 自回归生成 depth 步               │            │
│  │ 每步选 top-K 个候选              │            │
│  │ 输出: draft_tokens (树形)        │            │
│  │        tree_mask, tree_position  │             │
│  │        retrieve_indices          │             │
│  └──────────────┬────────────────────┘            │
│                 │                                  │
│                 ▼                                  │
│  ┌───────────────────────────────────┐            │
│  │ Step 2b: Target Verification     │            │
│  │ 目标模型并行验证所有候选token      │            │
│  │ 基于 tree_mask 的注意力掩码       │            │
│  │ 一次前向即可验证整棵树            │            │
│  └──────────────┬────────────────────┘            │
│                 │                                  │
│                 ▼                                  │
│  ┌───────────────────────────────────┐            │
│  │ Step 2c: Evaluate & Accept       │            │
│  │ 贪心: argmax匹配                 │            │
│  │ 采样: 按概率接受/拒绝             │            │
│  │ 输出: 接受的token序列            │            │
│  └──────────────┬────────────────────┘            │
│                 │                                  │
│                 ▼                                  │
│  ┌───────────────────────────────────┐            │
│  │ Step 2d: Update State            │            │
│  │ 更新 KV Cache                    │            │
│  │ 更新 input_ids                   │            │
│  │ 准备下轮草稿生成                  │            │
│  └──────────────┬────────────────────┘            │
│                 │                                  │
│                 ▼                                  │
│        循环直到 EOS / 达到 max_tokens              │
└─────────────────────────────────────────────────┘
```

### Phase 1: 预填充 (`initialize_tree`)

```python
def initialize_tree(input_ids, model, past_key_values, logits_processor):
    # 1. 目标模型前向传播，获取多层隐藏状态
    outputs, orig, hidden_states = model(
        input_ids, past_key_values=past_key_values, output_orig=True
    )
    
    # 2. 从最后一个token的logits中采样/argmax得到第一个新token
    if logits_processor is not None:
        logits = orig[:, -1]
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(orig[:, -1])
        token = token[None, None]
    
    input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    
    # 3. EAGLE-3特有: 拼接三层隐藏状态
    hidden_states = torch.cat(outputs["hidden_states"], dim=-1)
    # shape: [1, seq_len, hidden_size * 3]
    
    # 4. 草稿模型生成候选树
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = \
        model.ea_layer.topK_genrate(hidden_states, input_ids, model.base_model.lm_head, logits_processor)
    
    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, orig, hidden_states, token
```

### Phase 2a: 草稿生成 (`topK_genrate`)

这是EAGLE-3最核心的推理函数，实现**树形候选token生成**：

```python
@torch.no_grad()
def topK_genrate(self, hidden_states, input_ids, head, logits_processor):
    total_tokens = self.total_tokens  # 总候选token数 (如59)
    depth = self.depth               # 树深度 (如7)
    top_k = self.top_k               # 每步保留的top-K (如10)
    
    sample_token = input_ids[:, -1]  # 已确认的最后一个token
    input_ids = input_ids[:, 1:]     # 去掉BOS
    
    # ===== 第一步: 处理完整隐藏状态序列 =====
    if self.stable_kv is not None:
        # 增量更新: 只处理KV缓存未覆盖的部分
        kv_len = self.stable_kv[0][0].shape[2]
        out_hidden, past_key_values = self(hidden_states, input_ids=input_ids[:, kv_len:],
                                           past_key_values=self.stable_kv, use_cache=True)
    else:
        # 首次调用: 处理完整序列
        out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True)
    self.stable_kv = past_key_values
    
    # 取最后一个位置的隐藏状态
    last_hidden = out_hidden[:, -1]
    last_headout = self.lm_head(self.norm(last_hidden))
    
    # ===== Top-K选择第一级候选 =====
    last_p = self.logsoftmax(last_headout)
    top = torch.topk(last_p, top_k, dim=-1)
    topk_index, topk_p = top.indices, top.values
    
    # Draft→Target词汇映射
    if self.config.vocab_size == self.config.draft_vocab_size:
        ss_token.append(topk_index)
        input_ids = topk_index
    else:
        ss_token.append(topk_index + self.d2t[topk_index])  # 映射回目标词汇表
        input_ids = topk_index + self.d2t[topk_index]
    
    input_hidden = last_hidden[None].repeat(1, top_k, 1)

    # ===== 自回归展开 depth 步 =====
    for i in range(depth):
        # 草稿模型单步前向
        out_hidden, past_key_values = self(input_hidden, input_ids=input_ids,
                                           past_key_values=past_key_values,
                                           position_ids=position_ids, use_cache=True)
        
        # 计算每个候选的logits
        last_headout = self.lm_head(self.norm(out_hidden[0]))
        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        
        # 累积得分 = 当前得分 + 父节点得分
        cu_scores = topk_p + scores[:, None]
        
        # 在所有候选中选择全局 top-K
        topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
        topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
        
        # 更新树掩码，追踪父子关系
        tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)
    
    # ===== 构建最终的草稿树 =====
    # 从所有累积得分中选择 total_tokens 个最优候选
    scores_list = torch.cat(scores_list, dim=0).view(-1)
    top_scores = torch.topk(scores_list, total_tokens, dim=-1)
    
    # 构建树形注意力掩码 (确保每个节点只能看到它的祖先)
    tree_mask = torch.eye(total_tokens + 1).bool()
    tree_mask[:, 0] = True
    for i in range(total_tokens):
        tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])
    
    # 构建检索索引 (用于从树中提取可能的完整序列)
    # retrieve_indices: [leaf_num, max_depth] — 每个叶子到根的路径
    
    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids
```

### Phase 2b: 目标模型验证 (`tree_decoding`)

```python
def tree_decoding(model, tree_candidates, past_key_values,
                  tree_position_ids, input_ids, retrieve_indices):
    # Position IDs = 树位置 + 已有序列长度
    position_ids = tree_position_ids + input_ids.shape[1]
    
    # 目标模型一次前向验证整棵树
    # 利用 tree_mask 确保正确的注意力模式
    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    
    # EAGLE-3: 同时提取三层隐藏状态用于下轮草稿
    hidden_state = torch.cat(outputs["hidden_states"], dim=-1)
    
    # 按 retrieve_indices 提取每条候选路径的logits
    logits = tree_logits[0, retrieve_indices]
    
    return logits, hidden_state, outputs
```

### Phase 2c: 验证与接受 (`evaluate_posterior`)

#### 贪心模式 (temperature = 0)

```python
if logits_processor is None:
    # 逐位置检查: 候选token是否等于argmax(logits)
    posterior_mask = (
        candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
    ).int()
    # 连续匹配长度 (一旦不匹配就停止)
    candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
    accept_length = candidates_accept_length.max()
    best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
```

#### 采样模式 (temperature > 0)

在采样模式下，使用**拒绝采样**策略确保输出分布与目标模型一致：

```python
else:
    for i in range(1, candidates.shape[1]):
        gt_logits = logits[fi, i - 1]
        gtp = torch.softmax(gt_logits, dim=0)     # 目标分布
        r = random.random()
        px = gtp[xi]                                # 目标模型对该token的概率
        qx = 1.0                                    # 草稿模型概率(简化为1)
        acp = px / qx                              # 接受率
        if r <= acp:
            # 接受该token
            accept_length += 1
        else:
            # 拒绝，调整分布后从修正分布中采样
            gtp[xi] = 0
            gtp = gtp / gtp.sum()
```

### Phase 2d: 状态更新 (`update_inference_inputs`)

```python
def update_inference_inputs(input_ids, candidates, best_candidate, accept_length, ...):
    # 1. 选择被接受的KV缓存位置
    select_indices = retrieve_indices[best_candidate, :accept_length + 1] + prev_input_len
    
    # 2. 追加接受的tokens到序列
    input_ids = torch.cat([input_ids, candidates[None, best_candidate, :accept_length + 1]], dim=-1)
    
    # 3. 更新KV缓存: 将接受路径上的KV复制到连续位置
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices, :]
        dst = past_key_values_data[..., prev_input_len:prev_input_len + tgt.shape[-2], :]
        dst.copy_(tgt, non_blocking=True)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])
    
    # 4. 采样下一个token (用于下一轮草稿生成)
    prob = sample_p
    token = torch.argmax(prob) if logits_processor is None else torch.multinomial(prob, 1)
    
    # 5. 提取被接受位置的隐藏状态，用于下一轮草稿生成
    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, :accept_length + 1]
    
    # 6. 草稿模型生成下一棵候选树
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = \
        model.ea_layer.topK_genrate(
            accept_hidden_state_new,
            input_ids=torch.cat((input_ids, token), dim=1),
            head=model.base_model.lm_head,
            logits_processor=logits_processor
        )
    
    return input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, ...
```

---

## 训练阶段完整流程

### 数据准备 (`dataprepare`)

```python
@torch.no_grad()
def dataprepare(self, input_ids, attention_mask, loss_mask):
    # 1. 目标模型前向（冻结权重）
    outs = self.target_model(input_ids=input_ids, attention_mask=attention_mask)
    
    # 2. 提取三层隐藏状态
    hidden_states0 = outs.hidden_states[0]   # 低层
    hidden_states1 = outs.hidden_states[1]   # 中层
    hidden_states2 = outs.hidden_states[2]   # 高层
    hidden_states = torch.cat((hidden_states0, hidden_states1, hidden_states2), dim=-1)
    
    # 3. 获取目标logits并左移（对齐预测目标）
    target = outs.logits
    target = padding(target, left=False)      # 左移一位（预测下一个token）
    input_ids = padding(input_ids, left=False) # 左移一位
    
    return hidden_states, target, loss_mask, input_ids
```

### 训练前向（含Training-Time Testing）

```python
def forward(self, input_ids, attention_mask, loss_mask):
    # 数据准备
    hidden_states, target, loss_mask, input_ids = self.dataprepare(input_ids, attention_mask, loss_mask)
    
    # 投影三层特征
    hidden_states = self.fc(hidden_states)  # [batch, seq, hidden*3] → [batch, seq, hidden]
    
    # Training-Time Testing: 循环7步模拟自回归
    cache_hidden = [[], []]  # KV缓存
    plosses = []
    
    for idx in range(self.length):  # length = 7
        # 获取当前步的token嵌入
        inputs_embeds = self.embed_tokens(input_ids)
        
        # 草稿Decoder前向 (含attention cache累积)
        layer_outputs, cache_hidden = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,  # 累积历史KV
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        
        hidden_states = layer_outputs[0]
        
        # 计算当前步的预测损失
        logits = self.lm_head(self.norm(hidden_states))
        
        # 损失函数: 目标分布的交叉熵 (KL散度)
        # 先将目标logits过滤到draft词汇表空间
        target_head = target[..., self.t2d]   # 只保留draft词汇表中的token
        target_p = nn.Softmax(dim=2)(target_head)
        
        out_logp = nn.LogSoftmax(dim=2)(logits)
        plogp = target_p * out_logp
        loss = -torch.sum(position_mask * plogp, 2).mean()
        plosses.append(loss)
        
        # 准备下一步: 左移模拟自回归
        input_ids = padding(input_ids, left=False)
        target = padding(target, left=False)
        loss_mask = padding(loss_mask, left=False)
    
    return plosses, vlosses, acces
```

### 训练中的特殊注意力机制

训练版本的Attention与推理版本不同——它维护了一个**显式的KV缓存列表**`cache_hidden`：

```python
class LlamaAttention:  # 训练版
    def forward(self, hidden_states, cache_hidden, ...):
        # 投影当前步的QKV
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # 累积KV到缓存
        local_cache_k.append(key_states)
        local_cache_v.append(value_states)
        
        # 第一部分: 标准因果注意力（在当前序列内）
        attn_weights = torch.matmul(query_states, k0.transpose(2, 3)) / sqrt(d)
        attn_weights = attn_weights + attention_mask  # 因果掩码
        
        # 第二部分: 与历史步的交叉注意力（每个历史步只有一个token的KV）
        for i in range(1, len(cache_k)):
            ki = cache_k[i]
            attn_weightsi = (query_states * ki).sum(-1) / sqrt(d)
            attn_weights = torch.cat((attn_weights, attn_weightsi[..., None]), dim=-1)
        
        # 合并注意力输出
        attn_weights = softmax(attn_weights)
        attn_output = matmul(attn_weights0, v0) + Σ(attn_weightsi * vi)
```

这种设计让训练时的每一步都能attend到之前所有步的KV，真实模拟推理时的KV Cache机制。

### 损失函数与权重

```python
# 多步损失加权: 越靠前的预测步权重越大
ploss_weight = [0.8 ** i for i in range(len(plosses))]
# 权重: [1.0, 0.8, 0.64, 0.512, 0.4096, 0.3277, 0.2621]
ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
```

---

## 树状投机解码机制

### 树结构设计

EAGLE使用**动态构建的稀疏树**而非固定树结构。关键参数：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `total_tokens` | 59 (60-1) | 一棵草稿树中的最大token数 |
| `depth` | 7 | 树的最大深度 |
| `top_k` | 10 | 每个节点展开的候选数 |
| `threshold` | 1.0 | 累积概率阈值（对数空间） |

### 树的构建过程

```
              root (已确认的token)
             /  |  \   ...   \
           t1  t2  t3  ...  t_k     ← 第1层: top-k候选
          /|\ /|\
        ... ... ...                  ← 第2层: 全局top-k(从k*k个中选k个)
        |   |   |
       ...                           ← 第3~7层: 继续展开
```

每一层的候选选择使用**全局累积得分**排序：

```python
cu_scores = topk_p + scores[:, None]           # 当前得分 + 父节点得分
topk_cs = torch.topk(cu_scores.view(-1), top_k)  # 全局top-k
```

最终从所有层的所有候选中选出 `total_tokens` 个得分最高的token构成草稿树。

### 树形注意力掩码 (`tree_mask`)

```python
# 构建原则: 每个节点可以看到
# 1. 根节点（root）
# 2. 它到根路径上的所有祖先节点
tree_mask = torch.eye(total_tokens + 1).bool()
tree_mask[:, 0] = True  # 所有节点都能看到root
for i in range(total_tokens):
    tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])  # 继承父节点的可见性
```

### 检索索引 (`retrieve_indices`)

用于从树中提取所有**从根到叶子的完整路径**（候选序列）：

```python
# retrieve_indices: [leaf_num, max_depth]
# 每一行是一条从根到叶子的路径（索引列表）
# 例如: [[0, 1, 5, 12], [0, 2, 7, -1], ...]
# -1 表示路径较短时的padding
```

### 位置编码 (`tree_position_ids`)

每个树节点的位置ID由其深度决定：

```python
tree_position_ids = torch.sum(tree_mask, dim=1) - 1
# 根节点: 0, 第1层: 1, 第2层: 2, ...
# 最终 position_ids = tree_position_ids + input_ids.shape[1]
```

---

## Draft词汇表压缩机制

### 动机

目标模型（如Qwen3-8B）通常拥有大词汇表（151936 tokens），而大多数token在实际生成中很少出现。EAGLE-3将草稿模型的词汇表压缩到一个较小的子集（如32000 tokens），显著减少LM Head的计算量：

- 原始LM Head: `Linear(4096, 151936)` → 622M参数
- 压缩LM Head: `Linear(4096, 32000)` → 131M参数

### 构建过程 (`scandata`)

```python
def scandata(self, datapath, tokenizerpath):
    N = self.draft_vocab_size  # 目标草稿词汇表大小
    
    # 1. 统计训练数据中所有token的频率
    token_dict = Counter()
    for sample in dataset:
        for token_id in sample:
            if loss_mask[token_id] == 1:  # 只统计需要预测的位置
                token_dict[token_id] += 1
    
    # 2. 选择频率最高的N个token
    top_N = token_dict.most_common(N)
    used_tokens = sorted([key for key, freq in top_N])
    
    # 3. 构建Draft→Target映射
    # d2t[draft_id] = target_id - draft_id (偏移量)
    d2t = [used_tokens[i] - i for i in range(len(used_tokens))]
    
    # 4. 构建Target→Draft布尔掩码
    # t2d[target_id] = True if target_id in used_tokens
    t2d = [i in used_tokens for i in range(self.vocab_size)]
```

### 推理时的映射

```python
# Draft → Target: 将草稿预测映射回目标词汇表
if self.config.vocab_size != self.config.draft_vocab_size:
    target_token_id = draft_token_id + self.d2t[draft_token_id]

# Target → Draft: 训练时将目标logits过滤到Draft空间
target_head = target[..., self.t2d]  # 布尔索引，只保留Draft词汇表中的token
```

---

## 草稿Token验证的完整流程详解

本节详细拆解草稿模型生成的候选token是如何被目标模型验证、接受或拒绝的完整数据流。

### 验证全流程总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                     一轮完整的"草稿-验证"循环                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─── 上一轮输出 ─────────────────────────────────────────────┐    │
│  │  input_ids: 已确认的完整token序列                           │    │
│  │  hidden_state: 三层隐藏状态拼接 [1, accepted_len, H*3]     │    │
│  │  sample_token: 最新采样的token (即将作为tree root)          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│        │                                                            │
│        ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Step 1: 草稿模型生成候选树 (topK_genrate)                  │   │
│  │                                                               │   │
│  │  输入: hidden_state, input_ids + sample_token                │   │
│  │  输出:                                                        │   │
│  │    • draft_tokens  [1, total_tokens+1]  // 树中所有token      │   │
│  │    • tree_mask     [1, 1, N+1, N+1]     // 树形注意力掩码    │   │
│  │    • tree_position_ids [N+1]            // 各节点的深度       │   │
│  │    • retrieve_indices [leaf_num, max_depth] // 路径索引       │   │
│  └──────────────────────┬──────────────────────────────────────┘   │
│                         │                                           │
│                         ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Step 2: 设置目标模型的tree_mask                             │   │
│  │                                                               │   │
│  │  model.base_model.model.tree_mask = tree_mask                │   │
│  │  // 目标模型在self-attention中使用此mask                      │   │
│  │  // 让每个树节点只能attend到其祖先节点                        │   │
│  └──────────────────────┬──────────────────────────────────────┘   │
│                         │                                           │
│                         ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Step 3: 目标模型并行前向验证 (tree_decoding)                │   │
│  │                                                               │   │
│  │  输入:                                                        │   │
│  │    • draft_tokens [1, N+1] — 整棵树的token序列               │   │
│  │    • position_ids = tree_position_ids + len(input_ids)       │   │
│  │    • past_key_values — 已有KV Cache                          │   │
│  │                                                               │   │
│  │  目标模型执行:                                                │   │
│  │    outputs, tree_logits, _ = model(draft_tokens, ...)        │   │
│  │    // tree_logits: [1, N+1, vocab_size]                      │   │
│  │    // 每个树节点位置都有完整的logits输出                      │   │
│  │                                                               │   │
│  │  提取路径logits:                                              │   │
│  │    logits = tree_logits[0, retrieve_indices]                 │   │
│  │    // logits: [leaf_num, max_depth, vocab_size]              │   │
│  │    // 每条候选路径的每个位置的目标模型logits                   │   │
│  │                                                               │   │
│  │  EAGLE-3额外操作:                                            │   │
│  │    hidden_state_new = cat(outputs["hidden_states"], dim=-1)  │   │
│  │    // 提取三层隐藏状态用于下一轮草稿生成                      │   │
│  └──────────────────────┬──────────────────────────────────────┘   │
│                         │                                           │
│                         ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Step 4: 构建候选路径 (candidates)                           │   │
│  │                                                               │   │
│  │  // 追加一个padding token到draft_tokens末尾                   │   │
│  │  draft_tokens = cat(draft_tokens, [-1])                      │   │
│  │                                                               │   │
│  │  // 按retrieve_indices提取每条从root到leaf的路径              │   │
│  │  candidates = draft_tokens[0, retrieve_indices]              │   │
│  │  // candidates: [leaf_num, max_depth]                        │   │
│  │  // 每行 = 一条候选token序列 [root, child, ..., leaf]        │   │
│  └──────────────────────┬──────────────────────────────────────┘   │
│                         │                                           │
│                         ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Step 5: 逐位置验证 (evaluate_posterior)                     │   │
│  │                                                               │   │
│  │  输入:                                                        │   │
│  │    • logits:     [leaf_num, max_depth, vocab_size]            │   │
│  │    • candidates: [leaf_num, max_depth]                        │   │
│  │                                                               │   │
│  │  输出:                                                        │   │
│  │    • best_candidate: 最优路径索引                              │   │
│  │    • accept_length:  该路径上被接受的token数                   │   │
│  │    • sample_p:       下一个token的采样分布                    │   │
│  │                                                               │   │
│  │  详见下方 "贪心验证" / "采样验证" 小节                        │   │
│  └──────────────────────┬──────────────────────────────────────┘   │
│                         │                                           │
│                         ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Step 6: 接受并更新状态 (update_inference_inputs)            │   │
│  │                                                               │   │
│  │  6a. 定位接受路径在KV Cache中的位置                           │   │
│  │      select_indices = retrieve_indices[best, :accept+1]      │   │
│  │                       + prev_input_len                        │   │
│  │                                                               │   │
│  │  6b. 追加接受的token到序列                                    │   │
│  │      input_ids = cat(input_ids, accepted_tokens)             │   │
│  │                                                               │   │
│  │  6c. 整理KV Cache: 把接受路径的KV复制到连续位置               │   │
│  │      kv[..., prev:prev+len, :] = kv[..., select_indices, :] │   │
│  │                                                               │   │
│  │  6d. 采样bonus token (从最后接受位置的logits)                 │   │
│  │      token = argmax(sample_p)  或  multinomial(sample_p)     │   │
│  │                                                               │   │
│  │  6e. 提取接受位置的隐藏状态给下一轮草稿模型                   │   │
│  │      accept_hidden = hidden_new[best, :accept+1]             │   │
│  │                                                               │   │
│  │  6f. 草稿模型立即生成下一棵候选树                              │   │
│  │      draft_tokens, ... = topK_genrate(accept_hidden, ...)    │   │
│  └──────────────────────┬──────────────────────────────────────┘   │
│                         │                                           │
│                         ▼                                           │
│                  返回 Step 1 继续循环                                │
└─────────────────────────────────────────────────────────────────────┘
```

### Step 3 详解：目标模型如何一次验证整棵树

投机解码的关键洞察是：**目标模型不需要逐token验证**，它可以利用树形注意力掩码在一次前向传播中同时计算所有候选节点的logits。

#### 树形注意力掩码的作用

假设草稿树如下（top_k=3, depth=2, 共9个候选+1个root = 10个位置）：

```
              root(t₀)                    depth=0
             /    |    \
           t₁    t₂    t₃               depth=1
          /|\   /|\   /|\
        t₄t₅t₆ t₇t₈t₉ t₁₀t₁₁t₁₂       depth=2
```

对应的 `tree_mask` (1=可见, 0=不可见)：

```
         t₀ t₁ t₂ t₃ t₄ t₅ t₆ t₇ t₈ t₉
    t₀ [  1  0  0  0  0  0  0  0  0  0 ]   ← root只看到自己
    t₁ [  1  1  0  0  0  0  0  0  0  0 ]   ← t₁看到root和自己
    t₂ [  1  0  1  0  0  0  0  0  0  0 ]   ← t₂看到root和自己
    t₃ [  1  0  0  1  0  0  0  0  0  0 ]
    t₄ [  1  1  0  0  1  0  0  0  0  0 ]   ← t₄看到root→t₁→自己
    t₅ [  1  1  0  0  0  1  0  0  0  0 ]   ← t₅看到root→t₁→自己
    t₆ [  1  1  0  0  0  0  1  0  0  0 ]
    t₇ [  1  0  1  0  0  0  0  1  0  0 ]   ← t₇看到root→t₂→自己
    t₈ [  1  0  1  0  0  0  0  0  1  0 ]
    t₉ [  1  0  0  1  0  0  0  0  0  1 ]   ← t₉看到root→t₃→自己
```

**关键**: 同层但不同父节点的节点之间互相不可见（t₄看不到t₇，因为它们属于不同分支）。这保证了每条路径的logits输出与该路径单独送入目标模型时完全一致。

#### 目标模型实际执行的操作

```python
def tree_decoding(model, tree_candidates, past_key_values,
                  tree_position_ids, input_ids, retrieve_indices):
    
    # 1. 计算position_ids: 每个节点的绝对位置 = 相对深度 + 已有序列长度
    #    例: input_ids长度=100, 则 root位置=100, depth1=101, depth2=102 ...
    position_ids = tree_position_ids + input_ids.shape[1]
    #    注意: 同一层的所有节点共享相同的position_id
    #    如 t₁,t₂,t₃ 都是 position=101
    
    # 2. 目标模型前向: 将tree_mask嵌入注意力计算
    #    在self-attention中:
    #    attn_mask[..., -tree_h:, -tree_w:] 的 0 位置 → -inf (masked)
    outputs, tree_logits, hidden_state = model(
        tree_candidates,         # [1, N+1] — 整棵树一次性送入
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    # tree_logits: [1, N+1, vocab_size]
    # tree_logits[0, i] = 目标模型在位置i输出的logits
    # 对于位置i, logits[i]的含义是: 
    #   "给定root到节点i路径上的所有token作为上下文, 目标模型预测的下一个token分布"
    
    # 3. 按retrieve_indices提取每条候选路径的logits
    logits = tree_logits[0, retrieve_indices]
    # logits: [leaf_num, max_depth, vocab_size]
    # logits[p, d] = 路径p在深度d处的目标模型logits
    
    return logits, hidden_state, outputs
```

#### 为什么能一次验证？

传统自回归逐token解码时，每步只输入1个token。而投机解码的验证阶段利用了两个关键性质：

1. **Transformer的并行性**: 给定注意力掩码，模型可以同时处理多个位置，只要掩码正确限制了信息流
2. **树掩码的因果性**: 每个节点只能看到其祖先路径，这等价于该路径被独立送入模型时的注意力模式

因此，一次前向传播（~60个token）产生的logits与将每条路径单独送入模型N次的结果完全一致，但计算开销是一次前向的成本。

### Step 4 详解：从树到候选路径的映射

#### retrieve_indices 的构建

`retrieve_indices` 是验证阶段最关键的数据结构，它将树形结构中的每条从根到叶子的路径提取为线性序列：

```python
# 构建过程:
# 1. 找出所有叶子节点（没有子节点的节点）
noleaf_index = torch.unique(mask_index)  # 所有非叶节点
leaf_num = total_tokens - (len(noleaf_index) - 1)

# 2. 对每个叶子节点, 沿父指针回溯到root, 记录路径
for i in range(total_tokens + 1):
    if i not in noleaf_index:  # 是叶子节点
        cid = i
        depth = position_ids_list[i]
        for j in reversed(range(depth + 1)):
            retrieve_indices[rid][j] = cid    # 从叶子到根依次记录
            cid = mask_index_list[cid - 1]    # 跳到父节点
```

#### 具体数值示例

以一棵小树为例（top_k=3, depth=2, total_tokens=9）:

```
树结构                     tree中的索引
       root(确认的token)     idx=0
      /     |     \
    "the"  "a"  "an"        idx=1,2,3
    / \     |
 "cat" "dog" "big"          idx=4,5,6

draft_tokens = [root, "the", "a", "an", "cat", "dog", "big"]
                 0      1     2    3     4      5      6
```

此时 `retrieve_indices` 提取出所有根到叶子路径:

```
retrieve_indices = [
    [0, 1, 4],    ← root → "the" → "cat"    (路径0)
    [0, 1, 5],    ← root → "the" → "dog"    (路径1)
    [0, 2, 6],    ← root → "a"   → "big"    (路径2)
    [0, 3, -1],   ← root → "an"             (路径3, depth=1, padding=-1)
]
```

经过 `candidates = draft_tokens[0, retrieve_indices]` 后:

```
candidates = [
    [root, "the", "cat"],     ← 候选序列0
    [root, "the", "dog"],     ← 候选序列1
    [root, "a",   "big"],     ← 候选序列2
    [root, "an",  padding],   ← 候选序列3
]
```

同时 `logits = tree_logits[0, retrieve_indices]`:

```
logits[0] = [logits_at_root,      logits_at_"the",   logits_at_"cat"  ]
logits[1] = [logits_at_root,      logits_at_"the",   logits_at_"dog"  ]
logits[2] = [logits_at_root,      logits_at_"a",     logits_at_"big"  ]
logits[3] = [logits_at_root,      logits_at_"an",    logits_at_padding]
```

注意 `logits[0]` 和 `logits[1]` 的前两列完全相同（它们共享 root→"the" 的路径），这正是树形结构共享前缀的优势。

### Step 5 详解：验证判定算法 (`evaluate_posterior`)

#### 贪心验证（temperature=0）完整流程

```
输入:
  logits:     [leaf_num, max_depth, vocab_size]  — 目标模型在每个位置的完整logits
  candidates: [leaf_num, max_depth]              — 每条路径的候选token序列

验证过程:
  ┌─────────────────────────────────────────────────────────────────┐
  │  对每条路径 p (p = 0, 1, ..., leaf_num-1):                     │
  │                                                                 │
  │  position 0:  candidates[p,1] (第1个草稿token)                 │
  │               vs argmax(logits[p,0]) (目标模型在root位置的预测) │
  │               匹配? → match[p,0] = 1, 否则 = 0                │
  │                                                                 │
  │  position 1:  candidates[p,2] (第2个草稿token)                 │
  │               vs argmax(logits[p,1]) (目标模型在pos1的预测)     │
  │               匹配? → match[p,1] = 1, 否则 = 0                │
  │                                                                 │
  │  ...                                                            │
  │                                                                 │
  │  accept_length[p] = 从match[p,0]开始连续为1的个数               │
  │                     (用cumprod实现: 一旦遇到0后续全部为0)        │
  └─────────────────────────────────────────────────────────────────┘

  best_candidate = argmax(accept_length)   // 选接受最长的路径
  final_accept   = accept_length[best_candidate]
```

具体代码实现：

```python
# 核心: 将候选token与目标模型的argmax预测逐位对比
# candidates[:, 1:] — 去掉root (root已确认), 得到 [leaf_num, max_depth-1]
# logits[:, :-1]    — 去掉最后一列 (最后位置的logits用于采样下一个token)

posterior_mask = (
    candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)
).int()
# posterior_mask[p, d] = 1 当且仅当 路径p在深度d+1的草稿token
#                         等于目标模型在深度d位置的argmax预测

# 连续匹配长度: cumprod使得一旦某位置不匹配(0), 后续全归零
candidates_accept_length = torch.cumprod(posterior_mask, dim=1).sum(dim=1)
# candidates_accept_length[p] = 路径p上连续匹配的token数

accept_length = candidates_accept_length.max()
best_candidate = torch.argmax(candidates_accept_length)
```

**数值示例**：

```
假设 top_k=3, depth=2, 有4条候选路径:

目标模型argmax:
  position 0 (root): argmax → "the"
  position 1 ("the" branch): argmax → "cat"
  position 1 ("a" branch):   argmax → "big"
  position 1 ("an" branch):  argmax → "red"

candidates:                  posterior_mask:           cumprod:      accept_len:
路径0: [root, "the", "cat"]  [1("the"=✓), 1("cat"=✓)]  [1, 1]       2 ← 最长!
路径1: [root, "the", "dog"]  [1("the"=✓), 0("dog"≠✓)]  [1, 0]       1
路径2: [root, "a",   "big"]  [0("a"≠✓),   —         ]  [0, 0]       0
路径3: [root, "an",  pad  ]  [0("an"≠✓),  —         ]  [0, 0]       0

→ best_candidate = 0, accept_length = 2
→ 接受 "the" 和 "cat"
→ 从 logits[0, 2] (路径0, 深度2) 采样/argmax得到下一个token
  (这个token叫做 "bonus token", 是免费获得的第3个token)
```

这一轮共接受了 **accept_length + 1 = 3** 个新token (2个匹配 + 1个bonus), 而原始自回归解码需要3次目标模型前向。

#### 采样验证（temperature>0）完整流程

采样模式需要保证输出分布与目标模型的原始采样分布完全一致（无损性质）。使用的是**修正拒绝采样**算法：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        采样验证详细流程                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  初始化:                                                            │
│    accept_length = 1  (root已接受)                                  │
│    accept_cand = [root_token]                                       │
│                                                                     │
│  对于每个深度 i = 1, 2, ..., max_depth-1:                           │
│    │                                                                │
│    │  ┌────────────────────────────────────────────────────┐       │
│    │  │ 1. 找到所有与已接受前缀匹配的候选路径               │       │
│    │  │    is_eq = (candidates[:, :i] == accept_cand).all() │       │
│    │  │    // 只有前缀完全一致的路径才需要验证               │       │
│    │  └──────────────┬─────────────────────────────────────┘       │
│    │                 │                                              │
│    │                 ▼                                              │
│    │  ┌────────────────────────────────────────────────────┐       │
│    │  │ 2. 获取目标分布                                     │       │
│    │  │    fi = 第一个匹配路径的索引                         │       │
│    │  │    gt_logits = logits[fi, i-1]                      │       │
│    │  │    gt_logits = logits_processor(gt_logits)          │       │
│    │  │    // 应用temperature/top_p/top_k变换               │       │
│    │  │    gtp = softmax(gt_logits)                         │       │
│    │  │    // gtp[v] = 目标模型产生token v的概率             │       │
│    │  └──────────────┬─────────────────────────────────────┘       │
│    │                 │                                              │
│    │                 ▼                                              │
│    │  ┌────────────────────────────────────────────────────┐       │
│    │  │ 3. 对每个匹配路径的候选token x 尝试接受              │       │
│    │  │                                                      │       │
│    │  │    for 每个匹配路径 j:                               │       │
│    │  │      x = candidates[j, i]                           │       │
│    │  │      if x 已被尝试过 → 跳过                         │       │
│    │  │                                                      │       │
│    │  │      r = random()           // 均匀随机数 [0,1)      │       │
│    │  │      p(x) = gtp[x]         // 目标模型概率           │       │
│    │  │      q(x) = 1.0            // 草稿模型概率(简化)     │       │
│    │  │      accept_rate = p(x)/q(x) = p(x)                │       │
│    │  │                                                      │       │
│    │  │      if r ≤ accept_rate:                             │       │
│    │  │        ✅ 接受! accept_length += 1                   │       │
│    │  │        accept_cand = append(accept_cand, x)         │       │
│    │  │        best_candidate = j                            │       │
│    │  │        → break, 进入下一深度                         │       │
│    │  │                                                      │       │
│    │  │      else:                                           │       │
│    │  │        ❌ 拒绝! 修正目标分布:                        │       │
│    │  │        gtp[x] = 0; gtp = gtp / gtp.sum()            │       │
│    │  │        // 移除已拒绝token的概率, 重新归一化          │       │
│    │  │        → continue, 尝试下一个候选token               │       │
│    │  └──────────────┬─────────────────────────────────────┘       │
│    │                 │                                              │
│    │  如果所有候选都被拒绝 → 结束验证                               │
│    │  如果有候选被接受 → 继续下一深度                                │
│    │                                                                │
│  结束深度循环                                                       │
│                                                                     │
│  ┌────────────────────────────────────────────────────────┐        │
│  │ 4. 确定下一个token的采样分布 (sample_p)                │        │
│  │                                                        │        │
│  │  if 最后一步发生了分布修正(adjustflag=True):            │        │
│  │    sample_p = 修正后的gtp                              │        │
│  │    // 从修正分布中采样, 保证整体分布正确                │        │
│  │                                                        │        │
│  │  else:                                                 │        │
│  │    gt_logits = logits[best, accept_length-1]           │        │
│  │    sample_p = softmax(logits_processor(gt_logits))     │        │
│  │    // 标准采样: 从最后接受位置的目标分布中采样          │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                     │
│  返回 (best_candidate, accept_length-1, sample_p)                  │
└─────────────────────────────────────────────────────────────────────┘
```

**为什么采样验证能保证无损?**

拒绝采样的数学保证：

$$P(\text{accept } x) = \min\left(1, \frac{p(x)}{q(x)}\right)$$

当某个token $x$ 被拒绝后，目标分布被修正为：

$$p'(v) = \begin{cases} 0 & \text{if } v = x \\ \frac{p(v)}{\sum_{v' \neq x} p(v')} & \text{otherwise} \end{cases}$$

下一个token从修正分布 $p'$ 中采样（或尝试另一个候选）。这个过程可以证明最终产生的token分布与直接从目标分布 $p$ 采样完全等价。

### Step 6 详解：接受后的状态更新

验证完成后，需要同步更新多个状态：

```
┌───────────────────────────────────────────────────────────────────┐
│                        状态更新详细流程                            │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  输入: best_candidate=0, accept_length=2                         │
│  当前: input_ids = [prompt..., prev_tokens...]  (长度=prev_len)  │
│                                                                   │
│  ┌─ 6a. 定位接受的KV Cache ──────────────────────────────────┐  │
│  │                                                             │  │
│  │  select_indices = retrieve_indices[0, :3] + prev_len       │  │
│  │  // = [0, 1, 4] + prev_len                                │  │
│  │  // 这是接受路径上各节点在KV Cache中的实际位置              │  │
│  │                                                             │  │
│  │  树形验证时, KV Cache中的位置:                              │  │
│  │    prev_len+0: root的KV                                    │  │
│  │    prev_len+1: "the"的KV  ← 接受                          │  │
│  │    prev_len+2: "a"的KV    ← 不在接受路径上                 │  │
│  │    prev_len+3: "an"的KV   ← 不在接受路径上                 │  │
│  │    prev_len+4: "cat"的KV  ← 接受                          │  │
│  │    prev_len+5: "dog"的KV  ← 不在接受路径上                 │  │
│  │    ...                                                      │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌─ 6b. 追加token到序列 ─────────────────────────────────────┐  │
│  │                                                             │  │
│  │  accepted_tokens = candidates[0, :3] = [root, "the", "cat"]│  │
│  │  input_ids = cat(input_ids, accepted_tokens)               │  │
│  │  // 序列长度增加了 accept_length + 1 = 3                   │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌─ 6c. 整理KV Cache (关键!) ─────────────────────────────────┐ │
│  │                                                             │  │
│  │  验证时KV Cache是非连续的:                                  │  │
│  │    [prompt KVs...][root][the][a][an][cat][dog][...]         │  │
│  │                     ↑         ↑↑         ↑                  │  │
│  │                     需要     抛弃        需要                │  │
│  │                                                             │  │
│  │  需要将接受路径的KV复制到连续位置:                          │  │
│  │    src = kv[..., select_indices, :]                         │  │
│  │    dst = kv[..., prev_len : prev_len+3, :]                 │  │
│  │    dst.copy_(src)                                           │  │
│  │                                                             │  │
│  │  整理后:                                                    │  │
│  │    [prompt KVs...][root][the][cat]                          │  │
│  │    ← 连续且正确 →                                          │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌─ 6d. 采样bonus token ─────────────────────────────────────┐  │
│  │                                                             │  │
│  │  // sample_p来自evaluate_posterior的最后输出                │  │
│  │  // = logits[best_candidate, accept_length] 的softmax      │  │
│  │  // 即目标模型在最后接受位置预测的"下一个"token分布         │  │
│  │                                                             │  │
│  │  贪心: token = argmax(sample_p)                            │  │
│  │  采样: token = multinomial(sample_p)                       │  │
│  │                                                             │  │
│  │  这个token是"免费"获得的 (无需额外目标模型前向)            │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌─ 6e. 提取隐藏状态给下轮草稿模型 ──────────────────────────┐  │
│  │                                                             │  │
│  │  // hidden_state_new 是验证时目标模型产生的三层隐藏状态     │  │
│  │  // shape: [1, N+1, hidden*3]                              │  │
│  │                                                             │  │
│  │  // 按retrieve_indices重排, 再取接受路径                    │  │
│  │  retrieve_hidden = hidden_state_new[:, retrieve_indices]   │  │
│  │  accept_hidden = retrieve_hidden[:, best, :accept+1]      │  │
│  │  // accept_hidden: [1, accept_length+1, hidden*3]          │  │
│  │                                                             │  │
│  │  这些隐藏状态直接传给topK_genrate作为下轮的输入             │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌─ 6f. 立即生成下一棵草稿树 ────────────────────────────────┐  │
│  │                                                             │  │
│  │  draft_tokens, retrieve_indices, tree_mask, ... =          │  │
│  │    model.ea_layer.topK_genrate(                             │  │
│  │        accept_hidden,                                       │  │
│  │        input_ids = cat(input_ids, bonus_token),             │  │
│  │        ...                                                  │  │
│  │    )                                                        │  │
│  │  // 草稿模型利用接受路径的隐藏状态, 增量更新自己的KV Cache  │  │
│  │  // 然后从最后位置开始展开新的候选树                         │  │
│  └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

### 完整数值走查示例

以一个简单的实际场景走查整个验证流程：

```
=== 初始状态 ===
input_ids = ["<s>", "Hello", ",", "how"]        (长度=4)
目标模型已确认最新token: "are" (通过prefill采样)
input_ids = ["<s>", "Hello", ",", "how", "are"]  (长度=5)

=== Step 1: 草稿模型生成候选树 (top_k=3, depth=3) ===

        "are" (root, idx=0)
       /      |       \
   "you"    "we"    "they"          (idx=1,2,3)
   / | \     |
"?"  "do" "to" "go"                (idx=4,5,6,7)
      |
    "ing"                           (idx=8)

draft_tokens = [are, you, we, they, ?, do, to, go, ing]
total_tokens = 8 (不含root)

tree_position_ids = [0, 1, 1, 1, 2, 2, 2, 2, 3]

=== Step 2: 设置tree_mask ===
(每个节点只能看到自己的祖先路径)

=== Step 3: 目标模型一次前向 ===
输入: 9个token (连同KV Cache一起)
输出: 9个位置的logits

=== Step 4: 提取候选路径 ===
叶子节点: "?" (idx=4), "to" (idx=6), "they" (idx=3), "go" (idx=7), "ing" (idx=8)

retrieve_indices:
  路径0: [0, 1, 4, -1]     → [are, you, ?,   pad ]
  路径1: [0, 1, 6, -1]     → [are, you, to,  pad ]
  路径2: [0, 3, -1, -1]    → [are, they, pad, pad]
  路径3: [0, 2, 7, -1]     → [are, we,  go,  pad ]
  路径4: [0, 1, 5, 8]      → [are, you, do,  ing ]

logits 同样按此索引提取

=== Step 5: 贪心验证 ===

目标模型在各位置的argmax:
  logits[root]       → argmax = "you"   ← 目标模型也认为"are"之后最可能是"you"
  logits["you"]      → argmax = "do"    ← 目标模型认为"you"之后最可能是"do"
  logits["we"]       → argmax = "can"   ← 目标模型认为"we"之后最可能是"can"
  logits["they"]     → argmax = "are"   
  logits["do"]       → argmax = "ing"   ← 目标模型认为"do"之后最可能是"ing"
  logits["?"]        → argmax = "</s>"  
  logits["to"]       → argmax = "help"  
  logits["go"]       → argmax = "home"  

验证每条路径:
  路径0 [are,you,?,pad]:    "you"=✓, "?"≠"do"=✗     → accept=1
  路径1 [are,you,to,pad]:   "you"=✓, "to"≠"do"=✗    → accept=1
  路径2 [are,they,pad,pad]: "they"≠"you"=✗           → accept=0
  路径3 [are,we,go,pad]:    "we"≠"you"=✗             → accept=0
  路径4 [are,you,do,ing]:   "you"=✓, "do"=✓, "ing"=✓ → accept=3 ★最长!

→ best_candidate = 4, accept_length = 3
→ 接受: "you", "do", "ing" (3个token!)
→ bonus token: 从 logits["ing"] 的argmax采样 → 假设得到 "?"

=== Step 6: 更新状态 ===
input_ids = ["<s>", "Hello", ",", "how", "are", "you", "do", "ing"]
bonus token "?" 在下一轮开始时使用
本轮实际生成了 4 个新token (3 accepted + 1 bonus)
而目标模型只做了 1 次前向！
```

### 验证正确性的数学保证

投机解码的核心定理：

> **定理**: 对于任意输入序列 $x_{1:t}$，投机解码产生token $x_{t+1}$ 的分布 $\tilde{P}(x_{t+1} | x_{1:t})$ 与目标模型的自回归分布 $P(x_{t+1} | x_{1:t})$ 完全相同。

**贪心模式证明**: 若 $\text{argmax}(P) = x_{draft}$，则接受。否则按 $P$ 的argmax选择。两种情况下输出都等于 $\text{argmax}(P)$，与原始贪心解码一致。

**采样模式证明**: 拒绝采样保证每一步的边际分布等于目标分布 $P$。当草稿token被拒绝时，修正分布 $P'$ 确保下一个采样的token与原始分布一致。详细证明见EAGLE论文附录。

---

## 性能分析

### 单步开销分析

| 操作 | 计算量 | 说明 |
|------|--------|------|
| 目标模型验证 | ~60 tokens并行前向 | 主要开销，约等于60个token的一次前向 |
| 草稿模型生成 | 7步自回归 × 单层Decoder | 约为目标模型单步的7/32 ≈ 22% |
| 树构建/验证 | 极小 | CPU侧整数操作 |
| KV Cache更新 | 少量内存拷贝 | scatter操作 |

### 加速比理论分析

设平均接受长度为 $\tau$，目标模型单次前向处理 $n$ 个token的时间为 $T_{target}(n)$，草稿模型生成时间为 $T_{draft}$：

$$\text{加速比} = \frac{\tau \cdot T_{target}(1)}{T_{target}(n) + T_{draft}}$$

由于GPU的并行性，$T_{target}(n) \approx T_{target}(1)$（当 $n \leq 60$ 时），因此：

$$\text{加速比} \approx \frac{\tau}{1 + T_{draft}/T_{target}(1)}$$

EAGLE-3通过多层特征融合提升 $\tau$（从EAGLE-1的~3提升到~5.6），同时保持 $T_{draft}$ 相对较小。

### 自动total_token调优

```python
if total_token == -1:
    cans = [40, 48, 50, 56, 60]
    x = [1, 1.05, 1.07, 1.1, 1.13]  # 归一化因子
    times = []
    for length in cans:
        input_ids = torch.randint(0, vocab_size, (1, length))
        # 运行20次取平均时间
        for _ in range(20):
            outputs = model.base_model(input_ids)
        times.append(time / x[i])
    total_token = cans[times.index(min(times))]
```

---

## 与OrinMLLM集成分析

OrinMLLM已有部分EAGLE-3适配工作（见 `tools/convert_eagle3.py`），其二进制格式包含：

### 权重导出格式

```
Header (44 bytes):
  Magic: b'EGL3'
  Version, num_layers, hidden_size, num_heads, num_kv_heads
  intermediate_size, target_vocab_size, draft_vocab_size
  total_elements

Data:
  D2T mapping: int32[draft_vocab_size]
  
  Weights (all FP16):
    fc.weight:                          [hidden_size, hidden_size * 3]
    norm.weight:                        [hidden_size]
    midlayer.input_layernorm.weight:    [hidden_size]
    midlayer.self_attn.q_proj.weight:   [hidden_size, hidden_size * 2]
    midlayer.self_attn.k_proj.weight:   [kv_dim, hidden_size * 2]
    midlayer.self_attn.v_proj.weight:   [kv_dim, hidden_size * 2]
    midlayer.self_attn.o_proj.weight:   [hidden_size, hidden_size]
    midlayer.post_attention_layernorm:  [hidden_size]
    midlayer.mlp.gate_proj.weight:      [intermediate_size, hidden_size]
    midlayer.mlp.up_proj.weight:        [intermediate_size, hidden_size]
    midlayer.mlp.down_proj.weight:      [hidden_size, intermediate_size]
    lm_head.weight:                     [draft_vocab_size, hidden_size]
    final_layernorm.weight:             [hidden_size]
```

### C++实现要点

在OrinMLLM的C++推理引擎中集成EAGLE-3需关注：

1. **三层隐藏状态提取**: 修改基座模型的forward，在指定层（第2层、中间层、倒数第3层）缓存隐藏状态并拼接
2. **草稿模型前向**: 实现单层Decoder（含双流输入的注意力机制），FC投影，LM Head
3. **树形KV Cache管理**: 针对草稿树的非连续KV Cache的高效scatter/gather操作
4. **Draft词汇表映射**: d2t/t2d的高效查表实现
5. **动态树构建与验证**: 包括topK选择、树形mask构建、路径提取、贪心/采样验证

### 关键优化机会

- **CUDA Kernel融合**: 将FC投影+RMSNorm+QKV投影融合为一个kernel
- **Draft LM Head优化**: 利用压缩词汇表（32K）减少output projection计算
- **树形注意力**: 针对稀疏树mask的定制FlashAttention变体
- **KV Cache管理**: 零拷贝的树形KV Cache裁剪策略

---

## 总结

EAGLE-3通过三个核心创新实现了目前最快的投机解码加速：

1. **多层特征融合**: 用低/中/高三层隐藏状态替代单一次顶层特征，提供更丰富的语义信息给草稿模型，提升预测准确率
2. **Training-Time Testing**: 训练过程模拟推理时的自回归生成，使草稿模型在训练期间就学会处理特征移位模式
3. **Draft词汇表压缩**: 将预测空间缩小到高频词汇子集，减少计算开销

整体架构保持了投机解码的**无损性质**——由于目标模型始终做最终验证，生成结果的分布与原始自回归解码完全一致（贪心模式下完全一致，采样模式下通过拒绝采样保证分布一致）。
