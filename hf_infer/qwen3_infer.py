from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import transformers
import modelscope

print(f"PyTorch 版本: {torch.__version__}")
print(f"Transformers 版本: {transformers.__version__}")
print(f"ModelScope 版本: {modelscope.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")

model_name = "/mnt/ssd/QwenModels/Qwen3-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # 根据模型和硬件情况选择合适的数据类型
    attn_implementation="eager"  # 使用eager模式以支持更长的上下文
).eval()

# prepare the model input
# prompt = "你好!，请简讲解一下CUDA的用法。" 
# prompt = "你好，请介绍一下你自己！"
prompt = "什么是CUDA？"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
# 先设置较小的max_new_tokens进行测试
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024,  # 先设置为100，看是否能够快速生成
    do_sample=False,  # 为了快速测试，可以先使用贪婪解码
)

print("生成完成，开始解码...")
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)