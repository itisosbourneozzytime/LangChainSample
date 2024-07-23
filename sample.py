import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
MODEL_NAME = "IlyaGusev/saiga_mistral_7b_lora"

# Загружаем модель
config = PeftConfig.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
	config.base_model_name_or_path,
	torch_dtype=torch.float16,
	device_map="auto"
)
model = PeftModel.from_pretrained(
	model,
	MODEL_NAME,
	torch_dtype=torch.float16
)
model.eval()

# Определяем токенайзер
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)

# Функция для обработки запросов
def generate(model, tokenizer, prompt, generation_config):
	data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
	data = {k: v.to(model.device) for k, v in data.items()}
	output_ids = model.generate(
    	**data,
    	generation_config=generation_config
	)[0]
	output_ids = output_ids[len(data["input_ids"][0]):]
	output = tokenizer.decode(output_ids, skip_special_tokens=True)
	return output.strip()

# Формируем запрос
PROMT_TEMPLATE = '<s>system\nТы — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.</s><s>user\n{inp}</s><s>bot\n'
inp = 'Какое расстояние до Луны?'
prompt = PROMT_TEMPLATE.format(inp=inp)

# Отправляем запрос в llm
output = generate(model, tokenizer, prompt, generation_config)

print(output)