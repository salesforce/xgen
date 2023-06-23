from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("checkpoints/xgen-7B")
model = AutoModelForCausalLM.from_pretrained("checkpoints/xgen-7B", torch_dtype=torch.bfloat16, revision="sharded")
inputs = tokenizer("The world is", return_tensors="pt")
sample = model.generate(**inputs, max_length=128)
print(tokenizer.decode(sample[0]))
