from transformers import AutoTokenizer, AutoModel
local_model_path = '/Users/haoyunou/PycharmProjects/private-models/chatglm3-6b'
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained(local_model_path, trust_remote_code=True).to('mps')
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)

