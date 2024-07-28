import ollama

# Existing code ...
messages = "Hi"

# 使用Ollama进行聊天
response = ollama.chat(model="llama3.1", messages=messages)
print("Response from Ollama API:", response)
