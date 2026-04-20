from faster_whisper import WhisperModel
import ollama

# 1. 轉錄設定
input_file = "test_output.wav" # 假設這是英文音檔
model = WhisperModel("small", device="cpu", compute_type="int8") # 確保穩定

print("第一階段：正在轉錄原文...")
segments, info = model.transcribe(input_file, beam_size=5)

full_text = ""
for segment in segments:
    full_text += segment.text + " "

print(f"轉錄完成，原文內容：\n{full_text}\n")

# 2. 翻譯設定 (串接 Ollama)
print("第二階段：正在進行 AI 繁體中文翻譯...")

response = ollama.chat(model='llama3', messages=[
  {
    'role': 'system',
    'content': '你是一個專業的翻譯官，請將用戶提供的內容翻譯成流暢的「繁體中文」，不要有中國用語。',
  },
  {
    'role': 'user',
    'content': full_text,
  },
])

print("-" * 30)
print("翻譯結果：")
print(response['message']['content'])
print("-" * 30)