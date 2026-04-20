import os
import time
from azure.core.credentials import AzureKeyCredential
from azure.ai.transcription import TranscriptionClient
from azure.ai.transcription.models import TranscriptionContent, TranscriptionOptions, TranscriptionDiarizationOptions
from dotenv import load_dotenv

# 1. 載入 .env
load_dotenv()
# 1. 設定資源資訊
endpoint = os.getenv("AZURE_ENDPOINT")
api_key = os.getenv("AZURE_API_KEY")

client = TranscriptionClient(
   endpoint=endpoint, credential=AzureKeyCredential(api_key)
)
audio_file_path= r'C:\Users\iec150094\STT_Test\czech_test_60s.wav'
with open(audio_file_path, "rb") as audio_file:
    # 確保這裡每一行前面的空格（縮排）都是一致的
    diarization_options = TranscriptionDiarizationOptions(
        enabled=True,
        max_speakers=2
    )
    
    # 修正縮排：這行開頭必須跟上面的 diarization_options 對齊
    #my_phrases = ["數智開發", "Inventec", "Instella", "Hayley"]
    
    # 2. 修改參數名稱為 phrase_list (去掉了 s)
    options = TranscriptionOptions(
        locales=["cs-CZ", "en-US"], 
        diarization_options=diarization_options,
        # 注意：這裡改成 phrase_list，並確保傳入的格式正確
        #phrase_list={"phrases": my_phrases} 
    )
    
    print("Sending transcription request (with Phrase List)...")
    start_time = time.time()
    result = client.transcribe(TranscriptionContent(definition=options, audio=audio_file))
    end_time = time.time()
    duration = end_time - start_time
    print(f"Transcription completed. Time taken: {duration:.2f} seconds.\n")
    print(f"API Time taken: {duration:.2f} seconds\n")
    # 先輸出完整逐字稿
    if hasattr(result, 'combined_transcript') and result.combined_transcript:
        print("完整逐字稿:")
        print(result.combined_transcript)
    # 再輸出切分結果
    for phrase in result.phrases:
        speaker = phrase.speaker if phrase.speaker is not None else "Unknown"
        start_sec = phrase.offset_milliseconds / 1000
        end_sec = (phrase.offset_milliseconds + phrase.duration_milliseconds) / 1000
        line = f"[{start_sec:.2f} - {end_sec:.2f}] Speaker {speaker}: {phrase.text}"
        print(line)