import base64    # <--- 補上這行
import requests  # 如果你下面有用到 requests
import os        # 處理路徑或環境變數會用到
import wave
import contextlib
import tempfile
from dotenv import load_dotenv
load_dotenv()

def transcribe_audio(api_key, audio_file_path):
    # 自動偵測音檔規格
    with wave.open(audio_file_path, "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()

    with open(audio_file_path, "rb") as f:
        audio_content = base64.b64encode(f.read()).decode("utf-8")

    url = f"https://speech.googleapis.com/v1/speech:recognize?key={api_key}"

    payload = {
        "config": {
            "enableWordTimeOffsets": True,
            "sampleRateHertz": sample_rate,
            "audioChannelCount": channels,
            "languageCode": "vi-VN",
            "enableAutomaticPunctuation": True,
        },
        "audio": {"content": audio_content},
    }

    import time
    t0 = time.perf_counter()
    response = requests.post(url, json=payload)
    elapsed = time.perf_counter() - t0

    if response.status_code == 200:
        return response.json(), elapsed
    else:
        return f"Error: {response.text}", elapsed


def parse_and_print_result(result):
    if isinstance(result, str):
        print(result)
        return
    if "results" not in result:
        print(result)
        return
    print("\n==============================\n Google STT v1 結果整理\n==============================")
    for idx, res in enumerate(result["results"]):
        alt = res["alternatives"][0]
        transcript = alt.get("transcript", "")
        confidence = alt.get("confidence", None)
        print(f"段落{idx+1}：")
        print(f"  逐字稿: {transcript}")
        if confidence is not None:
            print(f"  信心度: {confidence:.3f}")
        # 時間戳
        if "words" in alt:
            for w in alt["words"]:
                word = w["word"]
                start = w.get("startTime", "?")
                end = w.get("endTime", "?")
                wconf = w.get("confidence", None)
                print(f"    [ {start} ~ {end} ] {word} " + (f"(信心度: {wconf:.3f})" if wconf is not None else ""))
        # 備選
        if len(res["alternatives"]) > 1:
            print("  其他備選：")
            for i, a in enumerate(res["alternatives"][1:], 2):
                print(f"    {i}. {a.get('transcript', '')}")
    print("==============================\n")

# %%
# 使用範例
MY_KEY = os.getenv("API_KEY")  # 從環境變數讀取 API Key
orig_path = r"C:\Users\iec150094\STT_Test\LAPTOP HP PAVILION 2025 - ƯU ĐÃI CỰC KHỦNG CHO HỌ SINH & SINH VIÊN [M3MSUhTsb-o].wav"
with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
    result, elapsed = transcribe_audio(MY_KEY, orig_path)
print(f"\nHTTP 200  |  推論時間: {elapsed:.2f}s")
parse_and_print_result(result)