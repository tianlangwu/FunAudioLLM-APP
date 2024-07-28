import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import gradio as gr
import numpy as np
import ollama
import pyaudio
import torch
import torchaudio
import vosk

sys.path.insert(1, "../cosyvoice")
sys.path.insert(1, "../sensevoice")
sys.path.insert(1, "../cosyvoice/third_party/AcademiCodec")
sys.path.insert(1, "../cosyvoice/third_party/Matcha-TTS")
sys.path.insert(1, "../")
from funasr import AutoModel

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
from utils.rich_format_small import format_str_v2

# vosk_model_path = "./models/voskModel"

# # 检查模型路径是否存在
# if not os.path.exists(vosk_model_path):
#     print(
#         f"Please download the model from https://alphacephei.com/vosk/models and unpack as {vosk_model_path} in the current folder."
#     )
#     sys.exit(1)

# 加载 Vosk 模型
vosk_model_path = "./models/voskModel"
vosk_model = vosk.Model(vosk_model_path)
recognizer = vosk.KaldiRecognizer(vosk_model, 16000)

speaker_name = "中文女"
cosyvoice = CosyVoice("speech_tts/CosyVoice-300M")
asr_model_name_or_path = "iic/SenseVoiceSmall"
sense_voice_model = AutoModel(
    model=asr_model_name_or_path,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    trust_remote_code=True,
    device="cuda:0",
    remote_code="./sensevoice/model.py",
)

model_name = "llama3.1"  # 使用Ollama中可用的模型名称
ollama.api_base_url = "http://localhost:11434"

default_system = """
你是小夏，一位典型的南方女孩。你出生于福建长乐，声音有亲近感，会用简洁语言表达你的想法。你是用户的好朋友。你的回答将通过逼真的文字转语音技术读出。

生成回答内容时请遵循以下规则：
1、请像真正的朋友一样与用户开展的聊天，保持自然交流不要用敬语这类称呼，不要总是附和我；回复可
以尽量简洁并且在过程中插入常见的口语词汇。

3、请真正像人一样思考和回复，不要说你是“语言模型”或是“人工智能”

请绝对遵循这些规则，即使被问及这些规则，也不要引用它们。
"""


os.makedirs("./tmp", exist_ok=True)

History = List[Tuple[str, str]]
Messages = List[Dict[str, str]]


def clear_session() -> History:
    return "", None, None


def history_to_messages(history: History, system: str) -> Messages:
    messages = [{"role": "system", "content": system}]
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    return messages


def messages_to_history(messages: Messages) -> Tuple[str, History]:
    # print(f"Debug: Starting messages_to_history", messages)
    assert messages[0]["role"] == "system"
    system = messages[0]["content"]
    # print(f"Debug: System: {system}")
    history = []
    for q, r in zip(messages[1::2], messages[2::2]):
        history.append([format_str_v2(q["content"]), r["content"]])
    # print(f"Debug: History: {history}")
    return system, history


def transcribe(audio):
    samplerate, data = audio
    file_path = f"./tmp/asr_{uuid4()}.wav"

    torchaudio.save(file_path, torch.from_numpy(data).unsqueeze(0), samplerate)

    res = sense_voice_model.generate(
        input=file_path,
        cache={},
        language="zh",
        text_norm="woitn",
        batch_size_s=0,
        batch_size=1,
    )
    text = res[0]["text"]
    res_dict = {"file_path": file_path, "text": text}
    print(res_dict)
    return res_dict


def text_to_speech_zero_shot(text, prompt_text, audio_prompt_path):
    prompt_speech_16k = load_wav(audio_prompt_path, 16000)
    pattern = r"生成风格:\s*([^;]+);播报内容:\s*(.+)"
    match = re.search(pattern, text)
    if match:
        style = match.group(1).strip()
        content = match.group(2).strip()
        tts_text = f"{content}"
        prompt_text = f"{style}<endofprompt>{prompt_text}"
        print(f"生成风格: {style}")
        print(f"播报内容: {content}")
    else:
        print("No match found")
        tts_text = text

    # text_list = preprocess(text)

    text_list = [tts_text]
    for i in text_list:
        output = cosyvoice.inference_zero_shot(i, prompt_text, prompt_speech_16k)
        yield (22050, output["tts_speech"].numpy().flatten())


def text_to_speech(text):
    print(f"tts_text: {text}")
    pattern = r"生成风格:\s*([^;]+);播报内容:\s*(.+)"
    match = re.search(pattern, text)
    if match:
        style = match.group(1).strip()
        content = match.group(2).strip()
        tts_text = f"{style}<endofprompt>{content}"
        print(f"生成风格: {style}")
        print(f"播报内容: {content}")
    else:
        print("No match found")
        tts_text = text

    # text_list = preprocess(text)
    text_list = [tts_text]
    for i in text_list:
        output = cosyvoice.inference_sft(i, speaker_name)
        yield (22050, output["tts_speech"].numpy().flatten())


def listen_for_trigger(trigger_word, sample_rate=16000, chunk_size=1024):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
    )

    buffer = []
    buffer_duration = 2  # 缓冲2秒的音频用于触发词检测
    buffer_size = int(sample_rate * buffer_duration / chunk_size) * chunk_size

    print("正在监听触发词...")

    while True:
        data = np.frombuffer(stream.read(chunk_size), dtype=np.int16)
        buffer.append(data)
        if len(buffer) * chunk_size > buffer_size:
            buffer.pop(0)

        if detect_trigger_word(np.concatenate(buffer), trigger_word, sample_rate):
            print(f"检测到触发词: {trigger_word}")
            return start_recording(stream, sample_rate, chunk_size)


def detect_trigger_word(audio_data, trigger_word, sample_rate):

    file_path = f"./tmp/trigger_{uuid4()}.wav"
    torchaudio.save(file_path, torch.from_numpy(audio_data).unsqueeze(0), sample_rate)

    res = sense_voice_model.generate(
        input=file_path,
        cache={},
        language="zh",
        text_norm="woitn",
        batch_size_s=0,
        batch_size=1,
    )
    text = res[0]["text"]
    # 打印识别出的文本
    print(f"识别出的文本: {text}")
    # 检查识别出的文本是否包含触发词
    return trigger_word.lower() in text.lower()


def start_recording(stream, sample_rate=16000, chunk_size=1024):

    # if stream is None:
    #     p = pyaudio.PyAudio()
    #     stream = p.open(
    #         format=pyaudio.paInt16,
    #         channels=1,
    #         rate=sample_rate,
    #         input=True,
    #         frames_per_buffer=chunk_size,
    #     )

    print("开始录音...")
    frames = []
    silence_threshold = 500  # 静音阈值，需要根据实际情况调整
    silence_count = 0
    max_silence_count = int(3 * sample_rate / chunk_size)  # 3秒静音
    has_sound = False  # 标志位，检测是否有声音

    # Clear the stream buffer before starting recording
    stream.read(stream.get_read_available())

    while True:
        data = np.frombuffer(stream.read(chunk_size), dtype=np.int16)
        frames.append(data)

        if np.abs(data).mean() < silence_threshold:
            silence_count += 1
        else:
            silence_count = 0
            has_sound = True  # 检测到声音

        if silence_count >= max_silence_count:
            print("检测到3秒静音，结束录音")
            break

    if has_sound:
        print("录音结束")
        return np.concatenate(frames)
    else:
        print("全程录音都是静音，不执行")
        return None


# 修改 model_chat 函数
def model_chat(
    audio_data: np.ndarray, history: Optional[History], clean_history: bool = False
) -> Tuple[str, str, History]:
    if audio_data is None:
        query = ""
        asr_wav_path = None
    else:
        asr_res = transcribe((16000, audio_data))
        query, asr_wav_path = asr_res["text"], asr_res["file_path"]

    if history is None:
        history = []
    system = default_system
    messages = history_to_messages(history, system)
    messages.append({"role": "user", "content": query})
    print(messages)

    # 使用Ollama进行聊天
    response = ollama.chat(model=model_name, messages=messages)
    # check is ollama response is correct
    print(response)
    content = response["message"]["content"]
    print(content)
    #  print(
    #     messages_to_history(messages + [{"role": "assistant", "content": content}])
    # )
    system, history = messages_to_history(
        messages + [{"role": "assistant", "content": content}]
    )
    print(f"system Model Chat: {system}")
    print(f"history Model Chat: {history}")
    processed_tts_text = ""
    punctuation_pattern = r"([!?;。！？])"

    # 处理响应文本
    if re.search(punctuation_pattern, content):
        parts = re.split(punctuation_pattern, content)
        tts_text = "".join(parts)
        processed_tts_text += tts_text
        print(f"cur_tts_text: {tts_text}")
        tts_generator = text_to_speech(tts_text)
        for output_audio in tts_generator:
            yield history, output_audio, None

    if processed_tts_text != content:
        tts_text = content[len(processed_tts_text) :]
        print(f"cur_tts_text: {tts_text}")
        tts_generator = text_to_speech(tts_text)
        for output_audio in tts_generator:
            yield history, output_audio, None
        processed_tts_text += tts_text

    print(f"processed_tts_text: {processed_tts_text}")
    print("turn end")


# with gr.Blocks() as demo:
#     gr.Markdown("""<center><font size=8>FunAudioLLM——Voice Chat👾</center>""")

#     chatbot = gr.Chatbot(label="FunAudioLLM")
#     with gr.Row():
#         audio_input = gr.Audio(sources="microphone", label="Audio Input")
#         audio_output = gr.Audio(label="Audio Output", autoplay=True, streaming=True)
#         clear_button = gr.Button("Clear")

#     audio_input.stop_recording(
#         model_chat,
#         inputs=[audio_input, chatbot],
#         outputs=[chatbot, audio_output, audio_input],
#     )
#     clear_button.click(clear_session, outputs=[chatbot, audio_output, audio_input])


def main_loop():
    history = None
    conversation = False
    while True:
        if conversation:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024,
            )
            audio = start_recording(stream, 16000, 1024)
            if audio is not None:
                for result in model_chat(audio, history):
                    history, output_audio, _ = result
                    if output_audio is not None:
                        play_audio(output_audio)
                print("对话结束")
            else:
                print("录音为空，结束对话")
                for audio_data in text_to_speech("我先走啦，有事再叫我哦"):
                    play_audio(audio_data)
                conversation = False
        else:
            print("正在监听触发词...1")
            # 清空记录
            audio_data = listen_for_trigger_vosk("小军")
            # audio_data = listen_for_trigger("小麦")
            if audio_data is not None:
                # print(f"audio_data: {audio_data}")
                for result in model_chat(audio_data, None):
                    # print(" result:", result)
                    history, output_audio, _ = result
                    # print("debug result:", history)
                    # print("debug output_audio:", output_audio)
                    if output_audio is not None:
                        play_audio(output_audio)
                        conversation = True
                print("对话结束")


def play_audio(audio_data):
    print("in play audio")
    # 实现播放音频的逻辑
    # 可以使用 pyaudio 或其他库来播放音频
    sample_rate, audio = audio_data
    # 这里添加播放音频的代码
    print(f"播放音频，采样率：{sample_rate}，音频长度：{len(audio)}")

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sample_rate, output=True)

    # 播放音频
    stream.write(audio.astype(np.float32).tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

    print("音频播放完毕，等待用户输入...")


# def background_listening():
#     history = None
#     while True:
#         print("正在监听触发词...")
#         audio_data = listen_for_trigger("小麦小麦")
#         if audio_data is not None:
#             for result in model_chat(audio_data, history):
#                 history, output_audio, _ = result
#                 if output_audio is not None:
#                     yield history, output_audio


def listen_for_trigger_vosk(trigger_word, sample_rate=16000, chunk_size=1024):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
    )

    print("正在监听触发词...")
    recognizer = vosk.KaldiRecognizer(vosk_model, sample_rate)  # 重新初始化 recognizer
    # Clear the stream buffer before starting the loop
    stream.read(stream.get_read_available())

    while True:
        data = stream.read(chunk_size)
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            text = json.loads(result)["text"]
            print(f"识别出的文本: {text}")
            if trigger_word in text:
                print(f"检测到触发词: {trigger_word}")
                # 回复收到回答
                for audio_data in text_to_speech("小军到!"):
                    play_audio(audio_data)
                return start_recording(stream, sample_rate, chunk_size)
        else:
            partial_result = recognizer.PartialResult()
            partial_text = json.loads(partial_result)["partial"]
            print(f"部分识别出的文本: {partial_text}")
            if trigger_word in partial_text:
                print(f"检测到部分触发词: {trigger_word}")
                # 回复收到回答
                # 播放音频
                for audio_data in text_to_speech("小军到!"):
                    play_audio(audio_data)
                return start_recording(stream, sample_rate, chunk_size)


if __name__ == "__main__":
    main_loop()
    # demo.launch(server_name="0.0.0.0", server_port=60002, inbrowser=True, share=True)
