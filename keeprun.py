from uuid import uuid4

import numpy as np
import pyaudio
import torch
import torchaudio

from myrun import sense_voice_model, text_to_speech_zero_shot, transcribe


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
    # 使用 transcribe 函数进行语音识别
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

    # 检查识别出的文本是否包含触发词
    return trigger_word.lower() in text.lower()


def start_recording(stream, sample_rate, chunk_size):
    print("开始录音...")
    frames = []
    silence_threshold = 500  # 静音阈值，需要根据实际情况调整
    silence_count = 0
    max_silence_count = int(2 * sample_rate / chunk_size)  # 2秒静音

    while True:
        data = np.frombuffer(stream.read(chunk_size), dtype=np.int16)
        frames.append(data)

        if np.abs(data).mean() < silence_threshold:
            silence_count += 1
        else:
            silence_count = 0

        if silence_count >= max_silence_count:
            print("检测到2秒静音，结束录音")
            break

    return np.concatenate(frames)


# 修改 model_chat 函数
def model_chat(history: Optional[History]) -> Tuple[str, str, History]:
    audio_data = listen_for_trigger("小麦小麦")
    if audio_data is not None:
        asr_res = transcribe((16000, audio_data))  # 假设采样率为16000
        query, asr_wav_path = asr_res["text"], asr_res["file_path"]
    else:
        query = ""
        asr_wav_path = None

    if history is None:
        history = []
    system = default_system
    messages = history_to_messages(history, system)
    messages.append({"role": "user", "content": query})
    print(messages)

    # 使用Ollama进行聊天
    response = ollama.chat(model=model_name, messages=messages)
    content = response["message"]["content"]

    system, history = messages_to_history(
        messages + [{"role": "assistant", "content": content}]
    )

    processed_tts_text = ""
    punctuation_pattern = r"([!?;。！？])"

    # 处理响应文本
    if re.search(punctuation_pattern, content):
        parts = re.split(punctuation_pattern, content)
        tts_text = "".join(parts)
        processed_tts_text += tts_text
        print(f"cur_tts_text: {tts_text}")
        tts_generator = text_to_speech_zero_shot(
            tts_text, "默认提示文本", "path/to/audio_prompt.wav"
        )
        for output_audio in tts_generator:
            yield history, output_audio, None

    if processed_tts_text != content:
        tts_text = content[len(processed_tts_text) :]
        print(f"cur_tts_text: {tts_text}")
        tts_generator = text_to_speech_zero_shot(
            tts_text, "默认提示文本", "path/to/audio_prompt.wav"
        )
        for output_audio in tts_generator:
            yield history, output_audio, None
        processed_tts_text += tts_text

    print(f"processed_tts_text: {processed_tts_text}")
    print("turn end")
