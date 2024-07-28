import os
import re
import sys
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import gradio as gr
import ollama
import torch
import torchaudio

sys.path.insert(1, "../cosyvoice")
sys.path.insert(1, "../sensevoice")
sys.path.insert(1, "../cosyvoice/third_party/AcademiCodec")
sys.path.insert(1, "../cosyvoice/third_party/Matcha-TTS")
sys.path.insert(1, "../")
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
from utils.rich_format_small import format_str_v2

# from funasr import AutoModel

speaker_name = "中文女"
cosyvoice = CosyVoice("speech_tts/CosyVoice-300M-Instruct")
asr_model_name_or_path = "iic/SenseVoiceSmall"
# sense_voice_model = AutoModel(model=asr_model_name_or_path,
#                   vad_model="fsmn-vad",
#                   vad_kwargs={"max_single_segment_time": 30000},
#                   trust_remote_code=True, device="cuda:0", remote_code="./sensevoice/model.py")


output = cosyvoice.inference_instruct(
    "在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。",
    "中文女",
    "A female speaker with high pitch, normal speaking rate, and happy emotion..",
)
torchaudio.save("instruct.wav", output["tts_speech"], 22050)
