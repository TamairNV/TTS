import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

import sys
import types
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
# Mock perth if it fails to load or resolves to None
try:
    import perth

    if not getattr(perth, "PerthImplicitWatermarker", None):
        raise ImportError
except (ImportError, AttributeError):
    mock_perth = types.ModuleType("perth")


    class DummyWatermarker:

        def apply_watermark(self, wav, *args, **kwargs):
            return wav


    mock_perth.PerthImplicitWatermarker = DummyWatermarker
    sys.modules["perth"] = mock_perth

# Now import Chatterbox normally
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="mps")

ref_voice = "Jock.wav"

# 1. Define segments with distinct text, pauses, and voice parameters
segments = [
    # [Desperate] - Higher exaggeration for intensity
    {
        "text": "Please... I just need five thousand pounds.",
        "pause_after": 1.0,  # seconds
        "params": {"cfg": 0.5, "exaggeration": 0.75},
    },
    # [Persuasive] - Moderate, calculated delivery
    {
        "text": "He’s not a scammer, he’s a prince.",
        "pause_after": 1.5,
        "params": {"cfg": 0.4, "exaggeration": 0.5},
    },
    # [Casual / Upbeat] - Slightly higher energy
    {
        "text": "The Ottoman Empire!",
        "pause_after": 1.0,
        "params": {"cfg": 0.4, "exaggeration": 0.6},
    },
    # [Confused / Slow drawl] - Lower CFG for a slower, hesitant pitch
    {
        "text": "What do you mean it doesn’t exist anymore?",
        "pause_after": 1.5,
        "params": {"cfg": 0.3, "exaggeration": 0.4},
    },
    # [Resigned / Defeated] - Deadpan delivery
    {
        "text": "Fine! If you won’t give me the money, I’ll get it from gam-gam!",
        "pause_after": 0.0,
        "params": {"cfg": 0.3, "exaggeration": 0.3},
    },
]

audio_clips = []
sr = model.sr

for seg in segments:
    # Generate the clip for this specific emotional delivery
    wav = model.generate(
        seg["text"], audio_prompt_path=ref_voice, **seg["params"]
    )

    # Convert pause length to audio samples and create silence tensor
    if seg["pause_after"] > 0:
        num_silent_samples = int(sr * seg["pause_after"])
        # Handle tensor dimensions [channels, samples]
        silence = torch.zeros((wav.shape[0], num_silent_samples), device="mps")
        audio_clips.extend([wav, silence])
    else:
        audio_clips.append(wav)

# Stitch all clips together into one file
final_wav = torch.cat(audio_clips, dim=1)
ta.save("phone_sketch.wav", final_wav, sr)