import sys
import types
import warnings
from pathlib import Path

# 1. Suppress harmless deprecation and SDPA warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore", message=".*sdpa.*attention does not support output_attentions.*"
)


# 2. Patch missing or incompatible perth watermarker
def _patch_perth():
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


_patch_perth()

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS


class VoiceGenerator:

    def __init__(self, device: str = "mps"):
        self.device = device
        self.model = ChatterboxTTS.from_pretrained(device=self.device)

    def generate(
        self,
        text: str,
        ref_audio: str,
        cfg_weight: float = 0.35,
        exaggeration: float = 0.4,
    ) -> torch.Tensor:
        """Generates a single audio clip without gradient tracking."""
        with torch.no_grad():
            wav = self.model.generate(
                text,
                audio_prompt_path=ref_audio,
                cfg_weight=cfg_weight,
                exaggeration=exaggeration,
            )

        if self.device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return wav

    def generate_sketch(
        self, segments: list[dict], ref_audio: str
    ) -> torch.Tensor:
        """Generates multi-emotion dialogue segments with precise silence padding."""
        audio_clips = []
        sr = self.model.sr

        for i, seg in enumerate(segments, 1):
            print(f"Generating segment {i}/{len(segments)}...")

            # Extract segment-specific params (fall back to defaults if not set)
            params = seg.get("params", {})
            cfg_weight = params.get("cfg_weight", 0.35)
            exaggeration = params.get("exaggeration", 0.4)

            # Generate segment audio
            wav = self.generate(
                text=seg["text"],
                ref_audio=ref_audio,
                cfg_weight=cfg_weight,
                exaggeration=exaggeration,
            )

            audio_clips.append(wav)

            # Add silence tensor if a pause is specified
            pause_after = seg.get("pause_after", 0.0)
            if pause_after > 0:
                num_silent_samples = int(sr * pause_after)
                silence = torch.zeros(
                    (wav.shape[0], num_silent_samples), device=wav.device
                )
                audio_clips.append(silence)

        # Concatenate along the time dimension
        return torch.cat(audio_clips, dim=1)

    def save_audio(self, wav: torch.Tensor, output_path: str):
        """Saves audio file with a fallback for missing torchaudio backends."""
        try:
            ta.save(output_path, wav, self.model.sr)
        except Exception:
            from scipy.io import wavfile

            audio_data = wav.cpu().squeeze().numpy()
            wavfile.write(output_path, self.model.sr, audio_data)


# Usage Example
if __name__ == "__main__":
    tts = VoiceGenerator(device="mps")

    # Your phone sketch structured with distinct text, pauses, and delivery params
    phone_sketch = [
        # 1. Casual opener - low CFG for a slow, laid-back Texas drawl
        {
            "text": (
                "Say man... [snort] remember last week... when we were playin'"
                " Catan... with Bobby?"
            ),
            "pause_after": 0.8,
            "params": {"cfg_weight": 0.3, "exaggeration": 0.4},
        },
        # 2. NPC2 simple reaction - short and direct
        {
            "text": "Yeah...",
            "pause_after": 1.0,
            "params": {"cfg_weight": 0.5, "exaggeration": 0.3},
        },
        # 3. [Whisper / Leaning in] - Lower exaggeration, trailing drawl with ellipses
        {
            "text": (
                "Don't you ever... get this feelin'... man? When you watch him play"
                " like that?"
            ),
            "pause_after": 1.2,
            "params": {"cfg_weight": 0.28, "exaggeration": 0.35},
        },
        # 4. NPC2 Confused - Stuttering punctuation forces micro-hesitations
        {
            "text": "Wha... What do you mean, man?",
            "pause_after": 0.8,
            "params": {"cfg_weight": 0.45, "exaggeration": 0.5},
        },
        # 5. [Passionate / Smooth] - Bumping exaggeration slightly for rhythm & flavor words
        {
            "text": (
                "I'm talkin' about the way he moves those little wooden"
                " pieces... [sigh] and builds... that long... beautiful road of his."
                " Pure poetry, bro."
            ),
            "pause_after": 1.5,  # Let the absurd statement sink in
            "params": {"cfg_weight": 0.32, "exaggeration": 0.6},
        },
        # 6. NPC2 Unsure / Hesitant
        {
            "text": "I... guess so...?",
            "pause_after": 1.0,
            "params": {"cfg_weight": 0.35, "exaggeration": 0.3},
        },
        # 7. [Casual / Deadpan Delivery] - The punchline, delivered totally straight-faced
        {
            "text": (
                "So what you're sayin' is... you also find him irresistibly"
                " sexy... when he plays Catan?"
            ),
            "pause_after": 3.0,  # The requested 3-second pause at the end
            "params": {"cfg_weight": 0.3, "exaggeration": 0.45},
        },
    ]

    print("Starting sketch generation...")
    sketch_audio = tts.generate_sketch(segments=phone_sketch, ref_audio="math.wav")

    tts.save_audio(sketch_audio, "phone_sketch.wav")
    print("Done! Saved complete sketch to phone_sketch.wav")