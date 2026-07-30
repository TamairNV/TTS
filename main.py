import sys
import types
import warnings
from pathlib import Path

# 1. Suppress harmless deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)


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
        """Generates audio without gradient tracking to prevent RAM memory spikes."""
        with torch.no_grad():
            wav = self.model.generate(
                text,
                audio_prompt_path=ref_audio,
                cfg_weight=cfg_weight,
                exaggeration=exaggeration,
            )

        # Free cached MPS memory back to system RAM
        if self.device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return wav

    def save_audio(self, wav: torch.Tensor, output_path: str):
        """Saves generated waveform using torchaudio or numpy fallback."""
        try:
            ta.save(output_path, wav, self.model.sr)
        except Exception:
            # Fallback if torchcodec / soundfile backend is missing
            from scipy.io import wavfile

            audio_data = wav.cpu().squeeze().numpy()
            wavfile.write(output_path, self.model.sr, audio_data)


# Usage Example
if __name__ == "__main__":
    tts = VoiceGenerator(device="mps")

    script = (
        "Yeah, physics class is cool and all... but have you ever tried landing a"
        " perfect sixty-yard spiral? Whole different science, bro. Absolute"
        " poetry."
    )

    audio = tts.generate(
        text=script,
        ref_audio="jock.wav",
        cfg_weight=0.7,
        exaggeration=0.85,
    )

    tts.save_audio(audio, "test-full.wav")
    print("Done! Audio saved to test-full.wav")