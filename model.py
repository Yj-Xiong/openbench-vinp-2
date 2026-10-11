"""OpenBench wrapper for VINP-oSpatialNet and its VEM inference."""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoint.pt"


@dataclass(frozen=True)
class AudioResult:
    audio_path: str
    rir_path: str
    sample_rate: int

    @property
    def text(self):
        return self.audio_path


class VINPModel:
    def __init__(self, model_path=None, device="auto"):
        import toml
        import torch

        from acoustics.feature import transforms
        from method.vem_rir_omit3bin_nb import VEM
        from network.osn.OSN_bimamba11 import BiSpatialNet

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device(device)

        config = toml.load(str(ROOT / "config.toml"))
        self.sample_rate = int(config["acoustic"]["sr"])
        self.tf = transforms(**config["acoustic"])
        self.network = BiSpatialNet(**config["model"]).to(self.device).eval()
        checkpoint = Path(model_path or DEFAULT_CHECKPOINT).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("Model checkpoint not found: {}".format(checkpoint))
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)["model"]
        self.network.load_state_dict(state, strict=True)

        vem_config = dict(config["vem"])
        if os.environ.get("VINP_MAX_STEPS"):
            vem_config["max_steps"] = int(os.environ["VINP_MAX_STEPS"])
        self.vem = VEM(**vem_config)

    def run_tool(self, audio_path, output_path=None):
        import librosa
        import numpy as np
        import soundfile as sf
        import torch

        source = Path(audio_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError("Input audio not found: {}".format(source))
        samples, rate = sf.read(str(source), dtype="float32", always_2d=True)
        if samples.shape[0] == 0:
            raise ValueError("Input audio is empty")
        mono = samples.mean(axis=1)
        if rate != self.sample_rate:
            mono = librosa.resample(mono, orig_sr=rate, target_sr=self.sample_rate)
        if not np.isfinite(mono).all():
            raise ValueError("Input audio contains non-finite samples")

        if output_path is None:
            output_dir = ROOT / ".runtime" / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            descriptor, destination = tempfile.mkstemp(
                prefix=source.stem + "_", suffix=".wav", dir=str(output_dir)
            )
            os.close(descriptor)
        else:
            destination = str(Path(output_path).expanduser().resolve())
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
        rir_destination = str(Path(destination).with_name(Path(destination).stem + "_rir.wav"))

        if np.max(np.abs(mono)) <= 1e-12:
            enhanced = np.zeros(len(mono), dtype=np.float32)
            rir = np.zeros(self.sample_rate * 2, dtype=np.float32)
        else:
            signal = torch.from_numpy(mono.copy())
            scale = signal.abs().max()
            with torch.no_grad():
                estimate, impulse, _, _ = self.vem.process(
                    signal / scale, self.network, self.tf, str(self.device)
                )
            enhanced = (estimate * scale).cpu().numpy()
            rir = impulse.cpu().numpy()
            peak = np.max(np.abs(rir)) if len(rir) else 0
            if peak > 0:
                rir = rir / peak
            if not np.isfinite(enhanced).all() or not np.isfinite(rir).all():
                raise RuntimeError("VINP produced non-finite audio")

        sf.write(destination, enhanced, self.sample_rate, subtype="FLOAT")
        sf.write(rir_destination, rir, self.sample_rate, subtype="FLOAT")
        return AudioResult(destination, rir_destination, self.sample_rate)


Model = VINPModel
