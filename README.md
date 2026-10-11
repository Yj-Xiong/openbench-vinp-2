# VINP-oSpatialNet + VEM: OpenBench SE Skill

This independent package implements VINP speech dereverberation for the
OpenBench `SE` task. It also saves the blind RIR estimate as a secondary WAV.

| Item | Value |
|---|---|
| Input | Noisy/reverberant audio path, resampled to 16 kHz mono |
| Output | Enhanced 16 kHz WAV in `audio_path`; estimated RIR in `rir_path` |
| Network | `network/osn/OSN_bimamba11.py:BiSpatialNet` |
| Inference | `method/vem_rir_omit3bin_nb.py:VEM`, 100 steps by default |
| Checkpoint | `checkpoint.pt`, parameter average of upstream epochs 24, 25 and 28 |
| Runtime | PyTorch FP32, CUDA, batch size 1 |
| MCP tool | `enhance_speech` |

The averaged checkpoint contains 324 model keys and loads with `strict=True`.
The original epoch checkpoints are in the parent `VINP-2/ckpt/` directory;
the upload package needs only `checkpoint.pt`.

## Install and Validate

Use Python 3.10+ and a CUDA build of PyTorch and `torchaudio` compatible with
the target GPU. The remaining dependencies are in `pyproject.toml`.

```bash
cd VINP-2/openbench_vinp2
bash setup.sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python validate.py
```

`validate.py` uses three VEM steps for a quick MCP smoke test. Production
inference uses 100 steps from `config.toml`; set `VINP_MAX_STEPS` only for an
explicit faster/shorter experiment. The local validation used PyTorch
2.2.0+cu121 and an RTX 3090.

## MCP Contract

The stdio MCP server accepts `initialize`, `tools/list`, `tools/call`, and
`ping`. Call `enhance_speech` with `{"audio_path": "/path/input.wav"}` and
optionally `output_path`. The result includes `audio_path`, `rir_path` and
`sample_rate=16000` in `structuredContent` and text content. If no output path
is given, files are saved under `.runtime/outputs/`.

The wrapper follows the upstream amplitude normalization and parameter
averaging path. Its output is a float WAV without playback peak normalization
so objective evaluation sees the original signal scale. Silent inputs return
silent outputs.

## Source and License

The algorithm, model code and weights originate from [VINP](https://github.com/Audio-WestlakeU/VINP).
The source code is distributed under the included [MIT license](LICENSE).
Check the original project for dataset and checkpoint redistribution terms.
