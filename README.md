# Qwen3-TTS

**中⽂・EN** — 純 Rust 語⾳合成，無需 Python 或 ONNX。

Pure Rust text-to-speech inference for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Alibaba). Built on [candle](https://github.com/huggingface/candle) — no Python or ONNX runtime required.

---

## 📦 快速使用 — Quick Usage

### 方式一：下載 ZIP 壓縮包（推薦）

從 [Releases](https://github.com/stevenke1981/qwen3-tts-rscandle/releases) 下載對應版本：

| ZIP | 適用環境 |
|-----|---------|
| `Qwen3-TTS-v0.4.0-win64.zip` (3.4 GB) | 任何 Windows x64 電腦 |
| `Qwen3-TTS-v0.4.0-cuda13.2-win64.zip` (3.9 GB) | 有 NVIDIA GPU 的電腦（CUDA 13.2） |

**使用步驟：**
1. 解壓縮到任意資料夾
2. 雙擊 `gui.exe` 開啟圖形介面
3. 選擇模型變體（預設為 1.7B CustomVoice）
4. 輸入文字，點擊 Generate
5. 聆聽或儲存生成的語音

> 可以直接從 GUI 切換五種模型變體。如果需要其他模型，執行 `scripts/download-models.ps1`。

**Download** the ZIP from Releases, extract, and double-click `gui.exe`.

### 方式二：從原始碼執行

```bash
# 下載模型（約 14 GB）
.\scripts\download-models.ps1

# CLI 模式（CPU）
cargo run --release --features cli --bin generate_audio -- `
  --model-dir models/1.7B-CustomVoice --tokenizer-dir models/tokenizer `
  --text "你好，世界" --speaker ryan --language english

# GUI 模式（CPU）
cargo run --release --features cli,gui --bin gui
```

---

## 🖥️ 圖形介面（GUI）使用說明

GUI 支援直覺的滑鼠操作，無需指令列。使用方式：

![GUI Demo](assets/images/gui-demo.png) *(screenshot placeholder)*

| 元件 | 說明 |
|------|------|
| **模型選擇** | 下拉選單切換 5 種模型變體，切換時自動更新路徑 |
| **輸出目錄** | 生成的 WAV 檔案儲存位置 |
| **語言切換** | GUI 介面中⽂/英⽂切換（右上角） |
| **文字輸入** | 要合成的文字內容 |
| **說話人** | 預設 9 種聲⾳可選（CustomVoice 模型） |
| **語⾔** | 目標語⾔（英文、中⽂、⽇⽂等） |
| **⾼級參數** | Temperature / Top-K / Top-P / Repetition Penalty |
| **⾳頻⻑度** | 最⼤⽣成秒數（約 12.5 幀/秒） |
| **隨機種⼦** | 固定種⼦可復現相同結果 |
| **進度條** | 顯⽰⽣成進度（幀數 / 總幀數） |

**GUI controls:**

| Control | Description |
|---------|-------------|
| **Model Selector** | Dropdown to switch between 5 model variants, auto-updates paths |
| **Output Directory** | Where generated WAV files are saved |
| **Language Switch** | Toggle GUI between Chinese / English (top-right) |
| **Text Input** | Text to synthesize |
| **Speaker** | Choose from 9 preset voices (CustomVoice models) |
| **Language** | Target language (English, Chinese, Japanese, etc.) |
| **Advanced** | Temperature / Top-K / Top-P / Repetition Penalty |
| **Duration** | Max seconds of audio (~12.5 frames/sec) |
| **Seed** | Fixed seed for reproducible generation |
| **Progress Bar** | Shows generation progress (frames / total) |

---

## 📋 指令列（CLI）使用說明

### 模型路徑結構

```
models/
├── 0.6B-Base/             語⾳克隆（⼩）
├── 0.6B-CustomVoice/      預設聲⾳（⼩）
├── 1.7B-Base/             語⾳克隆（⼤）
├── 1.7B-CustomVoice/      預設聲⾳（⼤）← 預設
├── 1.7B-VoiceDesign/      描述聲⾳
├── speech_tokenizer/      共享語⾳編碼器
└── tokenizer/             共享⽂字分詞器
```

### CLI 選項

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model-dir` | `models/1.7B-CustomVoice` | 模型路徑 |
| `--tokenizer-dir` | `models/tokenizer` | 分詞器路徑 |
| `--text` | `"Hello"` | 要合成的文字 |
| `--speaker` | `ryan` | 預設聲⾳（CustomVoice 專⽤） |
| `--language` | `english` | 目標語⾔ |
| `--instruct` | | 聲⾳描述（VoiceDesign 專⽤） |
| `--ref-audio` | | 參考⾳頻 WAV（Base 專⽤） |
| `--ref-text` | | 參考⾳頻逐字稿（需搭配 `--ref-audio`） |
| `--x-vector-only` | | 僅聲⾳嵌入，無 ICL（搭配 `--ref-audio`） |
| `--output` | | 輸出 WAV 路徑 |
| `--device` | `auto` | 裝置：`auto`, `cpu`, `cuda`, `cuda:N`, `metal` |
| `--duration` | | 最大秒數 |
| `--frames` | `2048` | 最大幀數（約 164 秒） |
| `--temperature` | `0.7` | 採樣溫度 |
| `--top-k` | `50` | Top-K 採樣 |
| `--top-p` | `0.9` | Top-P 採樣 |
| `--repetition-penalty` | `1.05` | 重複懲罰 |
| `--seed` | `42` | 隨機種⼦ |

### CLI 使用範例

```bash
# CustomVoice：預設聲⾳
cargo run --release --features cli --bin generate_audio -- `
  --model-dir models/1.7B-CustomVoice --tokenizer-dir models/tokenizer `
  --text "Hello world" --speaker ryan --language english

# Base：語⾳克隆（ICL，最佳品質）
cargo run --release --features cli --bin generate_audio -- `
  --model-dir models/1.7B-Base --tokenizer-dir models/tokenizer `
  --text "Hello world" --ref-audio reference.wav --ref-text "transcript"

# VoiceDesign：描述聲⾳
cargo run --release --features cli --bin generate_audio -- `
  --model-dir models/1.7B-VoiceDesign --tokenizer-dir models/tokenizer `
  --text "Hello world" --instruct "A calm female voice with warm tone"
```

---

## 📥 模型下載 — Model Download

```powershell
# 下載全部 5 種模型變體 + 共享組件（約 14 GB）
.\scripts\download-models.ps1

# 或從 HuggingFace ⼿動下載
```

| 模型 | HF Repo | ⼤⼩ | 適⽤場景 |
|------|---------|------|---------|
| 0.6B Base | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 1.7 GB | 語⾳克隆（⼩，快速） |
| 0.6B CustomVoice | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 1.7 GB | 預設聲⾳（⼩，快速） |
| 1.7B Base | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 3.6 GB | 語⾳克隆（⾼品質） |
| 1.7B CustomVoice | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 3.6 GB | 預設聲⾳（⾼品質）← 推薦 |
| 1.7B VoiceDesign | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 3.6 GB | 聲⾳描述 |
| Speech Tokenizer | `Qwen/Qwen3-TTS-Tokenizer-12Hz` | 682 MB | 共享語⾳編碼器 |
| Text Tokenizer | `Qwen/Qwen2-0.5B` | 7 MB | 共享⽂字分詞器 |

### Which model should I use?

- **Clone a specific voice?** Use a **Base** model with `--ref-audio`.
- **Quick preset voice?** Use a **CustomVoice** model with `--speaker`.
- **Describe a voice in text?** Use **1.7B VoiceDesign** with `--instruct`.
- **Not sure?** Start with **1.7B CustomVoice** (the default in GUI).

---

## 🛠️ 從原始碼編譯 — Build from Source

### 環境需求

- **Rust** 1.85+（最新穩定版）
- **CUDA 13.2**（僅 CUDA 版本需要，從 VsDevCmd 或 `build-release-cuda.bat` 編譯）
- **Visual Studio 2026 Build Tools**（僅 Windows CUDA 需要）

### 編譯

```bash
# CPU 版本
cargo build --release --features cli,gui

# CUDA 版本（需從 VS 開發者命令提⽰字元執⾏）
.\build-release-cuda.bat

# 打包 ZIP
powershell -File package.ps1        # CPU ZIP
powershell -File package-cuda.ps1   # CUDA ZIP
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `cpu` (default) | CPU inference |
| `cuda` | NVIDIA GPU acceleration |
| `flash-attn` | Flash Attention 2 (bf16, requires CUDA toolkit) |
| `metal` | Apple Silicon GPU acceleration |
| `mkl` | Intel MKL for faster CPU inference |
| `accelerate` | Apple Accelerate framework |
| `hub` | HuggingFace Hub model downloads |
| `cli` | Command-line tools |
| `gui` | Desktop GUI (egui) |

---

## 🏗️ 架構 — Architecture

```
Text → TalkerModel → Semantic Token → CodePredictor → [16 codes] → Decoder → Audio
            ^                                ^
       (autoregressive)              (per frame, 15 acoustic codes)
```

Three-stage pipeline:

1. **TalkerModel**: 28-layer transformer, generates semantic tokens from text (MRoPE)
2. **CodePredictor**: 5-layer decoder, generates 15 acoustic tokens per semantic token
3. **Decoder12Hz**: ConvNeXt blocks + transposed conv upsampling → 24kHz audio

---

## ⚡ 效能 — Performance

Non-streaming RTF (real-time factor), < 1.0 = faster than real-time:

| Model | CUDA BF16 | CPU F32 |
|-------|-----------|---------|
| 0.6B Base | **0.48** | — |
| 1.7B Base | **0.65** | — |
| 1.7B CustomVoice | **0.64** | 5.39 |
| 1.7B VoiceDesign | **0.64** | — |

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full results.

---

## 📜 更新紀錄 — Changelog

### 0.4.0
- Pre-allocated KV cache with InplaceOp2 (zero-copy CUDA writes)
- GPU-side repetition penalty mask (incremental slice_assign)
- Deferred acoustic codes transfer (bulk GPU→CPU at end)
- Fused residual + RMSNorm CUDA kernel
- GPU→CPU syncs reduced from 3/frame to 1/frame
- **GUI with progress bar** and model variant selector
- **Multi-model management** with shared tokenizer/speech_tokenizer
- **Portable ZIP releases** (CPU + CUDA)

### 0.3.0
- GPU-side sampling: batched argmax, on-device top-k/top-p/repetition penalty
- Eliminated 15 of 16 GPU→CPU syncs per frame in code predictor
- Cached token suppression mask in streaming sessions
- Tokenizer fallback from vocab.json + merges.txt
- Profiling infrastructure: Chrome tracing, flamegraph, Nsight Systems

### 0.2.0
- ICL voice cloning now works correctly with proper reference audio
- Fixed WAV output format (WAVEX/float32 → standard WAV/PCM16)
- Improved tokenizer path resolution with `--tokenizer-dir` override

---

## 🙏 致謝 — Acknowledgements

- [Qwen Team (Alibaba)](https://github.com/QwenLM) — Qwen3-TTS model & weights
- [candle](https://github.com/huggingface/candle) — Rust ML framework by Hugging Face
- [Claude Code](https://claude.ai/code) — wrote the code

## License

MIT License.
