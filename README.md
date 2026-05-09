# LLM-Tuner

AI-powered LLM configuration optimizer for **llama.cpp**, **Ollama**, and **vLLM**.

Scans your hardware, recommends optimal parameters, runs benchmarks, and generates ready-to-run config files. Solves the pain of figuring out what model to use, what context size, n_gpu_layers, quantization, and all other technical parameters.

## Features

- **System Scanner** — Full hardware detection: GPU VRAM/model, CPU cores/clock, RAM speed, disk, PCIe
- **Model Browser** — Live HuggingFace search, filter by use case, auto-recommend based on your hardware
- **3-Tier Config** — Optimum (speed), Balanced, Max Performance profiles for every hardware+use-case combo
- **11 Use Cases** — Code, Creative Writing, Chat, RAG, Translation, Math, Roleplay, Summarization, Agent, Vision, Fine-Tuning
- **Benchmark Engine** — Run llama.cpp or Ollama benchmarks, measure tokens/sec + accuracy (20 Q&A test)
- **Export & Launch** — Generate Ollama Modelfile, llama.cpp .bat/.sh scripts, JSON configs. One-click launch.
- **Persistent Settings** — Local config history, preferences saved automatically

## Installation

### From Source

```bash
git clone https://github.com/YOUR_USERNAME/llm-tuner.git
cd llm-tuner
pip install -r requirements.txt
python main.py
```

### Dependencies

- Python 3.10+
- PySide6, psutil, GPUtil, py-cpuinfo, requests

## Usage

1. **System Tab** — Click "Scan System" to detect your hardware
2. **Models Tab** — Search models or click "Auto-Recommend" for best picks
3. **Configure Tab** — Choose profile + use case, get recommended parameters
4. **Benchmark Tab** — Point to a .gguf file, run benchmark + accuracy test
5. **Export Tab** — Export config files, launch llama.cpp or Ollama

## Building Standalone Executable

```bash
pip install pyinstaller
build\build.bat
```

Produces `dist\LLM-Tuner.exe` — single file, no Python install needed.

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Supported Engines

| Engine | Support | Notes |
|--------|---------|-------|
| llama.cpp | Full | Benchmark, config, auto-launch |
| Ollama | Full | Benchmark, Modelfile export |
| vLLM | Config export | Roadmap: benchmark integration |

## Hardware Support

- **NVIDIA GPU** — Full VRAM detection, CUDA compute capability, optimal n_gpu_layers
- **AMD GPU** — Basic detection via wmic
- **CPU-only** — Thread count optimization, context sizing for RAM
- **Cross-platform** — Windows, Linux, macOS

## Use Cases

| Use Case | What it does |
|----------|-------------|
| Code Generation | Recommends coding models with large context |
| Creative Writing | Story/poetry models with quality quantization |
| General Chat | Lightweight, fast response models |
| RAG/Knowledge | Models optimized for document QA with large context |
| Translation | Multilingual models (NLLB, Qwen) |
| Math & Reasoning | Logic-focused models |
| Roleplay | Character AI personas |
| Summarization | Efficient text compression models |
| AI Agent | Tool-use, function calling models |
| Vision/Multimodal | Image understanding (LLaVA, Qwen-VL) |
| Fine-Tuning | Base models suitable for adaptation |

## Architecture

```
llm-tuner/
├── main.py                    # Entry point
├── pyproject.toml             # Package config
├── llmtuner/
│   ├── app.py                 # PySide6 app bootstrap
│   ├── ui/                    # 5 tab widgets
│   │   ├── main_window.py     # MainWindow with menu/navigation
│   │   ├── system_scan.py     # Hardware dashboard
│   │   ├── model_browser.py   # HuggingFace search
│   │   ├── config_panel.py    # Parameter editor (3 tiers)
│   │   ├── benchmark.py       # Benchmark runner UI
│   │   └── export_launch.py   # Config export + launch
│   ├── core/                  # Business logic
│   │   ├── system_info.py     # Hardware detection
│   │   ├── model_db.py        # HuggingFace API client
│   │   ├── recommender.py     # Config recommendation engine
│   │   ├── benchmark_runner.py # Benchmark execution
│   │   └── config_export.py   # Config file generation
│   ├── data/                  # Curated data
│   │   ├── use_cases.json     # 11 use case definitions
│   │   └── models.json        # 10 curated models
│   └── utils/                 # Utilities
│       ├── persistence.py     # Local JSON prefs/history
│       └── logger.py          # Logging
├── tests/                     # pytest tests (24 tests)
└── build/build.bat            # PyInstaller build script
```

## License

MIT
