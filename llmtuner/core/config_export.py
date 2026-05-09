import os
import platform
from typing import Dict, Optional
from llmtuner.utils.logger import logger


def export_ollama_modelfile(
    output_path: str,
    model_config: Dict,
    model_path: str = ""
) -> str:
    lines = []
    if model_path:
        lines.append(f"FROM {model_path}")
    else:
        lines.append("FROM <model.gguf>")

    if model_config.get("n_ctx"):
        lines.append(f'PARAMETER num_ctx {model_config["n_ctx"]}')
    if model_config.get("n_gpu_layers"):
        lines.append(f'PARAMETER num_gpu {model_config["n_gpu_layers"]}')
    if model_config.get("n_batch"):
        lines.append(f'PARAMETER num_batch {model_config["n_batch"]}')
    if model_config.get("n_threads"):
        lines.append(f'PARAMETER num_thread {model_config["n_threads"]}')

    lines.append("")
    lines.append('SYSTEM You are a helpful assistant.')
    lines.append("")

    content = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path


def export_llama_cpp_config(
    output_path: str,
    model_config: Dict,
    model_path: str = ""
) -> str:
    lines = []
    lines.append(f"# LLM-Tuner generated config")
    lines.append(f"# Model: {model_path or '<model.gguf>'}")
    lines.append(f"# Profile: {model_config.get('profile', 'balanced')}")
    lines.append(f"# Use case: {model_config.get('use_case', 'chat')}")
    lines.append("")

    if platform.system() == "Windows":
        exe = "llama-cli.exe"
    else:
        exe = "llama-cli"

    cmd = f"{exe} -m \"{model_path or '<model.gguf>'}\""
    cmd += f" -c {model_config.get('n_ctx', 4096)}"
    cmd += f" -b {model_config.get('n_batch', 2048)}"
    cmd += f" -t {model_config.get('n_threads', 8)}"
    cmd += f" -n {model_config.get('n_predict', 256)}"

    if model_config.get("n_gpu_layers", 0) > 0:
        cmd += f" -ngl {model_config['n_gpu_layers']}"
    if model_config.get("flash_attention"):
        cmd += " -fa"
    if model_config.get("mmap"):
        cmd += " --mmap"
    if not model_config.get("mmap"):
        cmd += " --no-mmap"

    lines.append(f"{cmd}")
    lines.append("")
    lines.append("# Parameters summary:")
    for k, v in sorted(model_config.items()):
        lines.append(f"#   {k}: {v}")

    content = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path


def export_json_config(
    output_path: str,
    model_config: Dict,
    model_path: str = ""
) -> str:
    import json
    config = model_config.copy()
    config["model_path"] = model_path
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)
    return output_path


def launch_config(
    model_config: Dict,
    model_path: str,
    engine: str = "llama.cpp"
) -> Optional[str]:
    if engine == "llama.cpp":
        return _launch_llama_cpp(model_config, model_path)
    elif engine == "ollama":
        return _launch_ollama(model_config, model_path)
    return None


def _launch_llama_cpp(config: Dict, model_path: str) -> Optional[str]:
    import shutil
    binary = None
    for name in ["llama-cli", "llama-cli.exe"]:
        found = shutil.which(name)
        if found:
            binary = found
            break

    if not binary:
        return "llama-cli not found in PATH"

    cmd = [binary, "-m", model_path]
    cmd += ["-c", str(config.get("n_ctx", 4096))]
    cmd += ["-b", str(config.get("n_batch", 2048))]
    cmd += ["-t", str(config.get("n_threads", 8))]
    cmd += ["-n", str(config.get("n_predict", 256))]
    if config.get("n_gpu_layers", 0) > 0:
        cmd += ["-ngl", str(config["n_gpu_layers"])]
    if config.get("flash_attention"):
        cmd += ["-fa"]

    logger.info(f"Launching: {' '.join(cmd)}")
    os.system(" ".join(cmd))
    return None


def _launch_ollama(config: Dict, model_path: str) -> Optional[str]:
    logger.info(f"Launching Ollama with model: {model_path}")
    os.system(f"ollama run {model_path}")
    return None
