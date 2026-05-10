import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict
from llmtuner.utils.logger import logger


@dataclass
class BenchmarkResult:
    model: str = ""
    engine: str = ""
    config: Optional[Dict] = None
    tokens_per_second: float = 0.0
    prompt_tokens_per_second: float = 0.0
    total_tokens: int = 0
    prompt_tokens: int = 0
    prediction_tokens: int = 0
    load_time_ms: float = 0.0
    accuracy_score: float = 0.0
    accuracy_total: int = 0
    accuracy_correct: int = 0
    raw_output: str = ""
    error: str = ""

    def __post_init__(self):
        if self.config is None:
            self.config = {}


ACCURACY_PROMPTS = [
    {"q": "What is the capital of France?", "a": "Paris"},
    {"q": "What is 2 + 2?", "a": "4"},
    {"q": "What planet is known as the Red Planet?", "a": "Mars"},
    {"q": "What is the largest ocean on Earth?", "a": "Pacific"},
    {"q": "Who wrote Romeo and Juliet?", "a": "Shakespeare"},
    {"q": "What is the chemical symbol for gold?", "a": "Au"},
    {"q": "How many continents are there?", "a": "7"},
    {"q": "What is the speed of light approximately in km/s?", "a": "300000"},
    {"q": "What year did World War 2 end?", "a": "1945"},
    {"q": "What is the largest mammal?", "a": "blue whale"},
    {"q": "What element has atomic number 1?", "a": "hydrogen"},
    {"q": "What is the currency of Japan?", "a": "yen"},
    {"q": "What is the boiling point of water in Celsius?", "a": "100"},
    {"q": "Which gas do plants absorb from atmosphere?", "a": "carbon dioxide"},
    {"q": "What is the hardest natural substance?", "a": "diamond"},
    {"q": "How many legs does a spider have?", "a": "8"},
    {"q": "What is the smallest prime number?", "a": "2"},
    {"q": "What language is spoken in Brazil?", "a": "portuguese"},
    {"q": "What is the square root of 144?", "a": "12"},
    {"q": "Which organ produces insulin?", "a": "pancreas"},
]


class BenchmarkRunner:
    def __init__(self, on_progress: Optional[Callable[[str], None]] = None):
        self.on_progress = on_progress or (lambda x: None)
        self._cancelled = False
        self._llama_cpp_path = None
        self._ollama_running = False

    def set_llama_cpp_path(self, path: str):
        self._llama_cpp_path = path

    def auto_detect_llama_cpp(self) -> Optional[str]:
        common_paths = [
            os.path.expanduser("~/.llm-tuner/bin/llama-cli"),
            os.path.expanduser("~/.llm-tuner/bin/llama-bench"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "llama.cpp/llama-cli.exe"),
        ]
        for p in common_paths:
            if os.path.isfile(p):
                return p

        import shutil
        found = shutil.which("llama-cli")
        if found:
            return found
        return None

    def run_llama_cpp_benchmark(
        self,
        model_path: str,
        config: Dict,
        test_prompt: Optional[str] = None
    ) -> BenchmarkResult:
        result = BenchmarkResult(model=model_path, engine="llama.cpp", config=config.copy())

        binary = self._llama_cpp_path or self.auto_detect_llama_cpp()
        if not binary:
            result.error = "llama-cli not found. Set path in settings or download llama.cpp."
            return result

        if not os.path.isfile(model_path):
            result.error = f"Model file not found: {model_path}"
            return result

        cmd = [binary, "-m", model_path, "--no-warmup"]
        cmd += ["-n", str(config.get("n_predict", 256))]
        cmd += ["-c", str(config.get("n_ctx", 4096))]
        cmd += ["-b", str(config.get("n_batch", 2048))]
        cmd += ["-t", str(config.get("n_threads", 8))]
        if config.get("n_gpu_layers", 0) > 0:
            cmd += ["-ngl", str(config["n_gpu_layers"])]
        if config.get("flash_attention"):
            cmd += ["-fa"]

        prompt = test_prompt or "Write a short Python function to compute fibonacci numbers."

        try:
            self.on_progress("Starting benchmark...")
            t0 = time.time()
            proc = subprocess.run(
                cmd + ["-p", prompt],
                capture_output=True, text=True, timeout=300
            )
            load_time = (time.time() - t0) * 1000

            output = proc.stdout + proc.stderr
            result.raw_output = output
            result.load_time_ms = round(load_time, 1)

            tps = self._parse_tps(output, "token/s")
            ptps = self._parse_prompt_tps(output)
            result.tokens_per_second = tps
            result.prompt_tokens_per_second = ptps

            tokens_match = re.search(r"([\d,]+) tokens \(([\d.]+) tokens/second", output)
            if tokens_match:
                result.total_tokens = int(tokens_match.group(1).replace(",", ""))
                if not result.tokens_per_second:
                    result.tokens_per_second = float(tokens_match.group(2))

            p_match = re.search(r"p prompt evals?\s*([\d.]+) tokens", output)
            if p_match:
                result.prompt_tokens = int(float(p_match.group(1)))

            return result
        except subprocess.TimeoutExpired:
            result.error = "Benchmark timed out after 5 minutes"
            return result
        except Exception as e:
            result.error = str(e)
            return result

    def run_accuracy_test(self, result: BenchmarkResult) -> BenchmarkResult:
        self.on_progress("Running accuracy test (20 questions)...")

        binary = self._llama_cpp_path or self.auto_detect_llama_cpp()
        if not binary or not result.model:
            result.error = "Cannot run accuracy: no model or binary"
            return result

        model_path = result.model
        if not os.path.isfile(model_path):
            result.error = f"Model file not found for accuracy test: {model_path}"
            return result

        config = result.config or {}
        correct = 0

        for i, qa in enumerate(ACCURACY_PROMPTS):
            if self._cancelled:
                break

            prompt = f"Question: {qa['q']}\nAnswer briefly: "
            try:
                cmd = [binary, "-m", model_path, "-n", "32", "-p", prompt,
                       "-t", str(config.get("n_threads", 8))]
                if config.get("n_gpu_layers", 0) > 0:
                    cmd += ["-ngl", str(config["n_gpu_layers"])]

                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                answer = proc.stdout + proc.stderr

                expected = qa["a"].lower()
                if any(word in answer.lower() for word in expected.split()):
                    correct += 1

                self.on_progress(f"Accuracy: {i+1}/{len(ACCURACY_PROMPTS)} checked")

            except Exception:
                continue

        result.accuracy_total = len(ACCURACY_PROMPTS)
        result.accuracy_correct = correct
        result.accuracy_score = (correct / len(ACCURACY_PROMPTS) * 100) if len(ACCURACY_PROMPTS) > 0 else 0
        return result

    def run_ollama_benchmark(
        self,
        model_name: str,
        test_prompt: Optional[str] = None
    ) -> BenchmarkResult:
        result = BenchmarkResult(model=model_name, engine="ollama")

        prompt = test_prompt or "Write a short Python function to compute fibonacci numbers."

        try:
            self.on_progress("Running Ollama benchmark...")
            cmd = ["ollama", "run", model_name, prompt]

            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            output = proc.stdout + proc.stderr
            result.raw_output = output

            tps = self._parse_tps(output, "token/s")
            result.tokens_per_second = tps or self._estimate_tps(output)

            return result
        except FileNotFoundError:
            result.error = "Ollama not found. Install Ollama first."
            return result
        except subprocess.TimeoutExpired:
            result.error = "Benchmark timed out"
            return result
        except Exception as e:
            result.error = str(e)
            return result

    def cancel(self):
        self._cancelled = True

    def _parse_tps(self, text: str, pattern: str = "token/s") -> float:
        match = re.search(r"([\d.]+)\s*" + re.escape(pattern), text)
        if match:
            return float(match.group(1))
        match = re.search(r"([\d.]+)\s*tokens?/sec", text)
        if match:
            return float(match.group(1))
        return 0.0

    def _parse_prompt_tps(self, text: str) -> float:
        match = re.search(r"([\d.]+) -> ([\d.]+) tokens/second", text)
        if match:
            return float(match.group(2))
        match = re.search(r"prompt eval rate: ([\d.]+)", text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r"([\d.]+)\s*tokens/second.*prompt", text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r"prompt.*?([\d.]+)\s*tokens?/sec", text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 0.0

    def _estimate_tps(self, text: str) -> float:
        match = re.search(r"([\d.]+)ms per token", text)
        if match:
            ms_per_token = float(match.group(1))
            return 1000.0 / ms_per_token if ms_per_token > 0 else 0.0
        return 0.0

    def _parse_result(self, output: str, model: str, engine: str,
                      config: Optional[Dict] = None) -> BenchmarkResult:
        """Parse raw subprocess output into a BenchmarkResult.
        
        Used by both the synchronous runner and the background thread.
        """
        result = BenchmarkResult(model=model, engine=engine, config=config or {})
        result.raw_output = output

        tps = self._parse_tps(output, "token/s")
        ptps = self._parse_prompt_tps(output)
        result.tokens_per_second = tps
        result.prompt_tokens_per_second = ptps

        tokens_match = re.search(r"([\d,]+) tokens \(([\d.]+) tokens/second", output)
        if tokens_match:
            result.total_tokens = int(tokens_match.group(1).replace(",", ""))
            if not result.tokens_per_second:
                result.tokens_per_second = float(tokens_match.group(2))

        p_match = re.search(r"p prompt evals?\s*([\d.]+) tokens", output)
        if p_match:
            result.prompt_tokens = int(float(p_match.group(1)))

        if not result.tokens_per_second and engine == "ollama":
            result.tokens_per_second = self._estimate_tps(output)

        return result
