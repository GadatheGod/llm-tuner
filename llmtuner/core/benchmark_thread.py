import subprocess
from PySide6.QtCore import QThread, Signal


class BenchmarkThread(QThread):
    """Run llama.cpp or Ollama benchmark in a background thread.
    
    Signals:
        progress(str): Progress/update message
        result(object): BenchmarkResult when done
        error(str): Error message if benchmark fails
    """

    progress = Signal(str)
    result = Signal(object)
    error = Signal(str)

    def __init__(self, runner, method, *args, **kwargs):
        super().__init__()
        self.runner = runner
        self.method = method
        self.args = args
        self.kwargs = kwargs
        self._cancelled = False

    def run(self):
        try:
            if self.method == "llama":
                self._run_llama()
            elif self.method == "ollama":
                self._run_ollama()
            elif self.method == "accuracy":
                self._run_accuracy()
        except Exception as e:
            self.error.emit(str(e))

    def _run_llama(self):
        model_path, config = self.args
        self.progress.emit("Starting benchmark...")
        proc = subprocess.run(
            [
                self.runner._llama_cpp_path or self.runner.auto_detect_llama_cpp(),
                "-m", model_path,
                "--no-warmup",
                "-n", str(config.get("n_predict", 256)),
                "-c", str(config.get("n_ctx", 4096)),
                "-b", str(config.get("n_batch", 2048)),
                "-t", str(config.get("n_threads", 8)),
            ]
            + (["-ngl", str(config.get("n_gpu_layers", 0))] if config.get("n_gpu_layers", 0) > 0 else [])
            + (["-fa"] if config.get("flash_attention") else [])
            + ["-p", "Write a short Python function to compute fibonacci numbers."],
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = proc.stdout + proc.stderr
        result = self.runner._parse_result(output, model_path, "llama.cpp", config)
        self.result.emit(result)

    def _run_ollama(self):
        model_name = self.args[0]
        self.progress.emit("Running Ollama benchmark...")
        proc = subprocess.run(
            ["ollama", "run", model_name, "Write a short Python function to compute fibonacci numbers."],
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = proc.stdout + proc.stderr
        result = self.runner._parse_result(output, model_name, "ollama", {})
        self.result.emit(result)

    def _run_accuracy(self):
        result = self.args[0]
        config = self.kwargs.get("config", {})
        self.progress.emit("Running accuracy test (20 questions)...")
        correct = 0
        for i, qa in enumerate(self.runner.ACCURACY_PROMPTS):
            if self._cancelled:
                break
            prompt = f"Question: {qa['q']}\nAnswer briefly: "
            try:
                cmd = [
                    self.runner._llama_cpp_path or self.runner.auto_detect_llama_cpp(),
                    "-m", result.model,
                    "-n", "32",
                    "-p", prompt,
                    "-t", str(config.get("n_threads", 8)),
                ]
                if config.get("n_gpu_layers", 0) > 0:
                    cmd += ["-ngl", str(config["n_gpu_layers"])]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                answer = proc.stdout + proc.stderr
                expected = qa["a"].lower()
                if any(word in answer.lower() for word in expected.split()):
                    correct += 1
                self.progress.emit(f"Accuracy: {i + 1}/{len(self.runner.ACCURACY_PROMPTS)} checked")
            except Exception:
                continue
        result.accuracy_total = len(self.runner.ACCURACY_PROMPTS)
        result.accuracy_correct = correct
        result.accuracy_score = (correct / len(self.runner.ACCURACY_PROMPTS) * 100) if self.runner.ACCURACY_PROMPTS else 0
        self.result.emit(result)

    def cancel(self):
        self._cancelled = True
