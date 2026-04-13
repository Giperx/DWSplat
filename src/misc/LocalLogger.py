import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image

# LOG_PATH = Path("outputs/local")
DEFAULT_LOG_PATH = Path("outputs/local")


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


class _TimestampedTee:
    def __init__(self, stream, log_file_path: Path) -> None:
        self._stream = stream
        self._log_file_path = log_file_path
        self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._log_file_path.open("a", encoding="utf-8")
        self._buffer = ""

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._stream.flush()

        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._write_line(line, newline=True)
        return len(text)

    def _write_line(self, line: str, newline: bool) -> None:
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        clean_line = _ANSI_ESCAPE_RE.sub("", line)
        newline_char = '\n' if newline else ''
        self._file.write(f"{timestamp} {clean_line}{newline_char}")

    def flush(self) -> None:
        self._stream.flush()
        if self._buffer:
            self._write_line(self._buffer, newline=False)
            self._buffer = ""
        self._file.flush()

    def close(self) -> None:
        self.flush()
        self._file.close()


def enable_console_log(log_dir: Optional[Union[str, Path]] = None) -> Path:
    log_path = Path(log_dir) if log_dir is not None else DEFAULT_LOG_PATH
    log_path.mkdir(parents=True, exist_ok=True)
    log_file_path = log_path / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_console.log"
    sys.stdout = _TimestampedTee(sys.stdout, log_file_path)
    sys.stderr = _TimestampedTee(sys.stderr, log_file_path)
    return log_file_path

class LocalLogger(Logger):
    # def __init__(self) -> None:
    #     super().__init__()
    #     self.experiment = None
    #     os.system(f"rm -r {LOG_PATH}")
    def __init__(self, log_dir: Optional[Union[str, Path]] = None) -> None:
        """
        一个简易的本地替代 Logger。
        当未传入 log_dir 时，使用 DEFAULT_LOG_PATH。
        """
        super().__init__()
        self.experiment = None
        self.log_path = Path(log_dir) if log_dir is not None else DEFAULT_LOG_PATH

        # 清理并准备目录
        # shutil.rmtree(self.log_path, ignore_errors=True)
        self.log_path.mkdir(parents=True, exist_ok=True)
        
    @property
    def name(self):
        return "LocalLogger"

    @property
    def version(self):
        return 0

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        pass

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any],
        step: Optional[int] = None,
        caption: Optional[list[str]] = None,
        **kwargs,
    ):
        # The function signature is the same as the wandb logger's, but the step is
        # actually required.
        assert step is not None
        for index, image in enumerate(images):
            path = self.log_path / f"{key}/{caption[0]}_{index:0>2}_{step:0>6}.jpg"
            path.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(image).save(path)
