import logging
import sys
from logging.handlers import RotatingFileHandler
from colorama import Fore, Style, init as colorama_init
from config.project_config import get_data_path


class _ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        msg = super().format(record)
        color = self.COLORS.get(record.levelno, "")
        return f"{color}{msg}{Style.RESET_ALL}" if color else msg


class Logger(logging.Logger):
    def __init__(self, name=__name__, log_file="main.log"):
        super().__init__(name)
        colorama_init(autoreset=True)

        self.setLevel(logging.INFO)
        self.propagate = False
        self.log_file_path = None

        if not self.handlers:
            self.log_file_path = get_data_path(file_name=log_file, as_string=True)

            file_handler = RotatingFileHandler(self.log_file_path, maxBytes=100_000_000, backupCount=5, encoding="utf-8")
            file_handler.setLevel(logging.INFO)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            formatter = logging.Formatter(
                "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            color_fmt = _ColorFormatter(
                "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )

            file_handler.setFormatter(formatter)
            console_handler.setFormatter(color_fmt)

            self.addHandler(file_handler)
            self.addHandler(console_handler)

    def callback_info(self, msg: str):
        """
        Write INFO log only to file handlers, without printing to console.
        """

        record = self.makeRecord(
            self.name, logging.INFO, fn="callback", lno=0,
            msg=msg, args=(), exc_info=None
        )
        for h in self.handlers:
            if isinstance(h, logging.FileHandler):
                if h.filter(record):
                    h.handle(record)  # <-- instead of emit()

    def get_log_file_path(self):
        return self.log_file_path