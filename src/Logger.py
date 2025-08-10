import logging
from logging.handlers import RotatingFileHandler
from colorama import Fore, Style, init as colorama_init
from project_config import get_data_path


class _ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        msg = super().format(record)
        return f"{color}{msg}{Style.RESET_ALL}" if color else msg


class SingletonLogger:

    _instance = None

    def __new__(cls, name=__name__, log_file="master.log"):
        """
        Singleton logger with rotating file and console handlers.

        Ensures a single :class:`logging.Logger` instance is used across the application,
        with a fixed format and a :class:`logging.handlers.RotatingFileHandler`.

        :param name: Logger name.
        :param log_file: Log file name (stored in the solver_logs directory).

        **Example**::

            logger = SingletonLogger()
            logger.info("Message")
        """
        if cls._instance is None:
            cls._instance = super(SingletonLogger, cls).__new__(cls)
            cls._instance._initialized = False
            cls._instance.name = name
            cls._instance.log_file = get_data_path(file_name = log_file, as_string=True)
            cls._instance.logger = logging.getLogger(name)
            cls._instance.logger.setLevel(logging.INFO)
            cls._instance._setup_handlers()
            cls._instance._initialized = True
        return cls._instance.logger

    def _setup_handlers(self):
        colorama_init(autoreset=True)
        file_handler = RotatingFileHandler(self.log_file, maxBytes=2000000000, backupCount=5)
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        color_fmt = _ColorFormatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(color_fmt)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        for handler in self.logger.handlers:
            handler.flush()
            handler.close()
        self.logger.propagate = False

    """
    def ___setup_handlers(self):
        
        # doppelte Handler vermeiden (minimal)
        if not any(isinstance(h, RotatingFileHandler) for h in self.logger.handlers):
            self.logger.addHandler(file_handler)
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            self.logger.addHandler(console_handler)

        self.logger.propagate = False  # vermeidet doppelte Ausgaben über Root
    """
