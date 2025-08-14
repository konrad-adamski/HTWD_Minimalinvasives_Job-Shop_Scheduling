# yag_email_notifier.py
import argparse
import os
import html
import time
import threading
import logging
from collections import deque
from typing import Optional
import yagmail
from project_config import get_data_path

from dotenv import load_dotenv

from src.Logger import SingletonLogger

load_dotenv()

class YagEmailNotifier:
    """
    E-Mail-Notifier mit yagmail + Log-Tail + Logger-Überwachung.

    Pflicht-ENV:
      - SMTP_USER / SMTP_PASS  (alternativ: YAG_USER / YAG_PASS)
      - EMAIL_TO  (kommagetrennt)

    Features:
      - send_email(...)
      - send_log_tail(log_file="master.log", lines=20, subject=None)
      - start_tail_thread(...), stop_tail_thread(), is_tail_thread_running
      - attach_warning_mail_handler(logger, level=logging.WARNING, ...)
        -> E-Mail sofort bei WARNING/ERROR/CRITICAL
    """

    def __init__(self):
        # Credentials aus ENV
        self.user = os.getenv("SMTP_USER") or os.getenv("YAG_USER")
        self.password = os.getenv("SMTP_PASS") or os.getenv("YAG_PASS")
        if not self.user or not self.password:
            raise ValueError("SMTP_USER/SMTP_PASS oder YAG_USER/YAG_PASS müssen gesetzt sein.")

        # Empfänger aus ENV
        self.recipients = self._recipients_from_env()
        if not self.recipients:
            raise ValueError("EMAIL_TO muss gesetzt sein (kommagetrennt).")

        self.yag = yagmail.SMTP(self.user, self.password)

        # Thread-Steuerung für Periodik
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # -------------------------
    # Basis-Funktionen
    # -------------------------
    @staticmethod
    def _recipients_from_env() -> list[str]:
        raw = os.getenv("EMAIL_TO", "")
        return [x.strip() for x in raw.split(",") if x.strip()]

    def send_email(
        self,
        subject: str,
        body_text: str = "",
        body_html: Optional[str] = None,
        attachments: Optional[list[str]] = None,
    ):
        contents: list = []
        if body_text:
            contents.append(body_text)
        if body_html:
            contents.append(yagmail.inline(body_html))
        if attachments:
            contents.extend(attachments)
        self.yag.send(to=self.recipients, subject=subject, contents=contents)

    @staticmethod
    def read_last_lines(file_path: str, n: int = 20) -> str:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                dq = deque(f, maxlen=n)
            return "".join(dq) if dq else "[Keine Logs vorhanden]"
        except FileNotFoundError:
            return "[Logdatei nicht gefunden]"

    def send_log_tail(
        self,
        log_file: str = "master.log",
        lines: int = 20,
        subject: Optional[str] = None,
    ):
        log_path = get_data_path(file_name=log_file, as_string=True)
        tail = self.read_last_lines(log_path, lines)
        self.send_email(
            subject=subject or f"Log-Update: letzte {lines} Zeilen",
            body_text=tail,
            body_html=f"<pre>{html.escape(tail)}</pre>",
        )

    # -------------------------
    # Periodischer Versand per Thread
    # -------------------------
    def start_tail_thread(
        self,
        *,
        log_file: str = "master.log",
        lines: int = 20,
        interval_hours: float = 4.0,
        subject: Optional[str] = None,
        daemon: bool = True,
        send_immediately: bool = True,
    ) -> threading.Thread:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Tail-Thread läuft bereits.")

        self._stop_event.clear()
        interval_seconds = max(60, int(interval_hours * 3600))  # min 60s

        def _loop():
            try:
                if send_immediately:
                    self.send_log_tail(log_file=log_file, lines=lines, subject=subject)
                # warte & sende in Intervallen
                while not self._stop_event.wait(interval_seconds):
                    self.send_log_tail(log_file=log_file, lines=lines, subject=subject)
            except Exception as e:
                # optional: eigenen Logger hier einhängen
                print(f"[LogTailThread] Fehler: {e}")

        self._thread = threading.Thread(target=_loop, name="LogTailThread", daemon=daemon)
        self._thread.start()
        return self._thread

    def stop_tail_thread(self, join: bool = True, timeout: Optional[float] = None):
        if not self._thread:
            return
        self._stop_event.set()
        if join:
            self._thread.join(timeout)
        self._thread = None

    @property
    def is_tail_thread_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # -------------------------
    # Logger-Überwachung: sofort bei WARNING mailen
    # -------------------------
    def attach_warning_mail_handler(
            self,
            logger: logging.Logger,
            *,
            level: int = logging.WARNING,
            subject_prefix: str = "[LOGGER]",
            debounce_seconds: float = 120.0,
            html_pre: bool = True,
            fmt: str = "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt: str = "%Y-%m-%d %H:%M:%S",
    ) -> logging.Handler:
        """
        Hängt einen Handler an 'logger', der bei Records >= 'level' sofort E-Mail sendet.
        KEIN Logfile-Tail mehr – verschickt nur den Logeintrag (plus Traceback, falls vorhanden).
        """
        handler = _MailOnLevelHandler(
            notifier=self,
            level=level,
            subject_prefix=subject_prefix,
            debounce_seconds=debounce_seconds,
            html_pre=html_pre,
            fmt=fmt,
            datefmt=datefmt,
        )
        handler.setLevel(level)
        logger.addHandler(handler)
        return handler


class _MailOnLevelHandler(logging.Handler):
    """
    Logging-Handler, der bei Records >= level sofort per Mail verschickt.
    Nutzt YagEmailNotifier.send_email(). Enthält Debounce.
    (Keine Logfile/Tail-Logik mehr.)
    """

    def __init__(
        self,
        *,
        notifier: "YagEmailNotifier",
        level: int,
        subject_prefix: str,
        debounce_seconds: float,
        html_pre: bool,
        fmt: str,
        datefmt: str,
    ):
        super().__init__(level)
        self.notifier = notifier
        self.subject_prefix = subject_prefix
        self.debounce_seconds = max(0.0, float(debounce_seconds))
        self.html_pre = bool(html_pre)
        self.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        self._last_sent_ts = 0.0

    def emit(self, record: logging.LogRecord):
        try:
            # Debounce
            now = time.time()
            if (now - self._last_sent_ts) < self.debounce_seconds:
                return
            self._last_sent_ts = now

            # Basistext (formatiert)
            formatted = self.format(record)

            # Exception-Info (falls vorhanden)
            if record.exc_info:
                import traceback
                tb = "".join(traceback.format_exception(*record.exc_info))
                formatted = f"{formatted}\n\nTraceback:\n{tb}"

            body_text = formatted
            body_html = f"<pre>{html.escape(formatted)}</pre>" if self.html_pre else None
            subject = f"{self.subject_prefix} {record.levelname} in {record.name}"

            self.notifier.send_email(subject=subject, body_text=body_text, body_html=body_html)

        except Exception:
            # Handler darf nie hart crashen – Fehler intern behandeln
            self.handleError(record)

# ------------------------------------------------------------
# Nutzung (CLI, minimal):
# ------------------------------------------------------------
# Intervall + Warning-Watcher (immer zusammen):
#   python -m src.EmailNotifier --interval-hours 1 [--no-immediate] [--debounce 300]
#
# Nur Warning-Watcher (ohne Intervall):
#   python -m src.EmailNotifier --watch-warnings [--debounce 300]
#
# Ohne Parameter (kein Intervall, kein Watcher):
#   schickt einmalig die letzten 20 Zeilen aus master.log
#
# ENV-Variablen (Pflicht):
#   SMTP_USER, SMTP_PASS, EMAIL_TO
# ------------------------------------------------------------

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Minimal-CLI: Intervall+Warnings oder nur Warnings. Intervall-Tail=20, Log=master.log."
    )
    p.add_argument("--interval-hours", type=float, default=None,
                   help="Wenn gesetzt: starte Intervall-Thread (inkl. Warning-Watcher).")
    p.add_argument("--watch-warnings", action="store_true",
                   help="Nur Logger überwachen und bei WARNING/ERROR/CRITICAL sofort Mail schicken.")
    p.add_argument("--debounce", type=float, default=120.0,
                   help="Cooldown (Sekunden) zwischen Warning-Mails. Default: 120.")
    p.add_argument("--no-immediate", action="store_true",
                   help="Intervallmodus: nicht sofort senden, erst nach erstem Intervall.")
    return p.parse_args()


# ------------------------------------------------------------
# Nutzung (CLI, minimal):
# ------------------------------------------------------------
# Intervall + Warning-Watcher (immer zusammen):
#   python -m src.EmailNotifier --interval-hours 1 [--no-immediate] [--debounce 300]
#
# Nur Warning-Watcher (ohne Intervall):
#   python -m src.EmailNotifier --watch-warnings [--debounce 300]
#
# Ohne Parameter (kein Intervall, kein Watcher):
#   schickt einmalig die letzten 20 Zeilen aus master.log
#
# ENV: SMTP_USER, SMTP_PASS, EMAIL_TO
# ------------------------------------------------------------
if __name__ == "__main__":
    import logging, time

    args = _parse_args()
    notifier = YagEmailNotifier()

    # Fixe Defaults nach Vorgabe
    LOG_FILE = "master.log"
    INTERVAL_TAIL_LINES = 20  # Intervall-Mails senden immer genau 20 Zeilen

    # Fall 1: Intervall gesetzt => Intervall + Warning-Watcher (immer zusammen)
    if args.interval_hours is not None:
        logger = SingletonLogger()  # dein globaler Logger

        # Warning-Watcher aktivieren (schickt nur die Warning-Zeile, kein Tail)
        notifier.attach_warning_mail_handler(
            logger,
            level=logging.WARNING,
            subject_prefix="[PROD]",
            debounce_seconds=args.debounce,
            html_pre=True,
        )

        # Intervall-Thread (Tail fix 20)
        t = notifier.start_tail_thread(
            log_file=LOG_FILE,
            lines=INTERVAL_TAIL_LINES,
            interval_hours=args.interval_hours,
            subject=None,  # Standardbetreff der Klasse verwenden
            send_immediately=not args.no_immediate,
            daemon=False,  # Prozess offen halten
        )
        try:
            t.join()
        except KeyboardInterrupt:
            notifier.stop_tail_thread()

    # Fall 2: Nur Warning-Watcher
    elif args.watch_warnings:
        logger = SingletonLogger()  # gleicher globaler Logger
        notifier.attach_warning_mail_handler(
            logger,
            level=logging.WARNING,
            subject_prefix="[PROD]",
            debounce_seconds=args.debounce,
            html_pre=True,
        )
        print(f"[INFO] Warning-Watcher aktiv (Cooldown={args.debounce}s)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[INFO] Warning-Watcher beendet.")

    # Fall 3: Keine Flags => einmalige Mail (Tail fix 20)
    else:
        notifier.send_log_tail(
            log_file=LOG_FILE,
            lines=INTERVAL_TAIL_LINES,
            subject=None,
        )
