# yag_email_notifier.py
# ------------------------------------------------------------
# Minimaler E-Mail-Notifier mit yagmail.
# - ENV-only (auch via .env): SMTP_USER/SMTP_PASS ODER YAG_USER/YAG_PASS, EMAIL_TO
# - Einzige öffentliche Methode:
#     send_log_tail(subject, log_file="master.log", lines=10)
#   -> verschickt genau die letzten `lines` Zeilen als Text + HTML <pre>
# ------------------------------------------------------------

import os
import re
import html
from collections import deque
from typing import Optional

import yagmail
from dotenv import load_dotenv
from config.project_config import get_data_path, get_config_path

# .env automatisch laden
load_dotenv(dotenv_path=get_config_path('.env', as_string=True))


class EmailNotifier:
    """
    Schlanker Notifier für Log-Mailings.

    Erwartete ENV-Variablen:
      - SMTP_USER / SMTP_PASS   (oder alternativ: YAG_USER / YAG_PASS)
      - EMAIL_TO                (Komma- oder Semikolon-getrennt)

    Nutzung:
        notifier = YagEmailNotifier()
        notifier.send_log_tail("Mein Betreff", log_file="master.log", lines=10)
    """

    def __init__(self) -> None:
        # Empfänger aus ENV (Komma/Semikolon-getrennt -> Liste)
        raw_to = os.getenv("EMAIL_TO", "")
        self.recipients = [a.strip() for a in re.split(r"[,\s]+", raw_to) if a.strip()]
        if not self.recipients:
            raise ValueError("EMAIL_TO muss gesetzt sein (Komma- oder Semikolon-getrennt).")

        # Zugangsdaten aus ENV
        smtp_user = os.getenv("SMTP_USER")
        smtp_pass = os.getenv("SMTP_PASS")
        yag_user  = os.getenv("YAG_USER")
        yag_pass  = os.getenv("YAG_PASS")

        if smtp_user and smtp_pass:
            self.mailer = yagmail.SMTP(user=smtp_user, password=smtp_pass)
        elif yag_user and yag_pass:
            self.mailer = yagmail.SMTP(user=yag_user, password=yag_pass)
        else:
            raise ValueError("SMTP_USER/SMTP_PASS oder YAG_USER/YAG_PASS müssen gesetzt sein.")

    # -------- interne Helfer --------
    @staticmethod
    def _read_last_lines(abs_path: str, lines: int) -> str:
        """Return exactly the last `lines` lines from file (preserve formatting)."""
        try:
            want = max(1, int(lines))
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                tail_lines = deque(f, maxlen=want)
            return "".join(tail_lines)
        except FileNotFoundError:
            return f"[Log file not found: {abs_path}]"

    def _send_email(self, subject: str, body_text: str, body_html: Optional[str] = None) -> None:
        contents = [yagmail.inline(body_html)] if body_html else [body_text]
        self.mailer.send(to=self.recipients, subject=subject, contents=contents)

    # -------- öffentliche API --------
    def send_log_tail(self, subject: str, log_file: str, lines: int = 60) -> None:
        """
        Versendet die letzten `lines` Zeilen von `log_file` (Pfadauflösung via get_data_path).
        - subject: Betreff der E-Mail (frei wählbar)
        - log_file: Logdateiname relativ zu deinem Projekt (Default: "master.log")
        - lines: Anzahl Zeilen vom Ende (Default: 60)
        """
        abs_path = get_data_path(file_name=log_file, as_string=True)
        tail = self._read_last_lines(abs_path, lines)
        self._send_email(
            subject=subject,
            body_text=tail,
            body_html=f"<pre>{html.escape(tail)}</pre>",
        )
