import math
import random

def get_time_str(minutes_in):
    minutes_total = int(minutes_in)
    seconds = int((minutes_in - minutes_total) * 60)
    days = minutes_total // 1440            # 1440 Minuten pro Tag
    remaining_minutes = minutes_total % 1440

    hours = remaining_minutes // 60
    minutes = remaining_minutes % 60

    return f"Day {days} {hours:02}:{minutes:02}:{seconds:02}"

def get_duration(minutes_in):
    minutes = int(minutes_in)
    seconds = int(round((minutes_in - minutes) * 60))
    parts = []
    if minutes:
        parts.append(f"{minutes:02} minute{'s' if minutes != 1 else ''}")
    if seconds:
        parts.append(f"{seconds:02} second{'s' if seconds != 1 else ''}")
    return " ".join(parts) if parts else ""

def duration_log_normal(duration, vc=0.2):
    sigma = vc
    mu = math.log(duration)
    result = random.lognormvariate(mu, sigma)
    return round(result, 2)