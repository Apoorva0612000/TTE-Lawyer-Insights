import logging
import logging.config
import os
from datetime import datetime

import pytz

# Ensure logs directory exists
os.makedirs("app/logs", exist_ok=True)


class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=pytz.UTC)
        dt_ist = dt.astimezone(pytz.timezone("Asia/Kolkata"))
        return dt_ist.strftime("%Y-%m-%d %H:%M:%S")


logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "()": ISTFormatter,
            "format": "{asctime} {levelname} {name} [{filename}:{lineno}] {message}",
            "style": "{",
        }
    },
    "handlers": {
        "logs": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": "app/logs/logs.log",
            "formatter": "verbose",
        },
        "error_5xx_logs": {
            "level": "ERROR",
            "class": "logging.FileHandler",
            "filename": "app/logs/error_5xx_logs.log",
            "formatter": "verbose",
        },
        "error_logs": {
            "level": "WARNING",
            "class": "logging.FileHandler",
            "filename": "app/logs/error_logs.log",
            "formatter": "verbose",
        },
    },
    "loggers": {
        "lawyer-suggestion": {
            "handlers": ["logs", "error_5xx_logs", "error_logs"],
            "level": "DEBUG",
            "propagate": False,
        }
    },
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger("lawyer-suggestion")
