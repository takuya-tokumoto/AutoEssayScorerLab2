import datetime
import logging
from datetime import datetime, timedelta, timezone
from logging import INFO, FileHandler, Logger, getLogger


# ロギングフォーマッタの拡張クラス
class JSTFormatter(logging.Formatter):

    JST = timezone(timedelta(hours=+9), "JST")

    def converter(self, timestamp):
        dt = datetime.fromtimestamp(timestamp, tz=self.JST)
        return dt.timetuple()

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=self.JST)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            try:
                s = dt.isoformat(timespec="milliseconds")
            except TypeError:
                s = dt.isoformat()
        return s


def set_logger(module_name: str) -> Logger:
    """ログの設定

    Args:
        module_name (str): ログ出力名

    Returns:
        Logger: ログ設定
    """
    # loggerの取得
    logger = getLogger(module_name)
    logger.setLevel(INFO)

    # 既存のハンドラをクリア
    if logger.hasHandlers():
        logger.handlers.clear()

    # 新しいハンドラを設定
    handler = FileHandler("training.log")
    handler.setLevel(INFO)
    handler.setFormatter(JSTFormatter("%(asctime)s:%(levelname)s:%(message)s"))
    logger.addHandler(handler)

    return logger
