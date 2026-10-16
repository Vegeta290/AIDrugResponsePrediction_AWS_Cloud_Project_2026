"""Structured logging (shipped to CloudWatch Logs by the awslogs docker driver)
and optional custom CloudWatch metrics."""
import json
import logging
import sys
import threading

from .config import settings


class JsonFormatter(logging.Formatter):
    def format(self, record):
        out = {"level": record.levelname, "logger": record.name, "msg": record.getMessage(),
               "time": self.formatTime(record)}
        out.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out)


def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


class Metrics:
    """Publishes metrics to CloudWatch in a background thread so requests never block on it."""

    def __init__(self):
        self.client = None
        if settings.cloudwatch_metrics:
            import boto3

            self.client = boto3.client("cloudwatch", region_name=settings.aws_region)

    def put(self, name, value, unit="Count", **dims):
        if self.client is None:
            return
        data = [{"MetricName": name, "Value": float(value), "Unit": unit,
                 "Dimensions": [{"Name": k, "Value": str(v)} for k, v in dims.items()]}]

        def send():
            try:
                self.client.put_metric_data(Namespace=settings.metrics_namespace, MetricData=data)
            except Exception:
                logging.getLogger("metrics").exception("put_metric_data failed")

        threading.Thread(target=send, daemon=True).start()


metrics = Metrics()
