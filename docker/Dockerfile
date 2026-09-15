# Illari live paper-trading lab — a long-running container that checks
# the strategy once per hour (see docker/run_forever.py). Runs on the
# host's own network, unlike a cloud sandbox (see docs/PLAN.md's note on
# why that path doesn't work here — egress to api.binance.com is blocked
# in Anthropic's cloud environments).
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "docker/run_forever.py"]
