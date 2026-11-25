FROM python:3.12-slim@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY pyproject.toml ./
RUN python -c "import tomllib; p=tomllib.load(open('pyproject.toml','rb')); open('/tmp/requirements.txt','w').write(chr(10).join(p['project']['dependencies'])+chr(10))" \
    && pip install --no-cache-dir setuptools wheel -r /tmp/requirements.txt
COPY src ./src
COPY scripts ./scripts
COPY README.md ./
RUN pip install --no-cache-dir --no-deps --no-build-isolation .
EXPOSE 8000
CMD ["python", "-m", "adult_monitoring.exporter", "--config", "configs/example.yaml"]
