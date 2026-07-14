FROM python:3.11-slim

RUN pip install --no-cache-dir \
    numpy==2.0.2 \
    pandas==2.2.2 \
    pyarrow==18.1.0 \
    tqdm==4.67.3 \
    scipy==1.16.3

WORKDIR /work

COPY shash_moment_matching.py .
COPY settings.toml .

CMD ["python", "shash_moment_matching.py"]