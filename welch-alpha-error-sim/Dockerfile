FROM python:3.11-slim

RUN pip install --no-cache-dir \
    numpy==2.0.2 \
    scipy==1.16.3 \
    statsmodels==0.14.6 \
    pandas==2.2.2 \
    pyarrow==18.1.0 \
    joblib==1.5.3

WORKDIR /work

COPY alpha_error_simulation.py .
COPY settings.toml .

CMD ["python", "alpha_error_simulation.py"]