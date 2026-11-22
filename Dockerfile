FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python ml_training/train_model.py && \
    python manage.py migrate && \
    python manage.py collectstatic --noinput && \
    chmod -R 777 /app

EXPOSE 7860

CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:7860"]