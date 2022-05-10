FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["gunicorn","-b", "0.0.0.0:8000", "app:app"]