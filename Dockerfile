FROM python:3.9

RUN apt-get update && apt-get install nano

WORKDIR /fetal-health
COPY . /fetal-health

RUN pip install --timeout 1000 -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]