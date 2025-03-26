FROM python:3.12-slim

WORKDIR /usr/src/app

COPY ./utils/ .
COPY . .

RUN pip install -r requirements.txt

EXPOSE 7860

CMD ["python", "app.py"]