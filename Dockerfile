FROM python:3.8-slim

WORKDIR /backend

COPY ./requirements.txt /backend/requirements.txt


RUN pip install fastapi
RUN pip install Jinja2 --upgrade
RUN pip install --upgrade -r /backend/requirements.txt

COPY ./src  /backend/src
COPY ./server.py  /backend/server.py

EXPOSE 1000
CMD ["python", "server.py"]
