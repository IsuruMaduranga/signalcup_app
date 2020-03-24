 
FROM python:3.6.9

WORKDIR /app

COPY . /app

RUN apt-get update

RUN apt-get -y install python3-pip

RUN python3 -m pip install --upgrade pip

RUN apt-get update

RUN pip3 install -r requirements.txt

RUN python3 -m pip install Pillow

RUN pip3 install Flask

EXPOSE 5000

ENTRYPOINT  ["python"]

CMD ["callbacks.py"]