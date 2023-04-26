FROM python:3.10

WORKDIR /project-app

ADD ./setup .

RUN pip install pandas numpy tslearn fastapi python-multipart 

COPY ./app ./app

CMD [ "python", "./app/main.py"]


 