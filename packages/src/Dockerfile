FROM python:3.9-alpine
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
# VOLUME /src:/app/src
# RUN adduser -D myuser
# USER myuser
ENTRYPOINT ["python"]
CMD ["-m pytest"]