# start by pulling the python image
FROM python:3.9.7-slim-bullseye
#RUN pip install --upgrade pip
#RUN apk --no-cache add --virtual .builddeps gcc gfortran musl-dev && pip install numpy==1.14.0     && apk del .builddeps     && rm -rf /root/.cache

# switch working directory
WORKDIR /

# copy the requirements file into the image
COPY requirements.txt /requirements.txt

# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt

# copy every content from the local file to the image
#COPY ./app /
#COPY ./templates /templates
#COPY ./resource /resource

# configure the container to run in an executed manner
ENTRYPOINT [ "flask" ]

CMD ["--app", "app", "run", "--host=0.0.0.0"]
