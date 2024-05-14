FROM python:3.9

WORKDIR /home/app

COPY requirements.txt /dependencies/requirements.txt
RUN pip install -r /dependencies/requirements.txt

COPY . /home/app

CMD streamlit run --server.port $PORT app.py

# FROM continuumio/miniconda3

# WORKDIR /home/app

# RUN apt-get update
# RUN apt-get install nano unzip
# RUN apt install curl -y

# RUN curl -fsSL https://get.deta.dev/cli.sh | sh
# RUN pip install altair pandas numpy streamlit pydeck
# COPY . /home/app

# CMD streamlit run --server.port $PORT app.py

## link: https://vroum-vroum-app-123638f97c58.herokuapp.com/