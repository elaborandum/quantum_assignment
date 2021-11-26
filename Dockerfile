# syntax=docker/dockerfile:1
FROM jupyter/minimal-notebook:python-3.9.7

LABEL author="Stefan Rombouts"

COPY SRC .
RUN pip install -r requirements.txt
