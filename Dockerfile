FROM python:3.10

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r Requirements.txt

EXPOSE $PORT

CMD gunicorn --workers=2 --bind 0.0.0.0:$PORT 'viz:app'