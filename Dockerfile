FROM python:3.10.5

COPY Requirements.txt .

RUN pip install --no-cache-dir -r Requirements.txt

COPY . .

EXPOSE 8050

CMD ["python","viz.py"]
