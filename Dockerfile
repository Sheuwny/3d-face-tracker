FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y git git-lfs ffmpeg libsm6 libxext6 cmake rsync libgl1 && rm -rf /var/lib/apt/lists/* && git lfs install

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
