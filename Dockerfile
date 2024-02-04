
FROM python:3.9

WORKDIR /app

COPY . .

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


RUN pip install -r requirements.txt

EXPOSE 8000

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]

