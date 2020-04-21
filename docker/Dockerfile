FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN python -m nltk.downloader -d /usr/local/share/nltk_data stopwords
