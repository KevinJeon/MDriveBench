FROM kevinjeon119/mdrive:v0.4

WORKDIR /app

COPY . /app

ENV PYTHONPATH=/app:${PYTHONPATH}

CMD ["bash", "-lc", "source /opt/conda/etc/profile.d/conda.sh && conda activate colmdriver && exec bash"]
