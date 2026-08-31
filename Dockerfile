# Single image, two roles. The default CMD serves the API; docker-compose.yml
# runs the same image again with its command overridden to start Streamlit
# instead, because ui/streamlit_app.py imports app.pipeline and app.schemas
# directly rather than only talking over HTTP -- it needs the same
# dependencies the API does, not a lighter subset.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgl1 + libglib2.0-0: opencv's runtime shared libraries (app/preprocess.py).
# tesseract-ocr + its ara/eng data: the traditional:tesseract engine: the
# binary pytesseract shells out to, not a Python package.
# poppler-utils: PyMuPDF's own PDF rendering does not depend on it, but it is
# the fallback a deployment reaches for when a malformed PDF beats PyMuPDF, and
# it is a few MB next to the image's other costs.
# curl: the container healthchecks in docker-compose.yml.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        tesseract-ocr \
        tesseract-ocr-ara \
        tesseract-ocr-eng \
        poppler-utils \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-ocr.txt ./
RUN pip install -r requirements.txt
# Opt-in, roughly +1 GB: paddlepaddle + paddleocr for OCR_BACKEND=paddle.
# Uncomment to enable traditional:paddle; the other three engines run without it.
# RUN pip install -r requirements-ocr.txt

COPY . .

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
