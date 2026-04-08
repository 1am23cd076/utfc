FROM python:3.11

WORKDIR /app

# Install dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY models.py .
COPY threat_generator.py .
COPY graders.py .
COPY environment.py .
COPY server/app.py .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
