FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn pydantic openai

# Copy the project files
COPY . .

# Expose port 7860 (default for HF Spaces)
EXPOSE 7860

# Simple FastAPI server to handle OpenEnv requests
# This satisfies the requirement for responding to /reset and /step
# However, usually the evaluation platform handles this by importing the env.py
# But providing a server is safer for deployment.

CMD ["uvicorn", "decision_env.server:app", "--host", "0.0.0.0", "--port", "7860"]
