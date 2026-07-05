FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
# If requirements.txt doesn't exist, we fall back to manual pip install for the required packages
RUN if [ -f "requirements.txt" ]; then pip install --no-cache-dir -r requirements.txt; else pip install --no-cache-dir requests pydantic pydantic-settings streamlit pandas; fi

# Copy application source
COPY . .

# Run the agent in status mode by default to verify it's working
ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]
