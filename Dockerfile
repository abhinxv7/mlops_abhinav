# Start with a computer that already has Python installed
FROM python:3.9

# Copy your tools list and install them
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your code and your saved brain into the container
COPY src/ /src/

# Tell the container how to turn on the Waiter
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "80"]