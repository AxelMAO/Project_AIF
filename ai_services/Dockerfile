#
FROM python:3.12

# RUN apt-get update && apt-get install -y \
#     gcc \
#     g++ \
#     libffi-dev \
#     libpq-dev \
#     python3-dev \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0


# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Script to download the model weights
RUN python download_data.py

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable for Flask to run in production mode
ENV FLASK_ENV=production

# Run mnist_api.py when the container launches
#CMD ["python", "movie_api.py", "--model_path", "weights/movie_net.pth"]
CMD ["python", "movie_api.py"]