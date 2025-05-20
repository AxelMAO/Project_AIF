#
FROM python:3.10.17-apline3.20

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Script to download the model weights
RUN python downloads_weights.py

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable for Flask to run in production mode
ENV FLASK_ENV=production

# Run mnist_api.py when the container launches
#CMD ["python", "movie_api.py", "--model_path", "weights/movie_net.pth"]
CMD ["python", "movie_api.py"]