# Use official Python image
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OVITO and display support
RUN apt-get update && apt-get install -y \
    xvfb \
    libgl1-mesa-glx \
    libglu1-mesa \
    libxrender1 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 \
    qtbase5-dev \
    libqt5gui5 \
    libqt5widgets5 \
    libqt5opengl5-dev \
    libqt5core5a \
    && rm -rf /var/lib/apt/lists/*

# Install key build tools and runtime dependencies 
RUN pip install --no-cache-dir setuptools wheel twine build numpy>=2.3.1 matplotlib>=3.10.3 ovito>=3.12.4

COPY requirements.txt /app/
# Install dependencies from requirements.txt (if any extra are needed)
RUN pip install --no-cache-dir -r requirements.txt

# Copy all of the package source code and metadata files into the image
COPY pyproject.toml /app/
COPY README.md /app/
COPY LICENSE /app/
COPY setup.py /app/
#COPY .dockerignore /app/
COPY CHANGELOG.md /app/
#COPY docs /app/docs/
#COPY docs/* /app/docs/
COPY lammpskit /app/lammpskit

# Note: Test infrastructure (tests/) is mounted as volume during CI/CD
# This ensures tests use the latest code without rebuilding the Docker image
# Volume mount in CI: -v ${PWD}/tests:/app/tests

# Build the package (creates dist/*.whl)
RUN python -m build

# Install the newly built wheel
RUN pip install --no-cache-dir dist/*.whl

# Create a non-root user and set home directory
RUN useradd --create-home lammpsuser
USER lammpsuser
WORKDIR /home/lammpsuser

# Start a bash shell when the container runs
CMD ["bash"]