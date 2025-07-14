# Use official Python image
FROM python:3.12-slim

WORKDIR /app

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
COPY .dockerignore /app/
COPY CHANGELOG.md /app/
COPY docs /app/docs/
COPY docs/* /app/docs/
COPY lammpskit /app/lammpskit


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