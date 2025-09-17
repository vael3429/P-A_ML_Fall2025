# Use an official Ubuntu base image compatible with ROCm
FROM ubuntu:20.04

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ca-certificates \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install ROCm libraries (for AMD GPU support)
# Adjust ROCm version if needed
RUN wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \
    && echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/5.7/ ubuntu main' | tee /etc/apt/sources.list.d/rocm.list \
    && apt-get update && apt-get install -y rocm-dev rocm-libs \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install specific older PyTorch + torchvision with ROCm support
# Replace <TORCH_VERSION> and <TORCHVISION_VERSION> with desired versions
RUN pip install torch==2.0.0 torchvision==0.15.1 -f https://download.pytorch.org/whl/rocm5.7/torch_stable.html

# Install other Python packages
RUN pip install numpy pandas matplotlib scikit-learn jupyterlab d2l

# Copy your code
COPY . /app

# Expose Jupyter
EXPOSE 8888

# Default command
CMD ["bash"]

