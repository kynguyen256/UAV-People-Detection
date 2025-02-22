
# UAV - Environment Setup

## Installation

To set up the project locally, follow these steps:

### System Preparation

1. **Update and install necessary packages:**
   ```bash
   sudo apt update
   sudo apt install python3-pip
   python3 -m pip install --upgrade pip
   sudo apt-get install protobuf-compiler libprotobuf-dev
   ```

2. **Install CPU monitoring tools:**
   ```bash
   sudo apt install snapd
   sudo snap install bpytop
   bpytop
   ```

3. **Install GPU monitoring tools:**
   ```bash
   sudo apt install pipx
   sudo apt install python3.8-venv
   pipx run nvitop
   ```

### Clone Repo and Set Up Environment

4. **Clone the repository and navigate into it:**
   ```bash
   git clone https://${GIT_USERNAME}:${GIT_PAT}@github.com/nexterarobotics/safety-tracking-inference.git
   cd safety-tracking-inference
   ```

5. **Create a virtual environment and activate it:**
   ```bash
   sudo apt install python3.9-venv
   python3 -m venv training
   source training/bin/activate
   python3 -m pip install --upgrade pip
   pip install --upgrade pip
   ```

6. **Install PyTorch and MMCV according to your CUDA version:**
   
   For **CUDA 11.3**:
   ```bash
   pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
   pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
   ```

   For **CUDA 12.1**:
   ```bash
   pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
   pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.10/index.html
   ```

7. **Install other project requirements:**
   ```bash
   pip install -r requirements.txt
   ```

