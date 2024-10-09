# Robotics - Follow Object and Navigate to Avoid Obstacles

This example workload integrates the following existing works:

- [Depth Anything V2](https://github.com/brade3190/Depth-Anything-V2)
- [YOLOv8](https://github.com/ultralytics/ultralytics)


## Installation

### Create conda environment

```bash
conda create -yn workload python=3.10
conda activate workload
```

### Install Python packages

```bash
pip install -r requirements.txt
```

### Install Depth Anything V2

```bash
cd shared/
git clone git@github.com:T-K-233/Depth-Anything-V2.git
cd Depth-Anything-V2/
pip install -e .
```


## Running the Demo

Connect a USB webcam to the computer and run the following command:

```bash
python run_demo.py
```
