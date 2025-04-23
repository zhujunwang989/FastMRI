# FastMRI Project

This repository contains the implementation of a deep learning model for FastMRI reconstruction.

## Project Structure
```
.
├── src/
│   ├── models/         # Model architecture definitions
│   ├── training/       # Training scripts
│   └── utils/          # Utility functions
├── checkpoints/        # Model checkpoints (not included in repo)
└── requirements.txt    # Python dependencies
```

## Setup
1. Clone the repository:
```bash
git clone https://github.com/zhujunwang989/FastMRI.git
cd FastMRI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Weights
The trained model weights (`best_model.pth`) are not included in this repository due to size constraints. To obtain the model weights:

1. Create a `checkpoints` directory in the project root:
```bash
mkdir checkpoints
```

2. Contact the repository owner to get access to the trained model file (`best_model.pth`).

3. Place the model file in the `checkpoints` directory:
```bash
mv /path/to/downloaded/best_model.pth checkpoints/
```

## Usage
[Add usage instructions here]

## License
[Add license information here] 