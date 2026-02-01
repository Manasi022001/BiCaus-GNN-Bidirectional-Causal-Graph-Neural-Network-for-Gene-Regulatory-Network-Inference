# Installation Guide

## Requirements
- Python 3.8+
- CUDA 11.7+ (for GPU support)
- 16GB+ RAM

## Step 1: Clone Repository
\`\`\`bash
git clone https://github.com/yourusername/BiCaus-GNN.git
cd BiCaus-GNN
\`\`\`

## Step 2: Create Environment
\`\`\`bash
conda create -n bicaus python=3.9
conda activate bicaus
\`\`\`

## Step 3: Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Step 4: Download Data
[Instructions for accessing GTEx data]

## Step 5: Verify Installation
\`\`\`bash
python -c "import torch; import torch_geometric; print('Success!')"
\`\`\`
