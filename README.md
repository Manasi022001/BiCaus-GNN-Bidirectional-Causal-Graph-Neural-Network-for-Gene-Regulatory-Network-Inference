# BiCaus-GNN-Bidirectional-Causal-Graph-Neural-Network-for-Gene-Regulatory-Network-Inference
A chromatin-aware, multi-task graph neural network for predicting gene-gene regulatory interactions with both strength and direction.
BiCaus-GNN is a deep learning framework that integrates graph topology, chromatin-state attention, and multi-task learning to jointly predict:

Interaction Strength: The magnitude of gene-gene regulatory effects (regression)
Regulatory Direction: Activation (+1) or inhibition (-1) classification

Key Features

Dual-Pathway Architecture: Separate structure and context pathways for comprehensive feature learning
Chromatin-State Attention: Biologically-informed attention mechanism using epigenetic context
Multi-Task Learning: Joint optimization of regression and classification objectives
Superior Performance: +147.8% improvement in Pearson correlation, +62% lower MSE vs baseline GCN
Biological Interpretability: Mechanistic insights into regulatory mechanisms

Performance Highlights
MetricBaseline GCNBiCaus-GNNImprovementPearson Correlation0.3260.807+147.8%MSE---62%Direction Accuracy~0.61~0.75+23.2%R²--+747.4%
Architecture
Input Graph
    ↓
┌─────────────────────────┐
│   Dual-Pathway GNN      │
├───────────┬─────────────┤
│ Structure │  Context    │
│ Pathway   │  Pathway    │
│           │  (Chromatin │
│           │  Attention) │
└─────┬─────┴─────┬───────┘
      └─────┬─────┘
            ↓
    Combined Embedding
            ↓
    ┌───────┴────────┐
    ↓                ↓
Regression      Classification
(Strength)      (Direction)
 Quick Start
Installation
bash# Clone the repository
git clone https://github.com/yourusername/bicaus-gnn.git
cd bicaus-gnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Basic Usage
pythonfrom src.models.bicaus_gnn import BiCausGNN
from src.data.data_loader import load_gene_interaction_data
import torch

# Load data
train_loader, val_loader, test_loader = load_gene_interaction_data(
    data_path='data/gene_interactions.csv',
    batch_size=32
)

# Initialize model
model = BiCausGNN(
    input_dim=64,
    hidden_dim=128,
    output_dim=1,
    num_classes=2,
    dropout=0.3
)

# Train model
from src.train import train_model
model, history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=200,
    lr=0.001
)

# Evaluate
from src.evaluate import evaluate_model
metrics = evaluate_model(model, test_loader)
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
print(f"Test Correlation: {metrics['correlation']:.4f}")
Running the Notebook
bash# Start Jupyter
jupyter notebook

# Open notebooks/MULTI-OMICS_WITH_GNN.ipynb
📁 Project Structure

📖 Method Details
Dataset
The dataset consists of gene-gene interactions represented as a graph:

Nodes: Genes with feature vectors (expression profiles, regulatory annotations)
Edges: Regulatory interactions with two labels:

Interaction strength (continuous)
Direction class (activation/inhibition)



BiCaus-GNN Components
1. Dual-Pathway Feature Extractor

Structure Pathway: Processes graph topology via message-passing
Context Pathway: Incorporates chromatin-state attention

2. Chromatin-State Attention Module

Computes attention weights from node features
Modulates embeddings to emphasize biologically active regions
Implemented using feed-forward layers + softmax

3. Multi-Task Output Heads

Regression Head: Predicts interaction strength (MSE loss)
Classification Head: Predicts regulatory direction (Cross-entropy loss)

Training
python# Loss function
total_loss = mse_loss(strength_pred, strength_true) + 
             ce_loss(direction_pred, direction_true)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        strength_pred, direction_pred = model(batch)
        loss = compute_total_loss(strength_pred, direction_pred, batch)
        loss.backward()
        optimizer.step()
📈 Results
Regression Performance
BiCaus-GNN shows tight clustering around the diagonal in prediction scatterplots, with:

Strong correlation (r = 0.807)
Narrow residual distribution
Captures full dynamic range of interactions

Classification Performance
Confusion matrix analysis shows:

Strong diagonal dominance
Effective separation of activation/inhibition classes
Superior to baseline GCN

Training Dynamics

Consistent convergence across all loss components
Validation accuracy reaches ~0.75
R² shows continual improvement, plateauing around epoch 100

🔬 Biological Insights
Pathway-Level Interpretability

Canonical pathways (repressive): Higher attention for negative correlations
Non-canonical pathways (activating): Higher attention for positive correlations
Attention mechanism reveals mechanistically interpretable regulatory patterns

Feature Importance
Top CpG sites correspond to:

Known regulatory regions
Biologically relevant biomarkers
Tissue-specific methylation patterns

🧪 Experiments
Baseline Comparison
bash# Train baseline GCN
python src/train.py --model baseline --epochs 200

# Train BiCaus-GNN
python src/train.py --model bicaus --epochs 200

# Compare results
python src/evaluate.py --compare baseline bicaus
Ablation Studies
Test individual components:
bash# Without chromatin attention
python src/train.py --model bicaus --no-attention

# Single-task (regression only)
python src/train.py --model bicaus --regression-only

# Single-task (classification only)
python src/train.py --model bicaus --classification-only
📚 Citation
If you use this code in your research, please cite:
bibtex@article{phadke2024bicaus,
  title={Tissue-Specific DNA Methylation–Gene Expression Associations via BiCaus-GNN},
  author={Phadke, Manasi},
  year={2024},
  institution={University of California}
}
