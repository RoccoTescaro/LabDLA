# 🚀 Deep Learning Applications Laboratories 🔬

## ✨ Labs Overview

*   **🧪 Lab 1: Deep Learning Foundations**
    *   Build & train MLPs and CNNs (MNIST, CIFAR-10).
    *   Explore residual connections, knowledge distillation.
*   **🎮 Lab 2: Reinforcement Learning Adventures**
    *   Implement REINFORCE & DQN algorithms.
    *   Interact with `gymnasium` environments.
*   **🤖 Lab 3: Transformers with Hugging Face**
    *   Sentiment analysis, feature extraction, fine-tuning (DistilBERT).
    *   Master PEFT techniques like LoRA.

## 🛠️ Quick Start

### 1. Clone 

```bash
git clone <your-repository-url>
cd <repository-directory-name>
```

### 2. Create Your Lab Environment (Recommended)

**Using `conda`:**

```bash
conda create -n dla_labs python=3.9 -y
conda activate dla_labs
```

**Or `venv`:**

```bash
python -m venv venv_dla
source venv_dla/bin/activate  # Linux/macOS
# .\venv_dla\Scripts\activate    # Windows
```

### 3. Install Dependencies

```bash
# Core & Deep Learning
pip install torch torchvision torchaudio numpy matplotlib tqdm scikit-learn jupyterlab tensorboard

# Lab 2: Reinforcement Learning
pip install gymnasium[classic_control,box2d] pygame

# Lab 3: Transformers
pip install transformers datasets evaluate peft accelerate bitsandbytes
```
*(**Pro-Tip:** For GPU acceleration with PyTorch, visit [pytorch.org](https://pytorch.org) for custom installation commands based on your CUDA version.)*

## 📂 Project Layout

```
.
├── Lab1.ipynb              # 🧪 DL Basics
├── Lab2.ipynb              # 🎮 Reinforcement Learning
├── Lab3.ipynb              # 🤖 Transformers
├── BaseTrainingPipeline.py # Core Trainer
├── SLTrainingPipeline.py   # Supervised Learning Trainer
├── RLTrainingPipeline.py   # Policy Gradient RL Trainer
├── QLTrainingPipeline.py   # Q-Learning RL Trainer
├── MLP.py                  # MLP & ResMLP Models
├── CNN.py                  # CNN & ResCNN Models
├── data/                   # (Likely) Datasets
├── checkpoints/            # (Likely) Model Saves
├── logs/                   # (Likely) TensorBoard Logs
└── README.md               # You are here!
```

## 🚀 Launching the Labs

1.  Activate your virtual environment.
2.  Start Jupyter: `jupyter lab` or `jupyter notebook`.
3.  Open the desired `LabX.ipynb` file and run the cells.

    *   **Lab 1:** Expect dataset downloads, model training visuals, and plots.
    *   **Lab 2:** RL agents will interact with environments; `pygame` might render visuals.
    *   **Lab 3:** Hugging Face models/datasets will download; `Trainer` API manages fine-tuning.

## 📊 Visualize Your Progress with TensorBoard

1.  In a new terminal (in the project root):
    ```bash
    tensorboard --logdir=./logs
    ```
2.  Open `http://localhost:6006` in your browser.

## 💡 Key Custom Modules

*   **`TrainingPipelines (*Pipeline.py)`:** Reusable code for supervised, policy-gradient RL, and Q-learning RL training loops, handling epochs, logging, and checkpoints.
*   **`MLP.py` & `CNN.py`:** Flexible classes for creating Multilayer Perceptrons and Convolutional Neural Networks, including residual variants.

---

