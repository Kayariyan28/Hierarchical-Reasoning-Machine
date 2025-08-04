# 🧠 Hierarchical Reasoning Model (HRM) for Arithmetic Proofs

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains a PyTorch implementation of a **Hierarchical Reasoning Model (HRM)**, a neural network designed to solve multi-step arithmetic problems by generating explicit proof-like steps. The architecture is inspired by concepts of iterative refinement and hierarchical processing, mimicking how humans might tackle a complex problem by breaking it down.

---

## 📜 Table of Contents

1.  [**Core Concepts**](#-core-concepts)
2.  [**Repository Structure**](#-repository-structure)
3.  [**How to Run**](#-how-to-run)
4.  [**Detailed Code Explanation**](#️-detailed-code-explanation)
5.  [**Is the Architecture Realistic and Useful?**](#-is-the-architecture-realistic-and-useful)
6.  [**Analysis of a Training Run**](#-analysis-of-a-training-run)
7.  [**License**](#️-license)

---

## 🔬 Core Concepts

The model attempts to solve a specific task: given an integer $T$, find integers $a,b,c$ such that $(a \times b)+c=T$. Instead of just outputting the numbers, it generates a textual "proof": `[LEMMA] a * b = I [LEMMA] I + c = T`.

The architecture combines several advanced ideas:

* **Hierarchical Recurrent Modules**: The model uses two interconnected Transformer-based modules:
    * **L-Module (Low-level, Fast)**: Processes information rapidly at each internal time step, analogous to intuitive, fast thinking (System 1).
    * **H-Module (High-level, Slow)**: Operates at a slower timescale, integrating information from the L-Module to update a high-level "state," analogous to deliberate, slow reasoning (System 2).

* **Iterative Refinement (Seed & Prover)**: The model doesn't just generate one answer. It operates in segments. In each segment, it produces a complete proof. An external `verifier` checks this proof. If the proof is wrong, the model can use its internal state from the failed attempt to "refine" its reasoning and try again in the next segment.

* **Adaptive Computation Time (ACT)**: The model learns *when to stop reasoning*. A `q_head` predicts Q-values for two actions: **Halt** or **Continue**. If the model is confident in its proof (high "Halt" Q-value), it stops. This prevents wasting computational resources. The training for this uses a simple Q-learning-style reward signal.

* **One-Step Gradient Approximation**: Training a deeply recurrent model like this is computationally expensive. This code uses a practical trick: it runs most of the internal reasoning steps `with torch.no_grad()` and only computes gradients on the final step of the internal loop. This makes training feasible while still allowing the model to learn.

---

## 📂 Repository Structure

The project is organized in a modular structure to separate concerns:

```
├── src/                  # Source code for the project
│   ├── config.py         # All hyperparameters and vocabulary
│   ├── data.py           # Data generation and loading logic
│   ├── model.py          # The PyTorch model architecture (HRM)
│   ├── train.py          # Training and validation functions
│   └── evaluate.py       # Inference, verification, and evaluation logic
├── .gitignore            # Specifies files for Git to ignore
├── LICENSE               # Project license (MIT)
├── README.md             # You are here!
├── main.py               # The main script to run the entire pipeline
└── requirements.txt      # Project dependencies
```

---

## 🚀 How to Run

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/hierarchical-reasoning-model.git](https://github.com/your-username/hierarchical-reasoning-model.git)
cd hierarchical-reasoning-model
```

### 2. Create a Virtual Environment (Recommended)

It's best practice to create a virtual environment to manage project dependencies.

```bash
# For Unix/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

Install PyTorch and any other required packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Run the Project

Execute the main script. This single command will handle data generation, model training, and the final evaluation on test examples.

```bash
python main.py
```

You should see output indicating the PyTorch version, the device being used, followed by the training progress and final evaluation results.

---

## ⚙️ Detailed Code Explanation

The code is logically structured into setup, model architecture, data generation, training, and evaluation.

* **1. Setup and Configuration**: Imports necessary libraries and sets random seeds for reproducibility. It automatically selects a CUDA GPU if available. The `CONFIG` dictionary centralizes all hyperparameters, which is excellent practice. The `VOCAB` provides the integer-to-token mapping.

* **2. HRM Core Architecture**: This is the heart of the project.
    * **`PositionalEncoding` & `TransformerBlock`**: Standard components of a Transformer that provide positional context and the self-attention mechanism, respectively.
    * **`RecurrentModule`**: A stack of `TransformerBlock` layers that forms the basis for both the L-Module and H-Module.
    * **`HierarchicalReasoningCore`**: The main model class. Its `forward` method orchestrates the internal reasoning loop (`run_hrm_loop`), which implements the fast/slow dynamic between the L- and H-modules and uses the one-step gradient approximation for efficient training.

* **3. Data Generation**: The `generate_proof_data` function creates the synthetic dataset. The data is then tokenized, padded, and loaded into PyTorch `DataLoader`s for efficient batching.

* **4. Training Loop**: The `train_step` function encapsulates one epoch of training.
    * **Deep Supervision Loop**: The model runs for `MAX_SEGMENTS` iterations, and the loss is averaged across segments. This encourages the model to generate a good proof early.
    * **ACT Loss (Q-Learning)**: It calculates a reward (1 if correct, 0 otherwise) and uses it to train the `q_head`, teaching the model when it's best to halt.
    * **Backpropagation**: The combined sequence and ACT loss is backpropagated. The states (`z_H`, `z_L`) are detached between segments to stabilize training.

* **5. Evaluation and Inference**: The `evaluate_hlrm` function showcases the full iterative refinement process. It gives the model an input, runs a reasoning segment, and uses the `simulated_verifier` to check the output. Based on the verifier's feedback and the model's own halt signal, it either stops or continues to another round of refinement.

---

## 🤔 Is the Architecture Realistic and Useful?

**Yes**, the underlying concepts are both realistic and useful, though the specific problem here is a toy example.

* **Realism**
    * **Hierarchical Processing**: The idea of combining fast and slow processing systems is a major area of research in AI and cognitive science. This code provides a plausible, simplified implementation of this concept.
    * **Iterative Refinement**: For complex tasks like code generation or mathematical theorem proving, a "one-shot" generation is often wrong. Iterative refinement is a very realistic and powerful paradigm.
    * **Gradient Approximation**: The one-step gradient trick is a practical and necessary simplification for training such deep recurrent architectures.

* **Usefulness**
    The architecture and training strategy could be adapted to solve much more significant problems:
    * 🤖 **Automated Theorem Proving**: Instead of arithmetic, the lemmas could be logical steps in a formal proof.
    * 💻 **Program Synthesis**: The model could generate code, with the "verifier" being a compiler or a set of unit tests, allowing it to iteratively fix its own errors.
    * ❓ **Complex Question Answering**: The model could generate intermediate "facts" or "conclusions" as lemmas before arriving at a final answer.
    * 🧪 **Scientific Discovery**: It could propose a sequence of molecular reactions (lemmas) to synthesize a target compound.

---

## 📊 Analysis of a Training Run

The provided output is very insightful.

* **Training Performance**: The model reaches **100% validation accuracy after just 25 epochs**. This is highly successful but also a bit **suspicious**. In a real-world, complex problem, reaching 100% accuracy this quickly is rare and suggests the synthetic data generation process is too simple or predictable.

* **Likely Cause**: The model learned a very effective **shortcut**: always use `a=2` and `c=1`. For example, to prove `19`, it generates `2 * 9 + 1`. This is a valid solution, but it shows the model found the simplest path to success rather than learning a general factorization ability. The dataset likely did not contain enough variety to prevent this.

* **Inference Performance & Accuracy**: The model achieves a **5/5 success rate** on the test examples, and the generated proofs are mathematically correct. The output successfully demonstrates the HLRM inference loop, including the ACT halting mechanism. For the proof of `11`, the `Q-Halt` value (1.023) is higher than the `Q-Continue` value (1.012), meaning the model correctly "knows" it has found the solution.

* **Conclusion**: The model works perfectly for the simple task it was trained on. To test true "reasoning," the data generation would need to be much more complex (e.g., forcing proofs with larger prime factors, using different operations, etc.).

---

## ⚖️ License

This project is licensed under the MIT License. See the `LICENSE` file for details.

