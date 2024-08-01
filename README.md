# Trigram Baby Name Generator

Welcome to the Trigram Baby Name Generator! This project uses a trigram-based model implemented in PyTorch to generate names based on a provided dataset of names.

## Overview

This project leverages a simple neural network to predict and generate names using a trigram approach. The model is trained on a dataset of names and uses the trigram (three-character sequence) context to predict the next character in the sequence. The generated names can be used for various purposes, such as generating baby names, character names for stories, or just for fun.

## Features

- Trigram-based name prediction
- Single-layer neural network
- Implemented in PyTorch
- Simple and easy-to-understand code structure

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Sudhan-Dahake/Trigram-Model-Baby-Name-Gen.git
    cd Trigram-Baby-Name-Generator
    ```

2. **Install dependencies:**
    ```bash
    pip install torch
    ```

3. **Prepare your dataset:**
    Ensure you have a `names.txt` file in the root directory. This file should contain a list of names, each on a new line.

## Usage

1. **Train the model**

2. **Generate names**

Run the script to train the model on the 'names.txt' dataset and generate a list of (100) names.
```bash
python Trigram_Model_Name_Generator.py
```

## Code Explanation

### Data Preparation

- **Reading and Processing Names:**
  The names are read from a `names.txt` file and processed to create input-output pairs for the trigram model.

- **Character Mapping:**
  Characters are mapped to integers and vice versa for easy processing.

### Model Training

- **Neural Network:**
  A single-layer neural network is implemented in PyTorch. The model takes two-character sequences as input and predicts the next character.

- **Loss Function:**
  The negative log likelihood loss function is used to train the model.
  Single it's a single layer linear neural net, the loss function comes out to be around 2.3 (not so bad for a basic neural net).
  The more the loss function is closer to 0, the better the model is, atleast for this particular dataset.

- **Gradient Descent:**
  The model is trained using gradient descent, with the weights updated after each iteration to minimize the loss.

### Name Generation

- **Sampling:**
  The trained model is used to generate new names by sampling from the probability distribution of the next character.

## Example Output

Here are some example names generated by the model:

- oworani.
- oe.
- ole.
- iaitoin.
- alox.
- allirasn.
- ha.
- on.
- halayly.
- gvkbej.
- taere.
- aeloneriya.
- adnu.
- budhishagan.
- nadfkwnbrquthani.
- amyn.
- coa.
- aryen.
- auinyl.
- olyna.
