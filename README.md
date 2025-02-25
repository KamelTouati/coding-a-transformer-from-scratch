# Transformer From Scratch

## Ever wondered how ChatGPT works under the hood?

In this deep dive, we’ll build a **decoder-only Transformer** from scratch—the fundamental architecture behind models like ChatGPT.

---

## How Transformers Process Input Using Attention Mechanisms

Unlike traditional RNNs that process text sequentially, **Transformers analyze entire sentences at once using self-attention**.

Self-attention allows a word to focus on other relevant words in a sentence, regardless of their position.

For example, in the sentence:

> *"The cat sat on the mat because it was tired."*

The word **"it"** must reference **"cat"**, not **"mat"**.

Self-attention assigns weights to words, helping the model understand relationships effectively.

### Mathematically, the process involves:

1. **Computing Query (Q), Key (K), and Value (V) matrices.**
2. **Calculating attention scores using the formula:**
   \[
   \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d}} \right) V
   \]
3. **Applying softmax activation to normalize the weights.**
4. **Generating a weighted sum of values (V) to determine the output representation.**

This mechanism enables parallel processing, making training faster while capturing long-range dependencies.

---

## Masked Self-Attention: Enforcing Unidirectional Processing

Since GPT models generate text step by step, they must not "see" future words during training.

To achieve this, **masked self-attention** is applied, where the model can only attend to previous words.

### How does it work?

- A **mask** is applied to the attention scores, setting future token weights to negative infinity (-\infty).
- This ensures that **softmax assigns zero probability to future words**, preventing information leakage.
- During training, the model only uses valid past context to make predictions.

#### Masked Attention Formula:
\[
\text{MaskedAttention}(Q, K, V, M) = \text{softmax}\left( \frac{QK^T}{\sqrt{d}} + M \right) V
\]

Where **M** is the mask matrix that assigns -\infty to future tokens.

For example, given the input **"The sun is"**, the model should predict **"shining"** based only on the preceding words.

---

## How to Implement Word Embeddings & Positional Encoding

Neural networks cannot process raw text directly, so words need to be converted into numerical representations.

### **Word Embeddings:**
Each word is mapped to a fixed-length numerical vector.

Example: **"dog"** \(\to\) \([0.3, 0.8, -0.1, 0.5]\)

Similar words (e.g., **"dog"** and **"puppy"**) have closer embeddings, capturing semantic relationships.

### **Positional Encoding:**
Unlike RNNs, Transformers do not inherently track word order. To address this, we add **sine and cosine functions** to represent positions:

\[
PE(pos, 2i) = \sin\left( \frac{pos}{10000^{(2i/d)}} \right)
\]
\[
PE(pos, 2i+1) = \cos\left( \frac{pos}{10000^{(2i/d)}} \right)
\]

This helps the model differentiate word order, preserving sentence structure during training.

---

## Efficient Training with Adam Optimizer

Once the model is built, an optimized training loop is essential for efficient learning.

- **Adam Optimizer:** Combines **momentum** and **RMSProp** to adaptively adjust the learning rate for faster convergence.
- **Loss Function:** Cross-entropy loss is used to measure how well the model predicts the next token.

### **Training Steps**

1. Tokenize input and generate embeddings.
2. Perform a forward pass through the Transformer blocks.
3. Compute the loss and backpropagate gradients.
4. Update model weights using the **Adam optimizer**.
5. Repeat for multiple epochs.

---

## Want the full PyTorch implementation?

Here is the link:

#AI #MachineLearning #DeepLearning #Transformers #NLP #PyTorch #ChatGPT #Coding #ArtificialIntelligence

