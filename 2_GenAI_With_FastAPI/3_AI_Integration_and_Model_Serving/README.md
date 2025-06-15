# ✨ Chapter 3: AI Integration and Model Serving

## 🎯 Chapter Goals
This chapter aims to teach how different Generative AI (GenAI) models work, how to integrate and serve them within a FastAPI application, and various strategies for model serving. It also covers building a user interface for prototyping and leveraging middleware for service monitoring.

---

## ⚡ Serving Generative Models

This section introduces various GenAI models and how to serve them, emphasizing that understanding model internals can help enhance outputs.

### 📝 Language Models
- **🔄 Transformers vs. Recurrent Neural Networks (RNNs)**  
  Historically, RNNs processed text sequentially with a memory store (state vector), struggling with long-range dependencies in large texts. Transformers, introduced in *Attention Is All You Need*, use a self-attention mechanism to model relationships between words regardless of their distance, processing words non-sequentially for efficient parallelization on GPUs.

- **💻 Hardware Requirements for Open Source Large Language Models (LLMs)**  
  Running large open-source LLMs like Snowflake Arctic (480B parameters) and Llama 3.1 (405B parameters) requires significant GPU VRAM (e.g., 8xH100 instances, each with 80 GB VRAM). Consumer GPUs (e.g., NVIDIA 4090 RTX with 24 GB VRAM) may not run models over 30B parameters without quantization (compression). Self-hosting LLMs is challenging due to high memory needs, making lightweight models (up to 3B) or third-party LLM provider APIs (e.g., OpenAI) common choices.

- **🔢 Tokenization and Embedding**  
  Neural networks process numbers, so text must be broken into "tokens" (words, syllables, symbols) via tokenization. These tokens are then converted into "embeddings"—dense vectors of real numbers that capture semantic information.

- **📏 Positional Encoding**  
  Since transformers process words simultaneously, positional encoding adds positional information to token embeddings, ensuring the model understands word order and context.

- **🔮 Autoregressive Prediction**  
  Transformers are autoregressive, meaning future predictions are based on past values, generating text token by token until a stop token is reached.

- **🚀 Integrating a language model into your application**  
  An example using TinyLlama (1.1 billion parameters) from Hugging Face's transformers library is provided, showing how to expose it via a FastAPI endpoint (`/generate/text`).

- **⚠️ Hallucinations**  
  LLMs can produce plausible but incorrect or made-up facts, known as hallucinations. It is crucial to warn users to fact-check outputs.

- **🖥️ Connecting FastAPI with Streamlit UI**  
  Streamlit is introduced as a simple Python package for quickly developing UIs to test and prototype with models.

- **🧠 Transformer Variants**  
  Three types exist:
  - **Encoder** (for understanding inputs, e.g., BERT)
  - **Decoder** (for generating outputs, e.g., GPT)
  - **Encoder-Decoder** (for sequence-to-sequence tasks, e.g., T5)

### 🔊 Audio Models
- **🎵 Bark Model**  
  Suno AI's transformer-based Bark model can generate realistic multilingual speech and audio.

### 🖼️ Vision Models
- **🎨 Stable Diffusion (SD)**  
  SD models encode images into a latent space, generating images via forward (adding noise) and reverse (removing noise) diffusion processes.

- **⚙️ Low-Rank Adaptation (LoRA)**  
  A fine-tuning strategy that reduces GPU memory needed for training.

### 📽️ Video Models
- **📹 OpenAI Sora**  
  A generalist large vision diffusion transformer model capable of generating high-definition videos.

### 🏗️ 3D Models
- **🌀 OpenAI Shap-E**  
  An open-source model for generating specific 3D shapes.

---

## 🔄 Strategies for Serving Generative AI Models

### 🔄 Model Agnostic: Swap Models on Every Request
- **Pros**: Frees up memory.
- **Cons**: Slower processing due to constant reloading.

### ⚡ Compute Efficient: Preload Models with the FastAPI Lifespan
- **Pros**: Avoids reloading heavy models.
- **Cons**: High RAM/VRAM usage.

### 🌐 Lean Deployment: Serve Models Externally
- **Cloud Providers:** Azure Machine Learning Studio.
- **BentoML:** A FastAPI-inspired framework for ML.
- **Model Provider APIs:** OpenAI, Anthropic, Mistral.

---

## 🛠️ The Role of Middleware in Service Monitoring
Middleware components intercept communication between API endpoints and clients.

### ✅ Common Use Cases
- Adding headers
- Basic checks
- Supporting CORS
- Logging and monitoring

### 📑 Example: Logging Service Usage
Middleware logs request details to a CSV file.

### ⚠️ **Warning**
Direct logging to disk in production containers can lead to data loss.

---

## 🏁 Summary

Chapter 3 covered:
✅ Various GenAI models (text, image, audio, video, 3D).  
✅ Integrating them into a FastAPI service with a Streamlit UI.  
✅ Model-serving strategies: model swapping, preloading, and external serving.  
✅ Middleware for service monitoring and logging.  
