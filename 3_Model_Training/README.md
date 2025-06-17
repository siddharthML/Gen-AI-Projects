To cover in this repo:
1) Full Finetuning
2) Supervised Finetuning (SFT)
3) Reinforcement Learning from Human Feedback (RLHF)
4) Mixture of Experts
5) PEFT (Parameter Efficient Finetuning)
    - LoRA
    - QLoRa

Training libraries to cover:
- Unsloth
- Llama-factory
- deepspeed

--------------------------------------
Finetuning Notes
--------------------------------------
(Notes created from Chip Huyen - AI Engineering)

# 🏆 1) Overview of Finetuning  

Finetuning is the process of adapting a model to a specific task by further training the whole or part of the model. It involves adjusting the model's weights, unlike prompt-based methods which guide behaviour through instructions, context, and tools.  

### 🔥 Finetuning can enhance various aspects of a model, including:  
- **Improving domain-specific capabilities**, such as coding or medical question answering.  
- **Strengthening its safety.**  
- **Improving the model's instruction-following ability**, especially for adherence to specific output styles and formats like JSON or YAML.  

Finetuning is a form of **transfer learning**, a concept introduced in 1976, which focuses on transferring knowledge gained from one task to accelerate learning for a new, related task. This is conceptually similar to how humans transfer skills—like learning a new musical instrument after mastering the piano. 🎹  

Transfer learning, including finetuning, improves **sample efficiency**, allowing a model to learn desired behaviours with fewer examples. For instance, training a legal question-answering model from scratch might require millions of examples, but finetuning a good base model might only need a few hundred. Ideally, much of the necessary learning is already in the base model, and finetuning merely refines its behaviour.  

OpenAI's InstructGPT paper (2022) suggested that finetuning **"unlocks"** capabilities a model already possesses but which are difficult for users to access via prompting alone.  

---

## 🔄 Forms of Finetuning  
Finetuning is part of a model's training process, serving as an extension of pre-training. It can take various forms, including:  

- **Self-supervised finetuning (continued pre-training) 🏗**  
  Finetuning a pre-trained model with self-supervision using cheap task-related data before using expensive task-specific data.  
  - Example: Finetuning a legal question-answering model on raw legal documents before using annotated (question, answer) data.  

- **Supervised finetuning (SFT) 🎯**  
  Training the model using high-quality annotated (input, output) pairs to refine its alignment with human usage and preference.  
  - Example: Instruction finetuning—where the input can be an instruction, and the output a response (open-ended or close-ended).  

- **Preference finetuning ❤️**  
  Further finetuning with reinforcement learning using comparative data (instruction, winning response, losing response) to maximize human preference.  

- **Long-context finetuning 📜**  
  Extending a model's context length, typically by modifying its architecture (e.g., adjusting positional embeddings).  
  - Note: This process is harder and might degrade performance on shorter sequences.  

Finetuning can be performed by both **model developers** (often as "post-training" before release) and **application developers** (adapting a pre-trained or post-trained model to specific needs). The more refined and relevant a base model is, the less adaptation work is required.  

---

## 📌 When to Finetune  
Finetuning generally requires significantly more resources (**data, hardware, ML talent**) compared to prompt-based methods and is usually attempted **after** extensive experimentation with prompting. However, finetuning and prompting are **not** mutually exclusive—they often complement each other.  

### ✅ Reasons to Finetune  
- **Improve model quality** 🌟  
  - Enhance general and task-specific capabilities, especially if the model wasn’t sufficiently trained on the specific task (e.g., handling less common SQL dialects or customer-specific queries).  

- **Ensure specific output formats** 🔢  
  - Improve adherence to structured output formats like **JSON or YAML**.  

- **Bias mitigation** ⚖️  
  - Counteract biases by exposing the model to curated data (e.g., finetuning on texts authored by women to reduce gender bias).  

- **Improve smaller models** 🚀  
  - Finetune a small model to mimic a larger model (**distillation**) for cost and speed efficiency.  

- **Leverage open-source models** 💡  
  - The rise of high-quality open-source models makes finetuning more viable and attractive.  

- **Optimize token usage** ⏳  
  - Reduce token costs by embedding examples into finetuning rather than prompts.  

### ❌ Reasons **Not** to Finetune  
- **Performance degradation on other tasks** ⚠️  
  - Finetuning can lead to a loss in **general** capabilities if overly task-specific.  

- **High up-front investment & maintenance** 💰  
  - Requires **data acquisition, ML knowledge, and infrastructure** investment.  

- **Rapid evolution of base models** 🏗  
  - New base models may quickly outperform finetuned versions, leading to re-training dilemmas.  

- **Prompting might be sufficient** ✍️  
  - A well-crafted prompt can often achieve desired improvements **without** finetuning.  

- **General-purpose models improving** 📈  
  - Some large models are **so good** that domain-specific finetuning may be unnecessary.  

- **Ossification risk** 🧊  
  - Pre-training can "freeze" model weights, making them **less adaptable** during finetuning.  

---

## 🔍 Relationship Between Finetuning & RAG  

The choice between **RAG (Retrieval-Augmented Generation)** and **finetuning** depends on whether a model's failures are **information-based** or **behaviour-based**.  

### 🔹 Use **RAG** for **information-based** failures:  
If the model lacks **factual knowledge**, is **outdated**, or **misses private information**, RAG helps by retrieving relevant external data.  
- **Studies show RAG outperforms finetuning** for fact-based improvements.  
- **Helps mitigate hallucinations** by grounding responses in external sources.  

### 🔹 Use **Finetuning** for **behaviour-based** failures:  
If the model struggles with **formatting**, **relevance**, **safety**, or **style adherence**, finetuning ensures it learns expected behaviour.  

### **Summary:**  
📌 **Finetuning** is for **form**, while **RAG** is for **facts**!  
If a model has **both** information and behaviour issues, start with RAG—it’s **easier to implement** and offers a **strong initial boost**.  

---

## 🚀 Recommended Development Path  
1️⃣ **Maximise performance** with simple **prompting** and context.  
2️⃣ **Add more examples** to prompts (**1-50 shots**).  
3️⃣ **Implement RAG** if missing **information**, starting with **term-based search**.  
4️⃣ If problems persist:  
   - **Use advanced RAG** (embedding-based retrieval).  
   - **Use finetuning** for **formatting & behaviour improvements**.  
5️⃣ **Combine RAG & finetuning** for maximum performance!  


# 🧠 2) Memory Bottlenecks  

Finetuning large-scale models is **memory-intensive**, leading to various techniques aimed at minimizing memory footprint. Understanding these bottlenecks helps in selecting the best finetuning method and estimating **hardware requirements**.  

### 🔑 Key Takeaways Regarding Memory Bottlenecks  
- **Memory is a bottleneck** for both **inference** and **finetuning** in foundation models, with finetuning demanding **much more** memory.  
- **Major contributors to a model's memory footprint** include:  
  - **Number of parameters** 🔢  
  - **Number of trainable parameters** 🎯  
  - **Numerical representations** 🏗  
- **More trainable parameters** ⬆️ lead to a **higher memory footprint**.  
  - Techniques like **Parameter-Efficient Finetuning (PEFT)** help reduce memory by **limiting** the number of trainable parameters.  
- **Quantization** 🏎 (converting to lower precision) is an **effective** way to reduce memory usage.  
- **Inference** typically uses the **smallest** possible bit representations (**16-bit, 8-bit, or even 4-bit**).  
- **Training** is more **precision-sensitive** and is often done in **mixed precision** (**FP32, FP16, or BF16**).  

---

## 🔁 Backpropagation & Trainable Parameters  

The memory needed for **trainable parameters** is crucial for finetuning, driven by the **backpropagation mechanism** used in training neural networks.  

### 🛠 **Trainable Parameter**  
A parameter that can be **updated** during finetuning:  
- **Pre-training:** All parameters are updated.  
- **Inference:** No parameters are updated.  
- **Finetuning:** Some or all parameters are updated, while others remain **"frozen"** ❄️.  

Each training step has **two phases**:  
1️⃣ **Forward pass** → Computes output from input.  
2️⃣ **Backward pass** → Updates model weights using signals from the forward pass.  

During the **backward pass**:  
✅ The model's output is **compared** to the expected result (**ground truth**) to compute **loss** (the mistake).  
✅ The **gradient** is computed (measuring how much each trainable parameter **contributes** to the mistake).  
✅ Each trainable parameter is **adjusted** using its gradient via an **optimizer** (e.g., **Adam** for transformer models).  

### 🏗 **Memory Impact of Trainable Parameters**  
Each **trainable parameter** requires **extra memory** for its **gradient** and **optimizer states**:  
- Example: **Adam optimizer** stores **two values per trainable parameter**.  
- The **more trainable parameters**, the **more memory** needed for these additional values.  

### 🔄 **Activation Memory Considerations**  
Activation memory can be **substantial**, sometimes **surpassing** the memory needed for model weights.  
- **Gradient checkpointing** (or activation recomputation) helps **reduce activation memory** by **recomputing activations** instead of storing them—at the **cost of increased training time**.  

---

## 📊 Memory Calculations  

### 🔍 **Memory Needed for Inference**  
Inference **only** executes the **forward pass**.  

Formula:  
🔹 **Memory for model weights** → `N × M` (where `N = parameter count`, `M = memory per parameter).`  
🔹 **Total memory (including activations & KV vectors, assumed to be 20% extra)** → `N × M × 1.2`.  

🔢 Example:  
A **13B-parameter model** with **2 bytes per parameter** →  
🔹 `13B × 2 bytes × 1.2 = 31.2 GB`.  

A **70B-parameter model** → **140 GB** (for weights alone).  

### 🛠 **Memory Needed for Training**  
Training **includes** memory for:  
✔️ Model weights  
✔️ Activations  
✔️ Gradients  
✔️ Optimizer states  

Formula:  
🔹 `Training memory = model weights + activations + gradients + optimizer states`.  

🔢 Example:  
A **13B model finetuned with Adam optimizer** (3 values per parameter) using **2 bytes/value** for gradients and optimizer states →  
🔹 `13B × 3 × 2 bytes = 78 GB` (for optimizer states alone).  

If **only** **1B** parameters are trainable → ✅ Memory drops to **6 GB**.  

---

## ⚡ Numerical Representations  

The memory required to **store values** **directly** impacts the model’s **memory footprint**.  
- Reducing memory per value (e.g., **4 bytes → 2 bytes**) **halves** the required memory! 🎯  

### 💾 **Floating-Point Formats (FP Family - IEEE 754 Standard)**  
- **FP32**: 32 bits (**4 bytes**), single precision.  
- **FP64**: 64 bits (**8 bytes**), double precision.  
- **FP16**: 16 bits (**2 bytes**), half precision.  
- **BF16**: 16 bits (**2 bytes**), optimized for better range handling.  
- **TF32**: 19 bits, designed for FP32 compatibility.  

⚠️ **Higher precision formats** require **more bits** → increasing memory footprint!  
⚠️ **Lower precision formats** (e.g., FP32 → FP16) **reduce memory usage**, but may **introduce errors**.  

The choice of format **depends on the workload** (numerical stability, sensitivity to precision changes, and hardware compatibility).  

---

## 🎯 Quantization Techniques  

Quantization reduces precision and **shrinks** the model's memory footprint.  
- **Generalizes well across tasks & architectures** 🏗.  
- **Improves inference speed** ⚡ and **batch size capacity** 📈.  

### 🔍 **Quantization vs. Reduced Precision**  
- **Strict Definition:** Quantization **converts values into integer formats** (e.g., INT8, INT4).  
- **Practical Definition:** Quantization refers to **all methods** that lower precision.  

### 🔎 **What to Quantize?**  
✔️ The **largest memory consumers** → **weights & activations**.  
✔️ **Weight quantization** is more common due to **better performance stability**.  

### 🕒 **When to Quantize?**  
**1️⃣ Post-training quantization (PTQ)** → Quantizing a fully trained model (**most common approach**).  
  - 🚀 Major ML frameworks support PTQ.  

**2️⃣ Quantization-aware training (QAT)** → Simulates **low-precision** behavior **during training** for **better inference quality**.  

**3️⃣ Inference Quantization** → Serving models in **16-bit, 8-bit, or even 4-bit**  
  - 🏎 **Popular methods**: LLM.int8(), QLoRA.  
  - 🎭 Can use **mixed precision** formats (FP8, FP4, INT8, INT4).  
  - 🔥 **BitNet b1.58** → Achieves **1-bit precision**, comparable to **16-bit Llama 2** (up to 3.9B parameters).  

### 🚀 **Training Quantization**  
✔️ Harder than inference quantization due to **backpropagation sensitivity**.  
✔️ Often done using **mixed precision**:  
   - **Higher precision** for weights & sensitive values.  
   - **Lower precision** for gradients & activations (e.g., **AMP functionality**).  
✔️ Models are typically trained in **higher precision** then **finetuned in lower precision** for **memory efficiency**.  




# 🔬 3) Finetuning Techniques  

This section covers **memory-efficient finetuning techniques**, focusing on **Parameter-Efficient Finetuning (PEFT)** and **model merging** for creating custom models.  

---

## 🎯 Parameter-Efficient Finetuning (PEFT) Methods  

### 🔹 **Full Finetuning**  
✅ **Updating all model parameters** → requires **significant memory** (e.g., **56GB** for a **7B model** in **16-bit with Adam optimizer**, excluding activations).  
⚠️ Exceeds consumer GPU capacity. Requires **large amounts of annotated data**.  

### 🔹 **Partial Finetuning**  
✅ **Updating only a subset of model parameters** (e.g., **just the last layer**).  
⚠️ Reduces memory footprint but **requires many trainable parameters** to match full finetuning performance.  

### 🔹 **PEFT (Parameter-Efficient Finetuning)**  
Introduced by **Houlsby et al. (2019)**, PEFT achieves **near-full finetuning performance** with **fewer trainable parameters** by:  
✔️ **Adding adapter modules** instead of modifying all model parameters.  
✔️ **Keeping original model parameters frozen** while only adjusting these adapters.  
✔️ **Achieving strong results** (e.g., **within 0.4%** of full finetuning while using **only 3%** of trainable parameters for **BERT** on the GLUE benchmark).  

### ⚖️ PEFT Trade-offs  
❌ **Early adapter methods increased inference latency** due to extra layers.  
✅ **More affordable** → Enables **finetuning on consumer GPUs**.  
✅ **Sample-efficient** → Strong performance even with **smaller datasets** (thousands of examples).  

### 🏗 **PEFT Categories**  
1️⃣ **Adapter-based methods** → Add **modules** to model weights (**LoRA, BitFit, IA3, LongLoRA**).  
2️⃣ **Soft prompt-based methods** → Modify input processing via **trainable tokens** (**Prefix-Tuning, P-Tuning, Prompt Tuning**).  

---

## 🚀 **LoRA (Low-Rank Adaptation) – The Most Popular PEFT Method**  

### 🔍 **How LoRA Works?**  
✔️ **Solves inference latency** issue by modifying **individual weight matrices** instead of adding layers.  
✔️ Decomposes **weight matrix (W)** into two **smaller matrices** `A` and `B` → only `A` and `B` are updated during finetuning.  
✔️ **Final matrix (W')** = Original W + Product of A and B.  

### 🔥 **Why Does LoRA Work Well?**  
✔️ **Low-rank factorization** → Approximates a **large matrix** using **two smaller matrices**, reducing:  
  - **Memory footprint** 📉  
  - **Computational load** ⚡  
✔️ **Does not increase inference latency** 🏎 → LoRA modules **can be merged back** before serving.  

### ⚙️ **LoRA Hyperparameters**  
- **Weight matrices**: Applying LoRA to **query & value matrices** often gives the best results.  
- **Rank (r)**: Typical range **4 to 64**. Increasing `r` **does not always** improve performance.  
- **Alpha (ɑ)**: Controls contribution of `W_AB` to final matrix → typically **ɑ:r between 1:8 and 8:1**.  

### 🏗 **Serving LoRA Adapters**  
1️⃣ **Merge LoRA weights into the original model** → ✅ No extra inference latency.  
2️⃣ **Keep LoRA adapters separate and dynamically load** → ✅ Lower storage, faster task-switching, ❌ Adds latency.  
   - Example: **100 finetuned models sharing a base model** → **1.68B full parameters → 23.3M parameters with LoRA**.  

### 🔥 **QLoRA (Quantized LoRA)**  
📌 **Stores model weights in 4-bit** precision (e.g., **NF4**) but **dequantizes to BF16** during training.  
✔️ **Optimizes CPU-GPU data transfer** via **paged optimizers**.  
✔️ **Enables finetuning large models** (e.g., **65B parameters on a single 48GB GPU**).  
⚠️ **NF4 quantization can increase training time** due to **extra computation**.  

---

# 🔄 **Model Merging & Multi-Task Finetuning Approaches**  

### 🏗 **What is Model Merging?**  
Model merging **combines multiple models** into a **single custom model**, offering **more flexibility than finetuning alone**.  
✔️ **Can be done without GPUs** if no further finetuning is applied afterward.  

### 🔥 **Why Merge Models?**  
✔️ **Leverage complementary strengths** for **better performance** than individual models alone.  
✔️ **Multi-task finetuning** → Avoids **catastrophic forgetting** (losing old skills when learning new ones).  
✔️ **On-device deployment** → Combines **multiple models into one**, reducing **memory requirements**.  
✔️ **Model ensemble** → Combines **outputs** instead of parameters for **superior results**.  

### 🔗 **Model Merging Approaches**  
🔹 **Summing** → Adds weight values of constituent models.  
🔹 **Linear combination** → Most effective for models finetuned from **same base model** (task vectors approach).  
🔹 **SLERP (Spherical Linear Interpolation)** → Merges along the **shortest path** on a spherical surface.  
🔹 **Pruning redundant task parameters** → Improves merged model **quality**.  
🔹 **Layer stacking ("Frankenmerging")** → Combines **blocks or layers** from different models (e.g., **Goliath-120B**).  
🔹 **Concatenation** → Joins constituent parameters → ❌ **Increases memory footprint**, less effective for optimizations.  

---

## 🛠 **Finetuning Tactics for Optimization**  

### 🚀 **Choosing Finetuning Frameworks & Base Models**  
✔️ **Start with the strongest affordable base model** → If it fails, weaker models likely **won’t** work.  
✔️ **Development paths:**  
   - **Progression path**: Test code **on cheapest model**, test data **on mid-tier model**, then **maximize performance**.  
   - **Distillation path**: Use **strongest model on small dataset**, generate **more data**, train **cheaper model**.  
✔️ **Recommended Methods:**  
   - **LoRA → best for starting**.  
   - **Full finetuning → for large datasets (tens of thousands to millions of examples).**  
   - **PEFT → for small datasets (hundreds to a few thousand examples).**  

### 🎯 **Finetuning APIs vs. Frameworks**  
✔️ **APIs** → Easy & fast, but **limited base model support**.  
✔️ **Frameworks** → Flexible, require **compute provisioning** (`unsloth`, `Axolotl`, `PEFT`).  
✔️ **Distributed Training Frameworks** → Needed for **multi-machine setups** (`DeepSpeed`, `ColossalAI`).  

### 🏎 **Hyperparameter Optimization for Finetuning**  
✔️ **Learning Rate** → **1e-7 to 1e-3** (small too slow, big won’t converge).  
✔️ **Batch Size** → **Too small (<8) → unstable training**, **too large → memory bottleneck**.  
✔️ **Epochs** → Small datasets **(4-10 epochs)**, large **(1-2 epochs)**. Monitor for **overfitting**.  
✔️ **Prompt Loss Weight** → Typically **10%** (prioritizes learning from responses).  




# 🏆 4)  Summary of Key Takeaways  

### ✅ **Finetuning Frameworks Make It Straightforward**  
The **actual finetuning process** is often simple due to **available frameworks** with **sensible defaults**.  

### ❓ **Complexity Lies in the Surrounding Context**  
Despite its ease, deciding **whether to finetune** at all is **complex**:  
- Finetuning requires **significant resources** compared to **prompt-based methods**.  
- **Relationship with RAG** plays a role in determining whether **finetuning** or **retrieval-based approaches** are better suited.  

### 🚀 **Full Finetuning vs. Memory-Efficient Methods**  
- **Full finetuning** was **common for smaller models**, but for **large foundation models**, its **memory & data demands** make it **impractical**.  
- This led to the rise of **memory-efficient finetuning techniques**, notably **Parameter-Efficient Finetuning (PEFT)**.  

### 🔥 **PEFT & Quantized Training**  
✔️ **PEFT methods, such as LoRA**, minimize **trainable parameters**, reducing the **memory footprint**.  
✔️ **Quantized training** further **lowers memory requirements** by using **fewer bits per value**.  

### 🌟 **LoRA – Highly Popular for Finetuning**  
📌 **Why LoRA is widely used?**  
✔️ **Parameter-efficient** → Uses fewer trainable parameters without sacrificing performance.  
✔️ **Data-efficient** → Requires fewer examples to achieve strong results.  
✔️ **Modular** → Simplifies **serving** and **combining** multiple finetuned models.  

### 🔄 **Model Merging for Multi-Task Learning & Deployment**  
✔️ **Model merging** combines multiple **finetuned models** into a **single, more capable model**.  
✔️ Useful for:  
  - **Multi-task learning** 🧠  
  - **On-device deployment** 📲  
  - **Model upscaling** 🚀  

### ⚠️ **Data Acquisition Challenges**  
One of the **biggest hurdles in finetuning** is acquiring **high-quality annotated data**, especially **instruction data** for training.  



