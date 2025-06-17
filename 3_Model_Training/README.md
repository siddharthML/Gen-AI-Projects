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

### 1) Overview of Finetuning

Finetuning is the process of adapting a model to a specific task by further training the whole or part of the model. It involves adjusting the model's weights, unlike prompt-based methods which guide behaviour through instructions, context, and tools.

Finetuning can enhance various aspects of a model, including:
*   Improving domain-specific capabilities, such as coding or medical question answering.
*   Strengthening its safety.
*   Most commonly, improving the model's instruction-following ability, especially for adherence to specific output styles and formats like JSON or YAML.

Finetuning is a form of *transfer learning*, a concept introduced in 1976, which focuses on transferring knowledge gained from one task to accelerate learning for a new, related task. This is conceptually similar to how humans transfer skills, like learning a new musical instrument after learning the piano. Transfer learning, including finetuning, improves *sample efficiency*, allowing a model to learn desired behaviours with fewer examples. For instance, training a legal question-answering model from scratch might require millions of examples, but finetuning a good base model might only need a few hundred. Ideally, much of the necessary learning is already in the base model, and finetuning merely refines its behaviour. OpenAI's InstructGPT paper (2022) suggested that finetuning "unlocks" capabilities a model already possesses but which are difficult for users to access via prompting alone.

Finetuning is part of a model's training process, serving as an extension of pre-training. It can take various forms, including:
*   **Self-supervised finetuning (continued pre-training)**: Finetuning a pre-trained model with self-supervision using cheap task-related data before using expensive task-specific data. For example, finetuning a legal question-answering model on raw legal documents before using annotated (question, answer) data.
*   **Supervised finetuning (SFT)**: Training the model using high-quality annotated (input, output) pairs to refine its alignment with human usage and preference. The input can be an instruction, and the output a response (open-ended or close-ended). This is also known as instruction finetuning, though the term can be ambiguous.
*   **Preference finetuning**: Further finetuning with reinforcement learning using comparative data (instruction, winning response, losing response) to maximize human preference.
*   **Long-context finetuning**: Extending a model's context length, typically by modifying its architecture (e.g., adjusting positional embeddings). This is harder and might degrade performance on shorter sequences.

Finetuning can be performed by both model developers (often as "post-training" before release) and application developers (adapting a pre-trained or post-trained model to specific needs). The more refined and relevant a base model is, the less adaptation work is required.

#### When to Finetune

Finetuning generally requires significantly more resources (data, hardware, ML talent) compared to prompt-based methods and is usually attempted after extensive experimentation with prompting. However, finetuning and prompting are not mutually exclusive and often complement each other for real-world problems.

**Reasons to finetune**:
*   **Improve model quality**: Enhance general and task-specific capabilities, especially if the model was not sufficiently trained on the specific task (e.g., handling less common SQL dialects or customer-specific queries).
*   **Ensure specific output formats**: Commonly used to improve a model's ability to generate outputs adhering to specific structures (e.g., JSON or YAML).
*   **Bias mitigation**: Exposing the model to carefully curated data during finetuning can counteract biases perpetuated from its original training data (e.g., reducing gender bias by finetuning on texts authored by women).
*   **Improve smaller models**: A common approach is to finetune a small model to imitate the behaviour of a larger model (distillation), making it cheaper and faster to use in production. A small, finetuned model might even outperform a much larger out-of-the-box model on a specific task.
*   **Leverage open-source models**: The proliferation of high-quality open-source models has made finetuning more viable and attractive.
*   **Optimise token usage (historical)**: Finetuning can replace including many examples in each prompt, leading to shorter prompts and reduced cost/latency. While prompt caching has reduced this benefit, finetuning still allows for an unlimited number of examples, unlike prompts limited by context length.

**Reasons not to finetune**:
*   **Performance degradation on other tasks**: Finetuning for a specific task can degrade performance on other tasks, especially for models intended for diverse prompts. This might necessitate finetuning on all relevant queries or using separate models.
*   **High up-front investment and maintenance**: Requires substantial investment in data acquisition (slow and expensive), ML knowledge (evaluating base models, understanding training knobs, monitoring, debugging), and serving infrastructure (inference optimization).
*   **Rapid evolution of base models**: New base models are developed rapidly and may outperform existing finetuned models, posing a dilemma for switching or re-finetuning.
*   **Prompting might be sufficient**: Many improvements can be achieved without finetuning through carefully crafted prompts and context, especially if prompt experiments were minimal or unsystematic initially.
*   **General-purpose models can improve**: General-purpose models are becoming increasingly capable, sometimes outperforming domain-specific models, making the argument for finetuning less compelling in some cases.
*   **Ossification**: For large datasets, finetuning on top of a pre-trained model can sometimes be worse than training from scratch due to ossification, where pre-training "freezes" weights, making them less adaptable to finetuning data. Smaller models are more susceptible to this.

#### Relationship between Finetuning and RAG

The choice between RAG (Retrieval-Augmented Generation) and finetuning depends on whether a model's failures are *information-based* or *behaviour-based*.

*   **RAG for information-based failures**: If the model fails due to a lack of information (e.g., factually wrong or outdated answers, missing private organizational info), a RAG system provides access to relevant external information. Studies show RAG outperforms finetuning for tasks requiring up-to-date information. RAG can also help mitigate hallucinations by providing models with external knowledge.
*   **Finetuning for behaviour-based failures**: If the model has behavioural issues (e.g., outputs that are factually correct but irrelevant, malformatted, or unsafe, or failing to follow expected output formats), finetuning can help. It helps the model understand and follow specific syntaxes and styles.

In summary, finetuning is for *form*, and RAG is for *facts*.
If a model has both information and behaviour issues, starting with RAG is often recommended because it's generally easier to implement and can provide a significant performance boost. RAG and finetuning are not mutually exclusive and can be combined for even greater performance.

A recommended development path:
1.  Maximise performance with simple prompting and context.
2.  Add more examples to prompts (1-50 shots).
3.  Implement RAG if information is missing, starting with basic retrieval methods like term-based search.
4.  If information-based failures persist, explore advanced RAG (e.g., embedding-based retrieval). If behavioural issues persist, opt for finetuning.
5.  Combine RAG and finetuning for maximum performance.

### 2) Memory Bottlenecks

Finetuning large-scale models is memory-intensive, leading many techniques to focus on minimizing memory footprint. Understanding these bottlenecks helps in selecting the best finetuning method and estimating hardware requirements.

Key takeaways regarding memory bottlenecks:
*   Memory is a bottleneck for both inference and finetuning of foundation models, with finetuning typically requiring much more memory due to how neural networks are trained.
*   Major contributors to a model's memory footprint during finetuning are its number of parameters, its number of *trainable parameters*, and its *numerical representations*.
*   More trainable parameters lead to a higher memory footprint. Techniques like Parameter-Efficient Finetuning (PEFT) reduce memory by reducing the number of trainable parameters.
*   Quantization (converting to lower precision) is a straightforward and efficient way to reduce a model's memory footprint.
*   Inference typically uses as few bits as possible (16, 8, or 4 bits).
*   Training is more sensitive to numerical precision and is often done in *mixed precision* (some operations in 32-bit, others in 16-bit or 8-bit).

#### Backpropagation and Trainable Parameters

The memory needed for each trainable parameter is a key factor in finetuning, driven by the *backpropagation* mechanism used to train neural networks.
*   **Trainable parameter**: A parameter that can be updated during finetuning. During pre-training, all parameters are updated; during inference, none are. In finetuning, some or all parameters can be updated, while unchanged parameters are "frozen".
*   Each training step involves two phases:
    1.  **Forward pass**: Computes the output from the input.
    2.  **Backward pass**: Updates the model's weights using aggregated signals from the forward pass.
*   During the backward pass:
    1.  The model's output is compared to the expected output (ground truth) to compute the *loss* (the mistake).
    2.  The *gradient* (how much each trainable parameter contributes to the mistake) is computed. There's one gradient value per trainable parameter.
    3.  Trainable parameter values are adjusted using their gradients, determined by an *optimizer* (e.g., Adam for transformer models).
*   Each trainable parameter requires additional memory for its gradient and optimizer states (e.g., Adam optimizer stores two values per trainable parameter). The more trainable parameters, the more memory is needed for these additional values.
*   *Activation memory* can also be substantial, sometimes dwarfing the memory needed for model weights if stored for gradient computation. Techniques like *gradient checkpointing* (or activation recomputation) reduce activation memory by recomputing activations instead of storing them, at the cost of increased training time.

#### Memory Calculations

Approximate formulas for memory usage:

*   **Memory needed for inference**: Only the forward pass is executed.
    *   Memory for model's weights: `N × M` (N = parameter count, M = memory per parameter).
    *   Total memory (including activations and KV vectors, assumed 20% of weights): `N × M × 1.2`.
    *   Example: A 13B-parameter model with 2 bytes/parameter needs `13B × 2 bytes × 1.2 = 31.2 GB`. A 70B-parameter model needs `140 GB` for weights alone.
*   **Memory needed for training**: Includes model weights, activations, gradients, and optimizer states.
    *   `Training memory = model weights + activations + gradients + optimizer states`.
    *   Example: Updating all parameters in a 13B model with Adam optimizer (3 values per parameter) and 2 bytes/value for gradients/optimizer states needs `13B × 3 × 2 bytes = 78 GB` for these states alone. If only 1B parameters are trainable, this drops to `6 GB`.

#### Numerical Representations

The memory required to represent each value directly contributes to the overall memory footprint. Reducing this memory (e.g., from 4 bytes to 2 bytes per parameter) halves the memory needed for weights.
*   Neural network values are traditionally represented as floating-point numbers (FP family, IEEE 754 standard):
    *   **FP32**: 32 bits (4 bytes), single precision.
    *   **FP64**: 64 bits (8 bytes), double precision.
    *   **FP16**: 16 bits (2 bytes), half precision.
    *   **BF16 (BFloat16)**: 16 bits (2 bytes), with more bits for range and fewer for precision than FP16.
    *   **TF32**: 19 bits, designed for functional compatibility with FP32.
*   Higher precision formats use more bits. Converting to a lower-precision format (e.g., FP32 to FP16) reduces precision and can introduce value changes or errors.
*   The choice of format depends on the workload's numerical values, sensitivity to changes, and hardware.

#### Quantization Techniques

Quantization (reducing precision) is a cheap and effective way to reduce a model's memory footprint. It is straightforward and generalises across tasks and architectures.
*   **Quantization vs. Reduced Precision**: Strictly, quantization means converting to integer formats, but in practice, it refers to all techniques that convert values to a lower-precision format.
*   **What to quantize**: Ideally, quantize the largest memory consumers (weights, activations). Weight quantization is more common due to more stable performance impact.
*   **When to quantize**:
    *   **Post-training quantization (PTQ)**: Quantizing a fully trained model. Most common for application developers. Major ML frameworks offer PTQ.
    *   **Quantization-aware training (QAT)**: Simulates low-precision behaviour during training to produce high-quality low-precision models for inference.
*   **Inference quantization**: Increasingly common to serve models in 16 bits, 8 bits, and even 4 bits (e.g., LLM.int8(), QLoRA). Can use minifloat formats (FP8, FP4) or integer formats (INT8, INT4). A model can be served in *mixed precision*. There are attempts at 1-bit representation (e.g., BitNet b1.58, comparable to 16-bit Llama 2 up to 3.9B parameters). Reduced precision often improves computation speed and allows larger batch sizes, though it can add latency due to format conversion.
*   **Training quantization**: Harder due to backpropagation's sensitivity to lower precision. Often done in *mixed precision*, where higher precision is maintained for weights or sensitive values, and lower precision for gradients and activations (e.g., AMP functionality). Models can be trained in higher precision then finetuned in lower precision.

### 3) Finetuning Techniques

This section covers memory-efficient finetuning techniques, focusing on parameter-efficient finetuning (PEFT), and model merging for creating custom models.

#### Parameter-Efficient Finetuning (PEFT) Methods

*   **Full finetuning**: Updating all model parameters. This requires substantial memory (e.g., 56GB for a 7B model in 16-bit with Adam optimizer, excluding activations), exceeding consumer GPU capacity. Also demands large amounts of high-quality annotated data.
*   **Partial finetuning**: Updating only a subset of model parameters (e.g., only the last layer). Reduces memory footprint but is *parameter-inefficient*, meaning many trainable parameters are needed to achieve performance comparable to full finetuning.
*   **PEFT (Parameter-Efficient Finetuning)**: Introduced by Houlsby et al. (2019). Aims to achieve performance close to full finetuning with *significantly fewer trainable parameters*. This is done by inserting additional parameters (adapter modules) into the model in specific places and updating only these adapters, keeping original model parameters frozen. This can achieve strong performance (e.g., within 0.4% of full finetuning with only 3% of trainable parameters for BERT on GLUE benchmark).
    *   **Downside**: Early adapter methods increased inference latency due to additional layers.
    *   **Benefits**: Enables finetuning on more affordable hardware, making it accessible. PEFT methods are generally both *parameter-efficient* and *sample-efficient* (strong performance with fewer examples, e.g., a few thousand).
*   **PEFT Categories**:
    *   **Adapter-based methods (additive methods)**: Involve additional modules added to the model weights (e.g., LoRA, BitFit, IA3, LongLoRA).
    *   **Soft prompt-based methods**: Modify how the model processes input by introducing special trainable tokens. These "soft prompts" are continuous vectors (like embeddings), not human-readable, and are optimized through backpropagation. They are inserted at different locations (e.g., prefix-tuning, P-Tuning, prompt tuning).

**LoRA (Low-Rank Adaptation)**: The most popular adapter-based method.
*   **Mechanism**: Addresses the latency issue of earlier adapter methods. Instead of adding layers, LoRA applies to individual weight matrices (W) by decomposing them into the product of two smaller matrices, A and B (dimensions `n × r` and `r × m` respectively, where `r` is the LoRA rank). Only A and B are updated during finetuning; W remains intact. The product `W_AB` is added to W to form `W'`.
*   **How it works**: Built on *low-rank factorization*, a dimensionality reduction technique where a large matrix is approximated by the product of two smaller matrices. This reduces parameters, computation, and memory. It is believed to work because LLMs have low intrinsic dimensions, and pre-training acts as a compression framework for downstream tasks.
*   **Benefits**: Parameter-efficient and sample-efficient. Achieves comparable or better performance than full finetuning with significantly fewer trainable parameters (e.g., ~0.0027% for GPT-3). It does not incur extra inference latency because the LoRA modules can be *merged back* into the original layers prior to serving.
*   **Configurations**:
    *   **Weight matrices to apply LoRA to**: Can be applied to individual weight matrices. Empirical observations suggest applying LoRA to more weight matrices (including feedforward layers) yields better results. If choosing only two attention matrices, query and value matrices often yield the best results.
    *   **Rank (r)**: The dimension of the smaller matrices. A small `r` (e.g., 4 to 64) is usually sufficient, though higher ranks might be necessary in some cases. Increasing `r` indefinitely does not necessarily increase performance.
    *   **Alpha (ɑ)**: A hyperparameter determining how much the product `W_AB` contributes to the new matrix. Often chosen so `ɑ:r` is between 1:8 and 8:1.
*   **Serving LoRA adapters**:
    1.  Merge LoRA weights (A and B) into the original model (W') before serving: No extra inference latency.
    2.  Keep LoRA adapters separate and load dynamically: Adds latency but significantly reduces storage needed for serving multiple finetuned models that share a base model (e.g., 100 finetuned models for customers can go from 1.68B full-rank parameters to 23.3M parameters with LoRA). Makes switching between tasks faster.
*   **Drawback**: Does not offer performance as strong as full finetuning in all cases. More challenging to implement manually than full finetuning, but PEFT frameworks (Hugging Face PEFT, Axolotl, unsloth, LitGPT) support it out-of-the-box for popular models.
*   **Quantized LoRA (QLoRA)**: Stores model weights in 4 bits (e.g., NF4) but dequantizes to BF16 for forward and backward passes. Also uses paged optimizers for CPU-GPU data transfer. Allows finetuning large models (e.g., 65B parameters) on a single 48GB GPU. Can be competitive with larger models like ChatGPT. NF4 quantization can be expensive and increase training time due to extra computation.

#### Model Merging and Multi-Task Finetuning Approaches

Model merging allows creating a custom model by combining multiple models, offering greater flexibility than finetuning alone. It can be done without GPUs if no further finetuning is applied after merging.
*   **Goal**: Create a single model that provides more value than constituent models separately, often through improved performance on a task by leveraging complementary strengths.
*   **Use Cases**:
    *   **Multi-task finetuning**: Addresses *catastrophic forgetting* (model forgetting old tasks when trained on new ones in sequential finetuning) by finetuning models on different tasks in parallel and then merging them.
    *   **On-device deployment**: Merging multiple task-specific models into one multi-task model reduces memory requirements for devices with limited capacity.
    *   **Model ensemble (related concept)**: Combines outputs of multiple models to achieve better performance than any single model. Differs from merging by keeping constituent models intact and combining outputs rather than parameters.
*   **Model Merging Approaches**: Differ in how constituent parameters are combined.
    *   **Summing**: Adding weight values of constituent models.
        *   **Linear combination**: Most effective for models finetuned on the same base model. Involves creating *task vectors* (delta parameters by subtracting base model from finetuned model) and linearly combining them.
        *   **Spherical Linear Interpolation (SLERP)**: Merges parameters along the shortest path on a spherical surface, especially effective for combining models from different training runs of the same base model. Only merges two vectors at a time.
        *   **Pruning redundant task-specific parameters**: Removing minor adjustments from finetuning that don't significantly contribute to performance. Improves quality of merged models, especially with more models.
    *   **Layer stacking (Frankenmerging)**: Combining different layers or blocks from multiple models (e.g., taking 72 layers from each of two Llama 2 models to create Goliath-120B). Can be used to train Mixture-of-Experts (MoE) models by taking a pre-trained model, copying layers, adding a router, and further training. Also used for *depthwise scaling* to create larger models by repeating layers.
    *   **Concatenation**: Concatenating parameters of constituent models. The merged component's parameter count is the sum of all constituent components. Not recommended for memory footprint reduction; performance gains might not be worth the increased parameters.

#### Finetuning Tactics for Optimization

Practical considerations for finetuning:
*   **Finetuning frameworks and base models**:
    *   **Base models**: Choose the most powerful model affordable initially to test feasibility. If it struggles, weaker models likely won't improve. If it meets needs, explore weaker models.
    *   **Development paths**:
        *   **Progression path**: Test code with cheapest model, test data with a middling model, then push performance with the best model, finally mapping price/performance frontier with all models.
        *   **Distillation path**: Start with a small dataset and strongest model, use finetuned model to generate more training data, then train a cheaper model with the new dataset.
    *   **Finetuning methods**: LoRA is good for starting; full finetuning is for larger datasets (tens of thousands to millions of examples). PEFT methods work well with smaller datasets (hundreds to a few thousand examples). Adapter-based methods like LoRA are efficient for serving multiple models sharing a base model.
    *   **Finetuning APIs vs. Frameworks**: APIs offer ease and speed but limited base model support and fewer customizable knobs. Frameworks (LLaMA-Factory, unsloth, PEFT, Axolotl, LitGPT) offer more flexibility but require provisioning compute. Distributed training frameworks (DeepSpeed, PyTorch Distributed, ColossalAI) are needed for multi-machine setups.
*   **Finetuning hyperparameters**:
    *   **Learning rate**: Determines how fast parameters change. Too small, learning is slow; too big, model won't converge. Typical range: 1e-7 to 1e-3. Can vary during training (learning rate schedules).
    *   **Batch size**: Number of examples learned from in each step. Small batch sizes (fewer than eight) can lead to unstable training. Larger batch sizes are more stable but require more memory. *Gradient accumulation* can mitigate small batch size issues by accumulating gradients over several batches before updating weights.
    *   **Number of epochs**: Passes over the training data. Small datasets might need more epochs (4-10) than large ones (1-2). Overfitting occurs if training loss decreases but validation loss increases; indicates too many epochs.
    *   **Prompt loss weight**: Determines how much prompt tokens contribute to the loss during training compared to response tokens. Typically set to 10% to prioritize learning from responses.

### 4) Summary of Key Takeaways


*   The actual finetuning process is often straightforward due to available frameworks and their sensible defaults.
*   However, the surrounding context of finetuning is complex, including the decision of whether to finetune at all, considering its resource intensity compared to prompt-based methods and its relationship with RAG.
*   While full finetuning was common for smaller models, its resource demands (memory, data) make it impractical for large foundation models. This led to the development of memory-efficient techniques.
*   PEFT methods, such as LoRA, are key to reducing finetuning's memory footprint by minimizing trainable parameters. Quantized training further reduces memory by using fewer bits for values.
*   LoRA is highly popular due to its parameter-efficiency, data-efficiency, and modularity, which simplifies serving and combining multiple finetuned models.
*   Model merging techniques enable combining multiple finetuned models into a single, more capable one, useful for multi-task learning, on-device deployment, and model upscaling.
*   A significant challenge in finetuning is acquiring high-quality annotated data, particularly instruction data.
