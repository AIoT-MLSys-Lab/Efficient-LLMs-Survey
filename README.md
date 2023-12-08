# Efficient Large Language Models: A Survey  

> **[Efficient Large Language Models: A Survey](http://arxiv.org/abs/2312.03863)**[ [arXiv]](http://arxiv.org/abs/2312.03863)

> *Zhongwei Wan<sup>1</sup>, Xin Wang<sup>1</sup>, Che Liu<sup>2</sup>, Samiul Alam<sup>1</sup>, Yu Zheng<sup>3</sup>, Zhongnan Qu<sup>4</sup>, Shen Yan<sup>5</sup>, Yi Zhu<sup>6</sup>, Quanlu Zhang<sup>7</sup>, Mosharaf Chowdhury<sup>8</sup>, Mi Zhang<sup>1</sup>*

> *<sup>1</sup>The Ohio State University, <sup>2</sup>Imperial College London, <sup>3</sup>Michigan State University, <sup>4</sup>Amazon AWS AI, <sup>5</sup>Google Research, <sup>6</sup>Boson AI, <sup>7</sup>Microsoft Research Asia, <sup>8</sup>University of Michigan*  

## What is This Survey About?

Large Language Models (LLMs) have demonstrated remarkable capabilities in important tasks such as natural language understanding, language generation, and complex reasoning and have the potential to make a substantial impact on our society. Such capabilities, however, come with the considerable resources they demand, highlighting the strong need to develop effective techniques for addressing the efficiency challenges posed by LLMs. In this survey, we provide a systematic and comprehensive review of efficient LLMs research. We organize the literature in a taxonomy consisting of three main categories, covering distinct yet interconnected efficient LLMs topics from model-centric, data-centric, and framework-centric perspective, respectively. We hope our survey can serve as a valuable resource to help researchers and practitioners gain a systematic understanding of the research developments in efficient LLMs and inspire them to contribute to this important and exciting field. We will actively maintain this repository and incorporate new research as it emerges. 

## Why Efficient LLMs are Needed?

![img/image.jpg](img/image.jpg)

Although LLMs are leading the next wave of AI revolution, the remarkable capabilities of LLMs come at the cost of their substantial resource demands. Figure 1 (left) illustrates the relationship between model performance and the carbon emissions during training for LLaMA series. As shown, the amount of carbon emitted grows exponentially as the number of model parameter scales up. In addition to training, inference also contributes quite significantly to the operational cost of LLMs. As depicted in Figure 2 (right), more advanced LLMs exhibit higher memory usage and energy consumption during inference, presenting challenges for these models in expanding their reach to a broader customer base and diverse applications in a cost-effective way. With the rapid expansion of applications and the customer base for LLMs, the operational cost during inference in terms of energy consumption and memory usage would increase and exceed the training cost and become the dominant factor for the overall environmental impact. 

## Table of Content
- [Model-Centric Methods](#Model-Centric) 
  - [Model Compression](#Model-Compression) 
    - [Quantization](#Quantization)
      - [Post-Training Quantitation](#Post-Training-Quantitation)
        - [Weight-based Quantization](#Weight-based-Quantization)
        - [Weight-Activation Co-Quantization](#Weight-Activation-Co-Quantization)
    - [Parameter Pruning](#Parameter-Pruning)
      - [Structured Pruning](#Structured-Pruning)
      - [Unstructured Pruning](#Unstructured-Pruning)
    - [Low-rank Decomposition](#Low-rank-Decomposition)
    - [Knowledge Distillation](#Knowledge-Distillation)
      - [White-Box KD](#White-Box-KD)
      - [Black-Box KD](#Black-Box-KD)
  - [Efficient Pre-Training](#Efficient-Pre-Training)
    - [Mixed Precision Acceleration](#Mixed-Precision-Acceleration)
    - [Scaling Models](#Scaling-Models)
    - [Initialization Techniques](#Initialization-Techniques)
    - [Optimization Strategies](#Optimization-Strategies)
    - [System-level Pre-Training Acceleration](#System-level-Techniques)
      - [Distributed Pre-Training](#Distributed-Pre-Training)
      - [Hardware-assisted Attention Acceleration](#Hardware-assisted-Attention-Acceleration)
  - [Efficient Fine-Tuning](#Efficient-Fine-Tuning) 
    - [Parameter Efficient Fine-Tuning](#Parameter-Efficient-Fine-Tuning)
      - [Adapter-Tuning](#Adapter-tuning)
      - [Low-Rank Adaptation](#Low-Rank-Adaptation)
      - [Prefix-Tuning](#Prefix-Tuning)
      - [Prompt-Tuning](#Prompt-Tuning)
    - [Memory Efficient Fine-Tuning](#Memory-Efficient-Fine-Tuning)
  - [Efficient Inference](#Efficient-Inference)
    - [Algorithm-level Inference Acceleration](#Algorithm-level-Inference-Acceleration)
      - [Speculative Decoding](#Speculative-Decoding)
      - [KV-cache Optimization](#KV-cache-Optimization)
      - [Sharing-based Attention Acceleration](#Sharing-based-Attention-Acceleration)
    - [System-level Inference Acceleration](#System-level-Inference-Acceleration)
  - [Efficient Architecture](#Efficient-Architecture)
    - [Efficient Attention](#Efficient-Attention)
      - [General Attention Optimization](#General-Attention-Optimization)
      - [Attention Optimization for LLMs](#Attention-Optimization-for-LLMs)
    - [Mixture of Experts](#Mixture-of-Experts)
      - [MoE-based LLMs](#MoE-based-LLMs)
      - [Algorithm-level MoE Optimization](#Algorithm-level-MoE-Optimization)
      - [System-level MoE Acceleration](#System-level-MoE-Acceleration)
    - [Long-context LLMs](#Long-Context-LLMs)
      - [Extrapolation and Interpolation](#Extrapolation-and-Interpolation)
      - [Recurrent Structure](#Recurrent-Structure)
      - [Window & Stream Structure](#Window-&-Stream-Structure)
      - [Memory-Retrieval Augmentation](#Memory-Retrieval-Augmentation)
    - [Transformer Alternative Architecture](#Transformer-Alternative-Architecture)
      - [State Space Models](#State-Space-Models)
      - [Other Sequential Models](#Other-Sequential-Models)
- [Data-Centric Methods](#Data-Centric)
  - [Data Selection](#Data-Selection)
    - [Data Selection for Efficient Pre-Training](#Data-Selection-for-Efficient-Pre-Training)
    - [Data Selection for Efficient Fine-Tuning](#Data-Selection-for-Efficient-Fine-Tuning)
  - [Prompt Engineering](#Prompt-Engineering)
    - [Few-Shot Prompting](#Few-Shot-Prompting)
      - [Demonstration Organization](#Demonstration-Organization)
        - [Demonstration Selection](#Demonstration-Selection)
        - [Demonstration Ordering](#Demonstration-Ordering)
      - [Template Formatting](#Template-Formatting)
        - [Instruction Generation](#Instruction-Generation)
        - [Multi-Step Reasoning](#Multi-Step-Reasoning)
    - [Prompt Compression](#Prompt-Compression)
    - [Prompt Generation](#Prompt-Generation)
- [LLM Frameworks](#LLM-Frameworks)
  - [LLM Frameworks Supporting Efficient Training and Inference](#LLM-Frameworks-Supporting-Efficient-Training-and-Inference)
  - [LLM Frameworks Supporting Efficient Inference Only](#LLM-Frameworks-Supporting-Efficient-Inference-Only)


## Model-Centric
### Model Compression
#### Quantization
##### Post-Training Quantitation
###### Weight-based Quantization
- LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale, <ins>NeurlPS, 2022</ins> [[Paper](https://openreview.net/forum?id=dXiGWqBoxaD)] [[Code](https://doi.org/10.48550/arXiv.2208.07339)]
- GPTQ: Accurate Quantization for Generative Pre-trained Transformers, <ins>ICLR, 2023</ins> [[Paper](https://openreview.net/forum?id=tcbBPnfwxS)] [[Code](https://github.com/IST-DASLab/gptq)]
- Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning, <ins>NeurIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2208.11580)] [[Code](https://github.com/IST-DASLab/OBC)]
- QuIP: 2-Bit Quantization of Large Language Models With Guarantees, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.13304)] [[Code](https://github.com/jerry-chee/QuIP)]
- AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.00978)] [[Code](https://github.com/mit-han-lab/llm-awq)]
- OWQ: Lessons learned from activation outliers for weight quantization in large language models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.02272)]
- SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/pdf/2306.03078)] [[Code](https://github.com/Vahe1994/SpQR)]
- FineQuant: Unlocking Efficiency with Fine-Grained Weight-Only Quantization for LLMs, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.09723)]
###### Weight-Activation Co-Quantization
- ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.09782)]
- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models, <ins>NeurlPS-ENLSP, 2022 </ins>[[Paper](https://arxiv.org/abs/2211.10438)] [[Code](https://github.com/mit-han-lab/smoothquant)]
- OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2304.07493)]
- RPTQ: Reorder-based Post-training Quantization for Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2304.01089)]
- Outlier Suppression+: Accurate quantization of large language models by equivalent and optimal shifting and scaling, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2304.09145)]
- QLLM: Accurate and Efficient Low-Bitwidth Quantization for Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.08041)]
##### Quantitation-aware Training
- Compression of Generative Pre-trained Language Models via Quantization, <ins>ACL, 2022</ins> [[Paper](https://aclanthology.org/2022.acl-long.331.pdf)]
- LLM-QAT: Data-Free Quantization Aware Training for Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.17888)] 
- BitNet: Scaling 1-bit Transformers for Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.11453)]
##### Fine-Tuning Enhanced Quantization
- QLoRA: Efficient Finetuning of Quantized LLMs, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.14314)]
- Memory-Efficient Fine-Tuning of Compressed Large Language Models via sub-4-bit Integer Quantization, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.14152)]
- QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2309.14717 )]
- LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.08659)] [[Code](https://github.com/yxli2123/LoftQ)]
#### Parameter Pruning
##### Structured Pruning
- LLM-Pruner: On the Structural Pruning of Large Language Models, <ins>Github, 2023</ins> [[Paper](https://arxiv.org/abs/2305.11627)] [[Code](https://github.com/horseee/LLM-Pruner)]
- Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning, <ins> Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.06694)] [[Code](https://github.com/princeton-nlp/LLM-Shearing)]
- Accelerated Sparse Neural Training: A Provable and Efficient Method to Find N:M Transposable Masks, <ins>NeurIPS, 2021</ins> [[Paper](https://proceedings.neurips.cc/paper/2021/file/b0490b85e92b64dbb5db76bf8fca6a82-Paper.pdf)] [[Code](https://github.com/papers-submission/structured_transposable_masks)]
- Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning, <ins>Arxiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2305.18403)]
##### Unstructured Pruning
- SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2301.00774)] [[Code](https://github.com/IST-DASLab/sparsegpt)]
- A Simple and Effective Pruning Approach for Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.11695)] [[Code](https://github.com/locuslab/wanda)]
- One-Shot Sensitivity-Aware Mixed Sparsity Pruning for Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/pdf/2310.09499v1.pdf)]
#### Low-rank Decomposition
- TensorGPT: Efficient Compression of the Embedding Layer in LLMs based on the Tensor-Train Decomposition, <ins>Arxiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2307.00526)]
- ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2303.08302)]
- LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation, <ins>ICML, 2023</ins>  [[Paper](https://arxiv.org/abs/2306.11222)] [[Code](https://github.com/yxli2123/LoSparse)]
#### Knowledge Distillation
##### White-Box KD
- Towards the Law of Capacity Gap in Distilling Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2311.07052)] [[Code](https://github.com/GeneZC/MiniMA)]
- Baby Llama: knowledge distillation from an ensemble of teachers trained on a small dataset with no performance penalty, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.02019)]
- Knowledge Distillation of Large Language Models <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.08543)] [[Code](https://github.com/microsoft/LMOps/tree/main/minillm)]
- GKD: Generalized Knowledge Distillation for Auto-regressive Sequence Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.13649)]
- Propagating Knowledge Updates to LMs Through Distillation, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.09306)] [[Code](https://github.com/shankarp8/knowledge_distillation)]
- Less is More: Task-aware Layer-wise Distillation for Language Model Compression, <ins>ICML, 2023</ins> [[Paper](https://arxiv.org/pdf/2210.01351.pdf)]
- Compression of Generative Pre-trained Language Models via Quantization, <ins>ACL, 2022</ins> [[Paper](https://aclanthology.org/2022.acl-long.331.pdf)]
- Token-Scaled Logit Distillation for Ternary Weight Generative Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.06744)]
##### Black-Box KD
- Instruction Tuning with GPT-4 <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2304.03277)] [[Code](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)]
- Meta-learning via Language Model In-context Tuning, <ins>Arxiv, 2021</ins>, [[Paper](https://arxiv.org/abs/2110.07814)] [[Code](https://github.com/yandachen/In-context-Tuning)]
- MetaICL: Learning to Learn In Context, <ins>Arxiv, 2021</ins>, [[Paper](https://arxiv.org/abs/2110.15943)] [[Code](https://github.com/facebookresearch/MetaICL)]
- In-context Learning Distillation: Transferring Few-shot Learning Ability of Pre-trained Language Models, <ins>Arxiv, 2022</ins> [[Paper](https://arxiv.org/abs/2212.10670)]
- Explanations from Large Language Models Make Small Reasoners Better, <ins>Arxiv, 2022</ins> [[Paper](https://arxiv.org/abs/2210.06726)]
- Lion: Adversarial Distillation of Closed-Source Large Language Model, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.12870)] [[Code](https://github.com/YJiangcm/Lion)]
- DISCO: Distilling Counterfactuals with Large Language Models, <ins>Arxiv, 2022</ins> [[Paper](https://arxiv.org/abs/2212.10534)] [[Code](https://github.com/eric11eca/disco)]
- Specializing Smaller Language Models towards Multi-Step Reasoning, <ins>ICML, 2023</ins> [[Paper](https://aclanthology.org/2022.findings-naacl.169.pdf)] [[Code](https://github.com/FranxYao/FlanT5-CoT-Specialization)]
- Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes, <ins>ACL, 2023</ins> [[Paper](https://arxiv.org/abs/2305.02301)]
- Large Language Models Are Reasoning Teachers, <ins>ACL, 2023</ins> [[Paper](https://arxiv.org/abs/2212.10071)] [[Code](https://github.com/itsnamgyu/reasoning-teacher)]
- SCOTT: Self-Consistent Chain-of-Thought Distillation, <ins>ACL’23, 2023</ins> [[Paper](https://arxiv.org/abs/2305.01879)] [[Code](https://github.com/wangpf3/consistent-CoT-distillation)]
- Symbolic Chain-of-Thought Distillation: Small Models Can Also "Think" Step-by-Step, <ins>ACL, 2023</ins> [[Paper](https://arxiv.org/abs/2306.14050)]
- Distilling Reasoning Capabilities into Smaller Language Models, <ins>ACL’23, 2023</ins> [[Paper](https://aclanthology.org/2023.findings-acl.441/)] [[Code](https://github.com/kumar-shridhar/Distiiling-LM)]
### Efficient Pre-Training
#### Mixed Precision Acceleration
- Mixed precision training, <ins>Arxiv, 2017</ins> [[Paper](https://arxiv.org/abs/1710.03740)]
- Bfloat16 Processing for Neural Networks, <ins>ARITH, 2019</ins> [[Paper](https://ieeexplore.ieee.org/document/8877390)]
- A study of BFLOAT16 for deep learning training, <ins>Arxiv, 2019</ins> [[Paper](https://arxiv.org/abs/1905.12322)]
- GACT: Activation compressed training for generic network architectures, <ins>ICML, 2022</ins> [[Paper](https://arxiv.org/abs/2206.11357)] [[Code](https://github.com/LiuXiaoxuanPKU/GACT-ICML)]
- Mesa: A memory-saving training framework for transformers, <ins>Arxiv, 2021</ins> [[Paper](https://arxiv.org/abs/2111.11124)] [[Code](https://github.com/ziplab/Mesa)]
#### Scaling Models
- Efficient Training of BERT by Progressively Stacking <ins>ICML, 2019</ins> [[Paper](https://proceedings.mlr.press/v97/gong19a/gong19a.pdf)] [[Code](https://github.com/gonglinyuan/StackingBERT)]
- Progressively Stacking 2.0: A Multi-stage Layerwise Training Method for BERT Training Speedup, <ins>Arxiv, 2020</ins> [[Paper](https://arxiv.org/abs/2011.13635)]
- Reusing Pretrained Models by Multi-linear Operators for Efficient Training, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/pdf/2310.10699v1.pdf)]
- On the Transformer Growth for Progressive BERT Training, <ins>Arxiv, 2020</ins> [[Paper](https://arxiv.org/abs/2010.12562)] [[Code](https://github.com/google-research/google-research/tree/master/grow_bert)]
- Knowledge Inheritance for Pre-trained Language Models, <ins>NAACL, 2022</ins> [[Paper](https://arxiv.org/abs/2105.13880)] [[Code](https://github.com/thunlp/Knowledge-Inheritance)]
- Staged Training for Transformer Language Models, <ins>ICML, 2022</ins> [[Paper](https://proceedings.mlr.press/v162/shen22f/shen22f.pdf)] [[Code](https://github.com/allenai/staged-training)]
- bert2BERT: Towards Reusable Pretrained Language Models, <ins>Arxiv, 2021</ins> [[Paper](https://arxiv.org/abs/2110.07143)]
- Learning to Grow Pretrained Models for Efficient Transformer Training, <ins>ICLR, 2023</ins> [[Paper](https://openreview.net/pdf?id=cDYRS5iZ16f)] [[Code](https://github.com/VITA-Group/LiGO)]
- 2x Faster Language Model Pre-training via Masked Structural Growth, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.02869)]
- FLM-101B: An Open LLM and How to Train It with $100 K Budget, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/pdf/2309.03852.pdf)] [[Code](https://huggingface.co/CofeAI/FLM-101B)]
#### Initialization Techniques
- On weight initialization in deep neural networks, <ins>Arxiv, 2017</ins> [[Paper](https://arxiv.org/abs/1704.08863)] [[Code](https://github.com/sidkk86/weight_initialization)]
- Delving deep into rectifiers: Surpassing humanlevel performance on imagenet classification, <ins>ICCV, 2015</ins> [[Paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf)]
- Fixup initialization: Residual learning without normalization, <ins>ICLR, 2019</ins> [[Paper](https://arxiv.org/abs/1901.09321)]
- ZerO initialization: Initializing neural networks with only zeros and ones, <ins>TMLR, 2022</ins> [[Paper](https://openreview.net/pdf?id=1AxQpKmiTc)] [[Code](https://github.com/jiaweizzhao/ZerO-initialization)]
- Batch normalization biases residual blocks towards the identity function in deep networks, <ins>NeurIPS 2020</ins> [[Paper](https://papers.neurips.cc/paper/2020/file/e6b738eca0e6792ba8a9cbcba6c1881d-Paper.pdf)]
- Rezero is all you need: Fast convergence at large depth, <ins>UAI, 2021</ins> [[Paper](https://proceedings.mlr.press/v161/bachlechner21a/bachlechner21a.pdf)] [[Code](https://github.com/majumderb/rezero)]
- Improving Transformer Optimization Through Better Initialization, <ins>ICML, 2020</ins> [[Paper](https://proceedings.mlr.press/v119/huang20f/huang20f.pdf)] [[Code](https://github.com/layer6ai-labs/T-Fixup)]
- Deepnet: Scaling transformers to 1,000 layers, <ins>Arxiv, 2022</ins> [[Paper](https://arxiv.org/abs/2203.00555)] [[Code](https://github.com/microsoft/unilm)]
#### Optimization Strategies
- Adam: A method for stochastic optimization, <ins>ICLR, 2015</ins> [[Paper](https://arxiv.org/abs/1412.6980)]
- Decoupled weight decay regularization, <ins>ICLR, 2019</ins> [[Paper](https://arxiv.org/abs/1711.05101)] [[Code](https://github.com/loshchil/AdamW-and-SGDW)]
- Symbolic Discovery of Optimization Algorithms, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2302.06675)]
- Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.14342)] [[Code](https://github.com/Liuhong99/Sophia)]
#### System-level Techniques  
##### Distributed Pre-Training
- PyTorch Distributed: Experiences on Accelerating Data Parallel Training
- Measuring the Effects of Data Parallelism on Neural Network Training
- PipeDream: Fast and Efficient Pipeline Parallel DNN Training
- GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism, <ins>NeurIPS, 2018</ins> 
- Maximizing Parallelism in Distributed Training for Huge Neural Networks  
- Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM, <ins>SC'21, 2021</ins>
- Tesseract: Parallelize the Tensor Parallelism Efficiently, <ins>ICPP, 2022</ins>, [[Paper](https://arxiv.org/abs/2105.14500)]
- An Efficient 2D Method for Training Super-Large Deep Learning Models, <ins>IPDPS, 2023</ins> [[Paper](https://arxiv.org/abs/2104.05343)] [[Code](https://github.com/xuqifan897/Optimus)]
- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models, <ins>SC'20, 2020</ins> [[Paper](https://arxiv.org/abs/1910.02054)]
- PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel, <ins>Proc. VLDB Endow, 2023</ins> [[Paper](https://dl.acm.org/doi/10.14778/3611540.3611569)]
- ZeRO-Offload: Democratizing Billion-Scale Model Training  
##### Hardware-assisted Attention Acceleration
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, <ins>NeurIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2205.14135)] [[Code](https://github.com/Dao-AILab/flash-attention)]
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.08691)] [[Code](https://github.com/Dao-AILab/flash-attention)]
### Efficient Fine-tuning
#### Parameter-Efficient Fine-Tuning
##### Adapter-Tuning
- LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/pdf/2304.01933.pdf)] [[Code](https://github.com/AGI-Edgerunners/LLM-Adapters)]
- Compacter: Efficient Low-Rank Hypercomplex Adapter Layers, <ins>NeurlPS, 2023</ins> [[Paper](https://openreview.net/forum?id=bqGK5PyI6-N)] [[Code](https://github.com/rabeehk/compacter)]
- Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning, <ins>NeurlPS, 2022</ins> [[Paper](https://openreview.net/forum?id=rBCvMG-JsPd)] [[Code](https://github.com/r-three/t-few)]
- Meta-Adapters: Parameter Efficient Few-shot Fine-tuning through Meta-Learning, <ins>AutoML, 2022</ins> [[Paper](https://openreview.net/forum?id=BCGNf-prLg5)]
- AdaMix: Mixture-of-Adaptations for Parameter-efficient Model Tuning, <ins>ACL, 2022</ins> [[Paper](https://aclanthology.org/2022.emnlp-main.388/)] [[Code](https://github.com/microsoft/AdaMix)]
- OpenDelta: A Plug-and-play Library for Parameter-efficient Adaptation of Pre-trained Models, <ins>ACL Demo, 2023</ins> [[Paper](https://arxiv.org/abs/2307.03084)] [[Code](https://github.com/thunlp/OpenDelta)]
##### Low-Rank Adaptation
- LoRA: Low-Rank Adaptation of Large Language Models, <ins>ICLR, 2022</ins> [[Paper](https://openreview.net/forum?id=nZeVKeeFYf9)] [[Code](https://github.com/microsoft/LoRA)]
- LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models Fine-tuning, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.03303)]
- LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.13269)] [[Code](https://github.com/sail-sg/lorahub)]
- LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2309.12307)] [[Code](https://github.com/dvlab-research/LongLoRA)]
- Multi-Head Adapter Routing for Cross-Task Generalization, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2211.03831)] [[Code](https://github.com/microsoft/mttl)]
- Parameter-efficient Fine-tuning Design Spaces, <ins>ICLR, 2023</ins> [[Paper](https://openreview.net/forum?id=XSRSWxyJIC)] [[Code](https://github.com/amazon-science/peft-design-spaces)]
- Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning, <ins>ICLR, 2023</ins> [[Paper](https://arxiv.org/pdf/2303.10512)] 
- DyLoRA: Parameter-Efficient Tuning of Pretrained Models using Dynamic Search-Free Low Rank Adaptation, <ins>EACL, 2023</ins> [[Paper](https://aclanthology.org/2023.eacl-main.239/)] [[Code](https://github.com/huawei-noah/KD-NLP/tree/main/DyLoRA)]
- CPET: Effective Parameter-Efficient Tuning for Compressed Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.07705)]
- Tied-Lora: Enhacing parameter efficiency of LoRA with weight tying, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2311.09578)]
##### Prefix-Tuning
- Prefix-Tuning: Optimizing Continuous Prompts for Generation <ins>ACL-IJCNLP, 2021</ins> [[Paper](https://arxiv.org/abs/2101.00190)] [[Code](https://github.com/XiangLi1999/PrefixTuning)]
- P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks <ins>ACL, 2022</ins> [[Paper](https://aclanthology.org/2022.acl-short.8/)] [[Code](https://github.com/THUDM/P-tuning-v2)]
- LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention, <ins>Arxiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2303.16199)] [[Code](https://github.com/ZrrSkywalker/LLaMA-Adapter)]
##### Prompt-Tuning
- The Power of Scale for Parameter-Efficient Prompt Tuning, <ins>EMNLP, 2021</ins> [[Paper](https://arxiv.org/abs/2104.08691)]
- GPT Understands, Too, <ins>AIOPEN, 2023</ins> [[Paper](https://arxiv.org/abs/2103.10385)]
- Multitask Pre-training of Modular Prompt for Chinese Few-Shot Learning <ins>ACL, 2023</ins> [[Paper](https://arxiv.org/abs/2210.07565)] [[Code](https://github.com/Hzfinfdu/MPMP)]
- PPT: Pre-trained Prompt Tuning for Few-shot Learning, <ins>ACL, 2022</ins> [[Paper](https://arxiv.org/abs/2109.04332)] [[Code](https://github.com/thu-coai/PPT)]
- Multitask Prompt Tuning Enables Parameter-Efficient Transfer Learning, <ins>ICLR, 2023</ins> [[Paper](https://arxiv.org/abs/2303.02861)]
#### Memory-Efficient Fine-Tuning
- Memory-Efficient Selective Fine-Tuning, <ins>ICML Workshop, 2023</ins> [[Paper](https://openreview.net/forum?id=zaNbLceVwm)]
- CocktailSGD: Fine-tuning Foundation Models over 500Mbps Networks, <ins>ICML, 2023</ins> [[Paper](https://openreview.net/forum?id=w2Vrl0zlzA)]
- Full Parameter Fine-tuning for Large Language Models with Limited Resources, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.09782)] [[Code](https://github.com/OpenLMLab/LOMO)]
- Fine-Tuning Language Models with Just Forward Passes, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.17333)] [[Code](https://github.com/princeton-nlp/MeZO)]
### Efficient Inference
#### Algorithm-level Inference Acceleration
##### Speculative Decoding
- Fast Inference from Transformers via Speculative Decoding, <ins>ICML, 2023</ins> [[Paper](https://arxiv.org/abs/2211.17192)]
- Accelerating LLM Inference with Staged Speculative Decoding, <ins>ES-FOMO at ICML, 2023</ins> [[Paper](https://arxiv.org/abs/2308.04623)]
- Accelerating Large Language Model Decoding with Speculative Sampling, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2302.01318)]
- Speculative Decoding with Big Little Decoder, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2302.07863)] [[Code](https://github.com/kssteven418/BigLittleDecoder)]
- SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification, <ins>Arxiv, 2023</ins>  [[Paper](https://doi.org/10.48550/arXiv.2305.09781)] [[Code](https://github.com/flexflow/FlexFlow/tree/inference)]
- Inference with Reference: Lossless Acceleration of Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2304.04487)] [[Code](https://github.com/microsoft/unilm)]
##### KV-cache Optimization
- SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.02628)]
- H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models, <ins>ICML Workshop, 2023</ins> [[Paper](https://arxiv.org/abs/2306.14048)]
- Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.17118)]
- Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.15805)]
##### Sharing-based Attention Acceleration
- Fast Transformer Decoding: One Write-Head is All You Need, <ins>Arxiv, 2019</ins> [[Paper](https://arxiv.org/abs/1911.02150)]
- GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, <ins>EMNLP, 2023</ins> [[Paper](https://arxiv.org/abs/2305.13245)]
#### System-level Inference Acceleration
- FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU, <ins>ICML, 2023</ins> [[Paper](https://arxiv.org/abs/2303.06865)] [[Code](https://github.com/FMInference/FlexGen)]
- Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time, <ins>ICML, 2023</ins> [[Paper](https://www.andrew.cmu.edu/user/beidic/)]
- Efficiently Scaling Transformer Inference, <ins>MLSys, 2023</ins> [[Paper](https://proceedings.mlsys.org/paper_files/paper/2023/file/523f87e9d08e6071a3bbd150e6da40fb-Paper-mlsys2023.pdf)]
- EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models, <ins>Arxiv, 2022</ins> [[Paper](https://arxiv.org/abs/2211.05102)]
- S3: Increasing GPU Utilization during Generative Inference for Higher Throughput, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.06000)]
- Efficient Memory Management for Large Language Model Serving with PagedAttention, <ins>SOSP, 2023</ins> [[Paper](https://dl.acm.org/doi/abs/10.1145/3600006.3613165)] [[Code](https://github.com/vllm-project/vllm)]
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, <ins>NeurIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2205.14135)] [[Code](https://github.com/Dao-AILab/flash-attention)]
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.08691)] [[Code](https://github.com/Dao-AILab/flash-attention)]
- DeepSpeed-inference: enabling efficient inference of transformer models at unprecedented scale, <ins>SC, 2022</ins> [[Paper](https://dl.acm.org/doi/abs/10.5555/3571885.3571946)]
- Orca: A Distributed Serving System for Transformer-Based Generative Models, <ins>OSDI, 2022</ins> [[Paper](https://www.usenix.org/conference/osdi22/presentation/yu)]
- Just-in-Time Dynamic-Batchin, <ins>NeurIPS Systems for ML Workshop, 2018</ins> [[Paper](https://arxiv.org/abs/1904.07421)]
- SMDP-Based Dynamic Batching for Efficient Inference on GPU-Based Platforms, <ins>ICC, 2023</ins> [[Paper](https://arxiv.org/abs/2301.12865)]
- Flash-Decoding for long-context inference, <ins>PyTorch, 2023</ins> [[Blog](https://pytorch.org/blog/flash-decoding/)]
- FlashDecoding++: Faster Large Language Model Inference on GPUs, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2311.01282)]
### Efficient Architecture
#### Efficient Attention
##### General Attention Optimization
###### Feature Information Reduction
- Funnel-transformer: Filtering out sequential redundancy for efficient language processing, <ins>Arxiv, 2020</ins> [[Paper](https://arxiv.org/abs/2006.03236)] [[Code](https://github.com/laiguokun/Funnel-Transformer)]
- Nyströmformer: A nyström-based algorithm for approximating self-attention, <ins>AAAI, 2021</ins> [[Paper](https://arxiv.org/abs/2102.03902)] [[Code](https://github.com/mlpen/Nystromformer)]
- Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks, <ins>ICML, 2019</ins> [[Paper](https://arxiv.org/abs/1810.00825)]
###### Kernelization or Low-Rank
- Sumformer: Universal Approximation for Efficient Transformers, <ins>ICML Workshop, 2023</ins> [[Paper](https://arxiv.org/abs/2307.02301)]
- FLuRKA: Fast fused Low-Rank & Kernel Attention, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.15799)]
- Scatterbrain: Unifying Sparse and Low-rank Attention,  <ins>NeurlPS, 2021</ins> [[Paper](https://openreview.net/forum?id=SehIKudiIo1)] [[Code](https://github.com/HazyResearch/fly)]
- Linformer: Self-Attention with Linear Complexity, <ins>Arxiv, 2020</ins> [[Paper](https://arxiv.org/abs/2006.04768)] [[Code](https://github.com/lucidrains/linformer)]
- Lightweight and Efficient End-to-End Speech Recognition Using Low-Rank Transformer, <ins>ICASSP, 2020</ins> [[Paper](https://arxiv.org/abs/1910.13923)]
- Rethinking Attention with Performers,  <ins>ICLR, 2021</ins> [[Paper](https://openreview.net/forum?id=Ua6zuk0WRH)] [[Code](https://github.com/lucidrains/performer-pytorch)]
- Random Feature Attention, <ins>ICLR, 2021</ins> [[Paper](https://arxiv.org/abs/2103.02143)]
- Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention, <ins>ICML, 2020</ins> [[Paper](https://arxiv.org/abs/2006.16236)] [[Code](https://github.com/idiap/fast-transformers)]
###### Fixed Pattern Strategies
- Big bird: Transformers for longer sequences, <ins>NeurIPS, 2020</ins> [[Paper](https://arxiv.org/abs/2007.14062)] [[Code](https://github.com/google-research/bigbird)]
- Poolingformer: Long Document Modeling with Pooling Attention, <ins>ICML, 2021</ins> [[Paper](https://arxiv.org/abs/2105.04371)]
- Longformer: The Long-Document Transformer, <ins>Arxiv, 2020</ins> [[Paper](https://arxiv.org/abs/2004.05150)] [[Code](https://github.com/allenai/longformer)]
- Blockwise Self-Attention for Long Document Understanding, <ins>EMNLP, 2020</ins> [[Paper](https://arxiv.org/abs/1911.02972v)] [[Code](https://github.com/xptree/BlockBERT)]
- Generating Long Sequences with Sparse Transformers, <ins>Arxiv, 2019</ins> [[Paper](https://arxiv.org/abs/1904.10509)]
- Faster Causal Attention Over Large Sequences Through Sparse Flash Attention, <ins>ICML Workshop, 2023</ins> [[Paper](https://arxiv.org/abs/2306.01160)]
###### Learnable Pattern Strategies
- Reformer: The Efficient Transformer,  <ins>ICLR, 2022</ins> [[Paper](https://openreview.net/forum?id=rkgNKkHtvB)] [[Code](https://github.com/lucidrains/reformer-pytorch)]
- Sparse Sinkhorn Attention, <ins>ICML, 2020</ins> [[Paper](https://arxiv.org/abs/2002.11296)]
- Fast Transformers with Clustered Attention, <ins>NeurIPS, 2020</ins> [[Paper](https://arxiv.org/pdf/2007.04825.pdf)] [[Code](https://github.com/idiap/fast-transformers)]
- ClusterFormer: Neural Clustering Attention for Efficient and Effective Transformer, <ins>ACL, 2022</ins> [[Paper](https://aclanthology.org/2022.acl-long.170/)]
- Efficient Content-Based Sparse Attention with Routing Transformers, <ins>TACL, 2020</ins> [[Paper](https://arxiv.org/abs/2003.05997)] [[Code](https://github.com/google-research/google-research/tree/master/routing_transformer)]
###### Hardware-assisted Attention
- A3: Accelerating Attention Mechanisms in Neural Networks with Approximation, <ins>HPCA, 2020</ins> [[Paper](https://arxiv.org/abs/2002.10941)]
- Efficient Memory Management for Large Language Model Serving with PagedAttention, <ins>SOSP, 2023</ins> [[Paper](https://dl.acm.org/doi/abs/10.1145/3600006.3613165)] [[Code](https://github.com/vllm-project/vllm)]
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, <ins>NeurIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2205.14135)] [[Code](https://github.com/Dao-AILab/flash-attention)]
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.08691)] [[Code](https://github.com/Dao-AILab/flash-attention)]
##### Attention Optimization for LLMs
- Fast Transformer Decoding: One Write-Head is All You Need, <ins>Arxiv, 2019</ins> [[Paper](https://arxiv.org/abs/1911.02150)]
- GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, <ins>EMNLP, 2023</ins> [[Paper](https://arxiv.org/abs/2305.13245)]
- Generating Long Sequences with Sparse Transformers, <ins>Arxiv, 2019</ins> [[Paper](https://arxiv.org/abs/1904.10509)]
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, <ins>NeurIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2205.14135)] [[Code](https://github.com/Dao-AILab/flash-attention)]
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.08691)] [[Code](https://github.com/Dao-AILab/flash-attention)]
- Accelerated Inference for Large Transformer Models Using NVIDIA Triton Inference Server, <ins>Nvidia Blog, 2022</ins> [[Blog](https://developer.nvidia.com/blog/accelerated-inference-for-large-transformer-models-using-nvidia-fastertransformer-and-nvidia-triton-inference-server/)]
#### Mixture of Experts
##### MoE-based LLMs
- GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding, <ins>Arxiv, 2020</ins> [[Paper](https://arxiv.org/abs/2006.16668)]
- Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity, <ins>JMLR, 2022</ins> [[Paper](https://jmlr.org/papers/volume23/21-0998/21-0998.pdf)] [[Code](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py)]
- Efficient Large Scale Language Modeling with Mixtures of Experts, <ins>EMNLP, 2022</ins> [[Paper](https://arxiv.org/abs/2112.10684)] [[Code](https://github.com/facebookresearch/fairseq/tree/main/examples/moe_lm)]
- BASE Layers: Simplifying Training of Large, Sparse Models, <ins>ICML, 2021</ins> [[Paper](https://arxiv.org/abs/2103.16716)] [[Code](https://github.com/pytorch/fairseq/)]
- PanGu-Σ: Towards Trillion Parameter Language Model with Sparse Heterogeneous Computing, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2303.10845)]
##### Algorithm-level MoE Optimization
- Mixture-of-Experts with Expert Choice Routing, <ins>NeurIPS, 2022</ins> [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/2f00ecd787b432c1d36f3de9800728eb-Paper-Conference.pdf)]
- StableMoE: Stable Routing Strategy for Mixture of Experts, <ins>Arxiv, 2022</ins> [[Paper](https://arxiv.org/pdf/2204.08396.pdf)] [[Code](https://github.com/Hunter-DDM/stablemoe)]
- On the Representation Collapse of Sparse Mixture of Experts, <ins>NeurIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2204.09179)]
- TA-MoE: Topology-Aware Large Scale Mixture-of-Expert Training, <ins>NeurIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2302.09915)] [[Code](https://github.com/Chen-Chang/TA-MoE)]
- Lifelong Language Pretraining with Distribution-Specialized Experts, <ins>ICML, 2023</ins> [[Paper](https://arxiv.org/abs/2305.12281)]
- Mixture-of-Experts Meets Instruction Tuning:A Winning Combination for Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.14705)]
##### System-level MoE Acceleration
- FastMoE: A Fast Mixture-of-Expert Training System, <ins>PPoPP, 2022</ins> [[Paper](https://arxiv.org/abs/2103.13262)] [[Code](https://github.com/laekov/fastmoe)]
- FasterMoE: modeling and optimizing training of large-scale dynamic pre-trained models, <ins>PPoPP, 2022</ins> [[Paper](https://dl.acm.org/doi/abs/10.1145/3503221.3508418)] [[Code](https://github.com/thu-pacman/FasterMoE)]
- DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale, <ins>ICML, 2022</ins> [[Paper](https://arxiv.org/pdf/2201.05596.pdf)] [[Code](https://github.com/microsoft/DeepSpeed)]
- Tutel: Adaptive mixture-of-experts at scale, <ins>MLSys, 2023</ins> [[Paper](https://arxiv.org/pdf/2206.03382.pdf)] [[Code](https://github.com/microsoft/tutel)]
- SmartMoE: Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization, <ins>USENIX ATC, 2023</ins> [[Paper](https://www.usenix.org/conference/atc23/presentation/zhai)]
#### Long-Context LLMs
##### Extrapolation and Interpolation
- Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation, <ins>ICLR, 2022</ins> [[Paper](https://arxiv.org/pdf/2108.12409.pdf)] [[Code](https://github.com/ofirpress/attention_with_linear_biases)]
- A Length-Extrapolatable Transformer, <ins>ACL, 2023</ins> [[Paper](https://aclanthology.org/2023.acl-long.816/)] [[Code](https://aka.ms/LeX-Transformer)]
- Extending Context Window of Large Language Models via Positional Interpolation, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.15595)]
- NTK interpolation, [[Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)]
- YaRN: Efficient Context Window Extension of Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2309.00071)] [[Code](https://github.com/jquesnelle/yarn)]
-  Functional Interpolation for Relative Positions Improves Long Context Transformers, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/pdf/2310.04418.pdf)]
- The EOS Decision and Length Extrapolation, <ins>EMNLP, 2020</ins> [[Paper](https://arxiv.org/abs/2010.07174)] [[Code](https://github.com/bnewm0609/eos-decision)]
- Exploring Length Generalization in Large Language Models, <ins>NeurIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2207.04901)]
- ∞-former: Infinite Memory Transformer, <ins>ACL, 2022</ins> [[Paper](https://arxiv.org/abs/2109.00301)]
##### Recurrent Structure
- Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context, <ins>ACL, 2019</ins> [[Paper](https://arxiv.org/abs/1901.02860)] [[Code](https://github.com/kimiyoung/transformer-xl)]
- Memformer: A Memory-Augmented Transformer for Sequence Modeling, <ins>Arxiv, 2020</ins> [[Paper]](https://arxiv.org/abs/2010.06891) [[Code](https://github.com/deep-spin/infinite-former)]
- Recurrent Memory Transformer, <ins>NeurIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2207.06881)] [[Code](https://github.com/booydar/LM-RMT)]
- Block-Recurrent Transformers, <ins>NeurIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2203.07852)] [[Code](https://github.com/google-research/meliad)]
##### Window & Stream Structure
- Efficient Streaming Language Models with Attention Sinks, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2309.17453)] [[Code](https://github.com/mit-han-lab/streaming-llm)]
- Parallel Context Windows for Large Language Models, <ins>ACL, 2023</ins> [[Paper](https://arxiv.org/abs/2212.10947)] [[Code](https://github.com/ai21labs/parallel-context-windows)]
- Structured Prompting: Scaling In-Context Learning to 1,000 Examples, <ins>Arxiv, 2022</ins> [[Paper](https://arxiv.org/abs/2212.06713)] [[Code](https://github.com/microsoft/LMOps)]
- Naive Bayes-based Context Extension, <ins>Github repository, 2023</ins>  [[Code](https://github.com/bojone/NBCE)]
- LongNet: Scaling Transformers to 1,000,000,000 Tokens, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.02486)] [[Code](https://github.com/microsoft/unilm/tree/master)]
- Efficient Long-Text Understanding with Short-Text Models, <ins>TACL, 2023</ins> [[Paper](https://arxiv.org/abs/2208.00748)] [[Code](https://github.com/Mivg/SLED)]
##### Memory-Retrieval Augmentation
- Memorizing Transformers, <ins>ICLR, 2022</ins> [[Paper](https://arxiv.org/abs/2203.08913)] [[Code]()]
- Landmark Attention: Random-Access Infinite Context Length for Transformers, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.16300)] [[Code](https://github.com/epfml/landmark-attention/)]
- Augmenting Language Models with Long-Term Memory, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.07174)]
- Unlimiformer: Long-Range Transformers with Unlimited Length Input, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2305.01625)] [[Code](https://github.com/abertsch72/unlimiformer)]
- Focused Transformer: Contrastive Training for Context Scaling, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2307.03170)] [[Code](https://github.com/CStanKonrad/long_llama)]
- Retrieval meets Long Context Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.03025)]
#### Transformer Alternative Architecture
##### State Space Models
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2312.00752)] [[Code](https://github.com/state-spaces/mamba)]
- Efficiently Modeling Long Sequences with Structured State Spaces, <ins>ICLR, 2022</ins> [[Paper](https://arxiv.org/abs/2111.00396)] [[Code](https://github.com/state-spaces/s4)]
- Diagonal State Spaces are as Effective as Structured State Spaces, <ins>NeurIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2203.14343)] [[Code](https://github.com/ag1988/dss)]
- Hungry Hungry Hippos: Towards Language Modeling with State Space Models, <ins>ICLR 2023</ins> [[Paper](https://arxiv.org/abs/2212.14052)] [[Code](https://github.com/HazyResearch/H3)]
- Long Range Language Modeling via Gated State Spaces, <ins>ICLR, 2023</ins> [[Paper](https://arxiv.org/abs/2206.13947)]
- Block-State Transformers, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2306.09539)]
##### Other Sequential Models
- RWKV: Reinventing RNNs for the Transformer Era, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.13048)]
- Hyena Hierarchy: Towards Larger Convolutional Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2302.10866)]
- MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/pdf/2305.07185.pdf)]

## Data-Centric
### Data Selection
#### Data Selection for Efficient Pre-Training
- Data Selection Strategies for Multi-Domain Sentiment Analysis, <ins>Arxiv, 2017</ins> [[Paper](https://arxiv.org/abs/1702.02426)]
- Data Selection with Cluster-Based Language Difference Models and Cynical Selection, <ins>IWSLT, 2017</ins> [[Paper](https://arxiv.org/abs/1904.04900)]
- Span Selection Pre-training for Question Answering, <ins>ACL, 2020</ins> [[Paper](https://arxiv.org/abs/1909.04120)] [[Code](https://github.com/IBM/span-selection-pretraining)]
- NLP From Scratch Without Large-Scale Pretraining: A Simple and Efficient Framework, <ins>ICML, 2022</ins> [[Paper](https://arxiv.org/pdf/2111.04130.pdf)] [[Code](https://github.com/yaoxingcheng/TLM)]
- Data Selection for Language Models via Importance Resampling, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2302.03169)] [[Code](https://github.com/p-lambda/dsir)]
#### Data Selection for Efficient Fine-Tuning
- Instruction Mining: When Data Mining Meets Large Language Model Finetuning, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.06290)]
- Data-Efficient Finetuning Using Cross-Task Nearest Neighbors, <ins>ACL, 2023</ins> [[Paper](https://aclanthology.org/2023.findings-acl.576.pdf)] [[Code](https://github.com/allenai/data-efficient-finetuning)]
- Data Selection for Fine-tuning Large Language Models Using Transferred Shapley Values, <ins>ACL SRW, 2023</ins> [[Paper](https://arxiv.org/abs/2306.10165)] [[Code](https://github.com/stephanieschoch/ts-dshapley)]
- Maybe Only 0.5% Data is Needed: A Preliminary Exploration of Low Training Data Instruction Tuning, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.09246)]
- AlpaGasus: Training A Better Alpaca with Fewer Data, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.08701)] [[Code](https://github.com/Lichang-Chen/AlpaGasus)]
- LIMA: Less Is More for Alignment, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.11206)]
### Prompt Engineering
#### Few-Shot Prompting
##### Demonstration Organization
###### Demonstration Selection  
- Self-Adaptive In-Context Learning: An Information Compression Perspective for In-Context Example Selection and Ordering, <ins>ACL, 2023</ins> [[Paper](https://aclanthology.org/2023.acl-long.79.pdf)] [[Code](https://github.com/Shark-NLP/self-adaptive-ICL)]
- What Makes Good In-Context Examples for GPT-3? <ins>DeeLIO, 2022</ins> [[Paper](https://arxiv.org/abs/2101.06804)]
- Selective Annotation Makes Language Models Better Few-Shot Learners, <ins>ICLR, 2023</ins> [[Paper](https://arxiv.org/abs/2209.01975)] [[Code](https://github.com/xlang-ai/icl-selective-annotation)]
- Learning To Retrieve Prompts for In-Context Learning, <ins>NAACL-HLT, 2022</ins> [[Paper](https://arxiv.org/abs/2112.08633)] [[Code](https://github.com/OhadRubin/EPR)]
- Unified Demonstration Retriever for In-Context Learning, <ins>ACL, 2023</ins> [[Paper](https://arxiv.org/abs/2305.04320)] [[Code](https://arxiv.org/abs/2305.04320)]
###### Demonstration Ordering
- Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity, <ins>ACL, 2022</ins> [[Paper](https://arxiv.org/abs/2104.08786)]
- What Makes Good In-Context Examples for GPT-3? <ins>DeeLIO, 2022</ins> [[Paper](https://arxiv.org/abs/2101.06804)]
##### Template Formatting
###### Instruction Generation
- Instruction Induction: From Few Examples to Natural Language Task Descriptions, <ins>ACL, 2023</ins> [[Paper](https://arxiv.org/abs/2205.10782)] [[Code](https://github.com/orhonovich/instruction-induction)]
- Large Language Models Are Human-Level Prompt Engineers, <ins>ICLR, 2023</ins> [[Paper](https://arxiv.org/abs/2211.01910)] [[Code](https://github.com/keirp/automatic_prompt_engineer)]
- Self-Instruct: Aligning Language Models with Self-Generated Instructions, <ins>Arxiv, 2022</ins> [[Paper](https://arxiv.org/abs/2212.10560)] [[Code](https://github.com/yizhongw/self-instruct)]
- Large Language Models as Optimizers, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2309.03409)]
###### Multi-step Reasoning
- Emergent Abilities of Large Language Models, <ins>TMLR, 2022</ins> [[Paper](https://arxiv.org/abs/2206.07682)]
- Automatic Chain of Thought Prompting in Large Language Models, <ins>ICLR, 2023</ins> [[Paper](https://arxiv.org/abs/2210.03493)] [[Code](https://github.com/amazon-science/auto-cot)]
- Measuring and Narrowing the Compositionality Gap in Language Models, <ins>EMNLP, 2023</ins> [[Paper](https://arxiv.org/abs/2210.03350)] [[Code](https://github.com/ofirpress/self-ask)]
- ReAct: Synergizing Reasoning and Acting in Language Models, <ins>ICLR, 2023</ins> [[Paper](https://arxiv.org/abs/2210.03629)] [[Code](https://github.com/ysymyth/ReAct)]
- Least-to-Most Prompting Enables Complex Reasoning in Large Language Models, <ins>ICLR, 2023</ins> [[Paper](https://arxiv.org/abs/2205.10625)]
- Graph of Thoughts: Solving Elaborate Problems with Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.09687)] [[Code](https://github.com/spcl/graph-of-thoughts)]
- Tree of Thoughts: Deliberate Problem Solving with Large Language Models, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2305.10601)] [[Code](https://github.com/princeton-nlp/tree-of-thought-llm)]
- Self-Consistency Improves Chain of Thought Reasoning in Language Models, <ins>ICLR, 2023</ins> [[Paper](https://arxiv.org/abs/2203.11171)]
- Graph of Thoughts: Solving Elaborate Problems with Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.09687)] [[Code](https://github.com/spcl/graph-of-thoughts)]
- Contrastive Chain-of-Thought Prompting, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/pdf/2311.09277.pdf)] [[Code](https://github.com/DAMO-NLP-SG/contrastive-cot)]
- Everything of Thoughts: Defying the Law of Penrose Triangle for Thought Generation, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2311.04254)]
#### Prompt Compression
- Learning to Compress Prompts with Gist Tokens, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2304.08467)]
- Adapting Language Models to Compress Contexts, <ins>EMNLP, 2023</ins> [[Paper](https://arxiv.org/abs/2305.14788)] [[Code](https://github.com/princeton-nlp/AutoCompressors)]
- In-context Autoencoder for Context Compression in a Large Language Model, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.06945)] [[Code](https://github.com/getao/icae)]
- LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.06839)] [[Code](https://github.com/microsoft/LLMLingua)]
- Discrete Prompt Compression with Reinforcement Learning, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.08758)]
- Nugget 2D: Dynamic Contextual Compression for Scaling Decoder-only Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.02409)]
#### Prompt Generation  
- Self-Instruct: Aligning Language Model with Self Generated Instructions, <ins>ACL, 2023</ins> [[paper](https://doi.org/10.48550/arXiv.2212.10560)] [[Code](https://github.com/yizhongw/self-instruct)]
- Tuning Language Models as Training Data Generators for Augmentation-Enhanced Few-Shot Learning, <ins>ICML, 2023</ins> [[Paper](https://arxiv.org/abs/2211.03044)] [[Code](https://github.com/yumeng5/FewGen)]
- Large Language Models Are Human-Level Prompt Engineers, <ins>ICLR, 2023</ins> [[Paper](https://arxiv.org/abs/2211.01910)] [[Code](https://github.com/keirp/automatic_prompt_engineer)]
- TempLM: Distilling Language Models into Template-Based Generators, <ins>Arxiv, 2022</ins> [[Paper](https://arxiv.org/abs/2205.11055)] [[Code](https://github.com/Tiiiger/templm)]
- PromptGen: Automatically Generate Prompts using Generative Models, <ins>Findings-NAACL, 2022</ins> [[Paper](https://aclanthology.org/2022.findings-naacl.3/)]
- AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts, <ins>EMNLP, 2020</ins> [[Paper](https://aclanthology.org/2020.emnlp-main.346.pdf)] [[Code](https://ucinlp.github.io/autoprompt/)]
- TeGit: Generating High-Quality Instruction-Tuning Data with Text-Grounded Task Design, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2309.05447)]

## LLM Frameworks
### LLM Frameworks Supporting Efficient Training and Inference
- DeepSpeed [[Code](https://github.com/microsoft/DeepSpeed)]
- Megatron [[Code](https://github.com/NVIDIA/Megatron-LM)]
- Alpa [[Code](https://github.com/alpa-projects/alpa)]
- ColossalAI [[Code](https://github.com/hpcaitech/ColossalAI)]
- FairScale [[Code](https://github.com/facebookresearch/fairscale)]
- Pax [[Code](https://github.com/google/paxml/)]
- Composer [[Code](https://github.com/mosaicml/composer)]
### LLM Frameworks Supporting Efficient Inference Only
- vLLM [[Code](https://github.com/vllm-project/vllm)]
- Parallelformers [[Code](https://github.com/tunib-ai/parallelformers)]
- OpenLLM [[Code](https://github.com/bentoml/OpenLLM)]
- Ray-LLM [[Code](https://github.com/ray-project/ray-llm)]
- MLC-LLM [[Code](https://github.com/mlc-ai/mlc-llm)]
- Sax [[Code](https://github.com/google/saxml)]
- Mosec [[Code](https://github.com/mosecorg/mosec)]
- LLM-Foundry [[Code](https://github.com/mosaicml/llm-foundry)]

## Citation
If you find this useful for your work, please consider citing:
```
@misc{wan2023efficient,
      title={Efficient Large Language Models: A Survey}, 
      author={Zhongwei Wan and Xin Wang and Che Liu and Samiul Alam and Yu Zheng and Zhongnan Qu and Shen Yan and Yi Zhu and Quanlu Zhang and Mosharaf Chowdhury and Mi Zhang},
      year={2023},
      eprint={2312.03863},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
## ❤️ Contribution

This repository is currently maintained by <ins>[tuidan](https://github.com/tuidan)</ins> [wang.15980@osu.edu] and <ins>[SUSTechBruce](https://github.com/SUSTechBruce)</ins> [wan.512@osu.edu]. **We appreciate any contributions** ❤️.


<!-- **<font color='red'>Work still in progress</font>**  🚀, **we appreciate any suggestions and contributions** ❤️. -->

If you have any suggestions or find any missed papers, feel free to send us an email submit a pull request:

1. Use following markdown format.

```markdown
**Paper Title, <ins>Conference/Journal/Preprint, Year</ins>  [[pdf](link)] [[other resources](link)].
```

2. If one preprint paper has multiple versions, please use **the earliest submitted year**.
   
3. Display the papers in **a year descending order** (the latest, the first).

