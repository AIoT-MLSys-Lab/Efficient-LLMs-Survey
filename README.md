# Efficient LLM: Survey, Papers, Benchmarks, and Open-Source Resources.  

### ToDo (05/10/2023) 
* Keep adding papers into the list.
* Put the papers under their correct categories. Create new categories if the papers do not fit the current ones.
* Add the link of the open source codes (e.g., github repo) if the paper has one.
* Add datasets, benchmarks used by the papers.

## What is Efficient LLM About?

## Survey and Perspective Papers
* Efficient LLM: A Survey.
* Mobile Computing in the Era of LLM: Challenges and Opportunities.

## Table of Content
- [Open LLM](#Open-LLM)
- [Efficient Fine-Tuning](#Efficient-Fine-Tuning)
  - [Data Efficient](#Data-Efficient)
  - [Memory Efficient](#Memory-Efficient)
  - [Parameter Efficient](#Parameter-Efficient)
- [Model Compression](#Model-Compression)
  - [Model Pruning](#Model-Pruning)
  - [Model Quantization](#Model-Quantization)
  - [Low-rank Decomposition](#Low-rank-Decomposition)
  - [Knowledge Distillation](#Knowledge-Distillation)
  - [Efficient Attention](#Efficient-Attention)
- [Efficient Inference](#Efficient-Inference)
- [Efficient Training](#Efficient-Training)
- [Efficient Systems/Library](#Efficient-Systems-Library)

## Open LLM
- Open LLM Leaderboard: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
- Awesome-LLM: https://github.com/Hannibal046/Awesome-LLM
- KoLA: Carefully Benchmarking World Knowledge of Large Language Models, [[Paper](https://paperswithcode.com/paper/kola-carefully-benchmarking-world-knowledge)] [[Code](https://github.com/thu-keg/kola)]
- Benchmarking LLM Inference Efficiency: https://ml.energy/leaderboard/?__theme=light
- awesome-huge-models Awesome: https://github.com/zhengzangw/awesome-huge-models
- LLM Collection: https://www.promptingguide.ai/models/collection
- https://sapling.ai/llm/llama-vs-opt
- https://huggingface.co/spaces/optimum/llm-perf-leaderboard
  
## Data-centric
### Prompt Engineering
#### Few-shot Prompting
- LARGE LANGUAGE MODELS AS OPTIMIZERS
- Hybrid Retrieval-Augmented Generation for Real-time Composition Assistance
- PREFER: Prompt Ensemble Learning via Feedback-Reflect-Refine
- Generate rather than Retrieve: Large Language Models are Strong Context Generators
- TRAC: Trustworthy Retrieval Augmented Chatbot
- Large Language Models Are Human-Level Prompt Engineers
- Self-Generated In-Context Learning: Leveraging Auto-regressive Language Models as a Demonstration Generator
- Compositional Exemplars for In-context Learning
- Larger language models do in-context learning differently
- How Many Demonstrations Do You Need for In-context Learning?
- Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?
- Rethinking the Role of Scale for In-Context Learning: An Interpretability-based Case Study at 66 Billion Scale
- Calibrate Before Use: Improving Few-shot Performance of Language Models
- RAVEN: In-Context Learning with Retrieval Augmented Encoder-Decoder Language Models
- A Survey on In-context Learning
- Large Language Models Are Implicitly Topic Models: Explaining and Finding Good Demonstrations for In-Context Learning
- Learning to Retrieve In-Context Examples for Large Language Models
- Finding Supporting Examples for In-Context Learning
- Unified Demonstration Retriever for In-Context Learning
- In-Context Learning with Many Demonstration Examples
- In-Context Demonstration Selection with Cross Entropy Difference
- RRAML: Reinforced Retrieval Augmented Machine Learning
- Learning To Retrieve Prompts for In-Context Learning
- What Makes Good In-Context Examples for GPT-3?
- Language Models are Few-Shot Learners
#### Prompt Tuning
- Soft Prompt Tuning for Augmenting Dense Retrieval with Large Language Models
- Making Pre-trained Language Models Better Few-shot Learners
- Prompt Tuning Pushes Farther, Contrastive Learning Pulls Closer: A Two-Stage Approach to Mitigate Social Biases
- On Conditional and Compositional Language Model Differentiable Prompting
- Preserving In-Context Learning ability in Large Language Model Fine-tuning
- ThoughtSource: A central hub for large language model reasoning data
- Can Instruction Fine-Tuned Language Models Identify Social Bias through Prompting?
#### Prompt Compression
- Learning to Compress Prompts with Gist Tokens
- Discrete Prompt Compression with Reinforcement Learning
- In-context Autoencoder for Context Compression in a Large Language Model
- Adapting Language Models to Compress Contexts
### Data Preprocessing
- D4: Improving LLM Pretraining via Document De-Duplication and Diversification
- Self-Instruct: Aligning Language Models with Self-Generated Instructions
- Scaling Laws and Interpretability of Learning from Repeated Data
- Deduplicating Training Data Makes Language Models Better
- SOTASTREAM: A Streaming Approach to Machine Translation Training
- Tuning Language Models as Training Data Generators for Augmentation-Enhanced Few-Shot Learning
### Data Selection
#### Training Data Selection
- Data Selection for Language Models via Importance Resampling
- NLP From Scratch Without Large-Scale Pretraining: A Simple and Efficient Framework
- Data Selection with Cluster-Based Language Difference Models and Cynical Selection
- Span Selection Pre-training for Question Answering
#### Fine-tuning Data Selection
- LIMA: Less Is More for Alignment
- InstructionGPT-4: A 200-Instruction Paradigm for Fine-Tuning MiniGPT-4
- Data Selection for Fine-tuning Large Language Models Using Transferred Shapley Values
- Platypus: Quick, Cheap, and Powerful Refinement of LLMs
- AlpaGasus: Training A Better Alpaca with Fewer Data
- Maybe Only 0.5% Data is Needed: A Preliminary Exploration of Low Training Data Instruction Tuning
- Instruction Mining: High-Quality Instruction Data Selection for Large Language Models
- Data-Efficient Finetuning Using Cross-Task Nearest Neighbors



## Efficient Fine-Tuning

#### Data Efficient
- Platypus: Quick, Cheap, and Powerful Refinement of LLMs, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.07317)] [[Code](https://platypus-llm.github.io/)]
- Tuning Language Models as Training Data Generators for Augmentation-Enhanced Few-Shot Learning, <ins>ICML, 2023</ins> [[Paper](https://arxiv.org/abs/2211.03044)] [[Code](https://github.com/yumeng5/FewGen)]
- Maybe Only 0.5% Data is Needed: A Preliminary Exploration of Low Training Data Instruction Tuning, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.09246)]
- AlpaGasus: Training A Better Alpaca with Fewer Data, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.08701)] [[Code](https://lichang-chen.github.io/AlpaGasus/)]
- Towards Robust and Efficient Continual Language Learning, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.05741)]
- LIMA: Less Is More for Alignment, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.11206)]
- Efficient Domain Adaptation of Sentence Embeddings using Adapters, <ins>RANLP, 2023</ins> [[paper](https://arxiv.org/abs/2307.03104)] [[Code](https://github.com/sebischair/Efficient-Domain-Adaptation-of-Sentence-Embeddings-using-Adapters)]
- Data-Efficient Finetuning Using Cross-Task Nearest Neighbors, <ins>ACL, 2023</ins> [[paper](https://doi.org/10.48550/arXiv.2212.00196)][[Code](https://github.com/allenai/data-efficient-finetuning)]
- Self-Instruct: Aligning Language Model with Self Generated Instructions, <ins>ACL, 2023</ins> [[paper](https://doi.org/10.48550/arXiv.2212.10560)] [[Code](https://github.com/yizhongw/self-instruct)]
- Instruction Mining: High-Quality Instruction Data Selection for Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](
https://arxiv.org/abs/2307.06290)]
- Data Selection for Fine-tuning Large Language Models Using Transferred Shapley Values, <ins>ACL SRW, 2023</ins> [[paper](https://doi.org/10.48550/arXiv.2306.10165)] [[Code](https://github.com/stephanieschoch/ts-dshapley)]

#### Memory Efficient
- Memory-Efficient Selective Fine-Tuning, <ins>ICML Workshop, 2023</ins> [[Paper](https://openreview.net/forum?id=zaNbLceVwm)]
- CocktailSGD: Fine-tuning Foundation Models over 500Mbps Networks, <ins>ICML, 2023</ins> [[Paper](https://openreview.net/forum?id=w2Vrl0zlzA)]
- Full Parameter Fine-tuning for Large Language Models with Limited Resources, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.09782)] [[Code](https://github.com/OpenLMLab/LOMO)]
- Fine-Tuning Language Models with Just Forward Passes, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.17333)] [[Code](https://github.com/princeton-nlp/MeZO)]
- NTK-approximating MLP Fusion for Efficient Language Model Fine-tuning <ins>ICML, 2023</ins> [[Paper](https://arxiv.org/abs/2307.08941)] [[Code](https://github.com/weitianxin/MLP_Fusion)]
- Gradient Sparsification For Masked Fine-Tuning of Transformers <ins>IJCNN, 2023</ins> [[Paper](https://arxiv.org/abs/2307.10098)]
- Full Parameter Fine-tuning for Large Language Models with Limited Resources

#### Parameter Efficient
- Comparison between parameter-efficient techniques and full fine-tuning: A case study on multilingual news article classification, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.07282)]
- LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models Fine-tuning, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.03303)]
- PromptSum: Parameter-Efficient Controllable Abstractive Summarization, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.03117)]
- LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.13269)] [[Code](https://github.com/sail-sg/lorahub)]
- CPET: Effective Parameter-Efficient Tuning for Compressed Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.07705)]
- Flacuna: Unleashing the Problem Solving Power of Vicuna using FLAN Fine-Tuning, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.02053)] [[Code](https://huggingface.co/declare-lab/flacuna-13b-v1.0)]
- OpenDelta: A Plug-and-play Library for Parameter-efficient Adaptation of Pre-trained Models, <ins>ACL Demo, 2023</ins> [[Paper](https://arxiv.org/abs/2307.03084)] [[Code](https://github.com/thunlp/OpenDelta)]
- Multi-Task Pre-Training of Modular Prompt for Few-Shot Learning, <ins>ACL, 2023</ins> [[Paper](https://arxiv.org/abs/2210.07565)] [[Code](https://github.com/Hzfinfdu/MPMP)]
- Multitask Prompt Tuning Enables Parameter-Efficient Transfer Learning, <ins>ICLR, 2023</ins> [[Paper](https://arxiv.org/abs/2303.02861)]
- KnowPrefix-Tuning: A Two-Stage Prefix-Tuning Framework for Knowledge-Grounded Dialogue Generation, <ins>ECML-PKDD, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2306.15430)]
- Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation, <ins>ACL, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2305.16938)] [[Code](https://github.com/uds-lsv/llmft)]
- Composing Parameter-Efficient Modules with Arithmetic Operations, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.14870)] [[Code](https://github.com/SJTU-LIT/PEM_composition)]
- Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning, <ins>ICLR, 2023</ins> [[Paper](https://arxiv.org/pdf/2303.10512)] 
- LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/pdf/2304.01933.pdf)] [[Code](https://github.com/AGI-Edgerunners/LLM-Adapters)]
- Fact: factor-tuning for lightweight adaptation on vision transformer, <ins>AAAI, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2212.03145)] [[Code](https://github.com/JieShibo/PETL-ViT)]
- One-for-All: Generalized LoRA for Parameter-Efficient Fine-tuning, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs//2306.07967)]
- QLORA: Efficient Finetuning of Quantized LLMs, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.14314)]
- Compacter: Efficient Low-Rank Hypercomplex Adapter Layers, <ins>NeurlPS, 2023</ins> [[Paper](https://openreview.net/forum?id=bqGK5PyI6-N)] [[Code](https://github.com/rabeehk/compacter)]
- LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention, <ins>Arxiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2303.16199)] [[Code](https://github.com/ZrrSkywalker/LLaMA-Adapter)]
- DyLoRA: Parameter-Efficient Tuning of Pretrained Models using Dynamic Search-Free Low Rank Adaptation, <ins>EACL, 2023</ins> [[Paper](https://aclanthology.org/2023.eacl-main.239/)] [[Code](https://github.com/huawei-noah/KD-NLP/tree/main/DyLoRA)]
- Parameter-efficient Fine-tuning Design Spaces, <ins>ICLR, 2023</ins> [[Paper](https://openreview.net/forum?id=XSRSWxyJIC)] [[Code](https://github.com/amazon-science/peft-design-spaces)]
- Parameter-efficient Fine-tuning of Large-scale Pre-trained Language Models, <ins>Nature Machine Intelligence, 2023</ins> [[Paper](https://doi.org/10.1038/s42256-023-00626-4)] [[Code](https://github.com/thunlp/OpenDelta)]
- PEFT: State-of-the-art Parameter-Efficient Fine-Tuning (PEFT) methods, <ins>Github, 2022</ins> [[Code](https://github.com/huggingface/peft)]
- LoRA: Low-Rank Adaptation of Large Language Models, <ins>ICLR, 2022</ins> [[Paper](https://openreview.net/forum?id=nZeVKeeFYf9)] [[Code](https://github.com/microsoft/LoRA)]
- Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning, <ins>NeurlPS, 2022</ins> [[Paper](https://openreview.net/forum?id=rBCvMG-JsPd)] [[Code](https://github.com/r-three/t-few)]
- Parameter-Efficient Sparsity for Large Language Models Fine-Tuning, <ins>IJCAI, 2022</ins> [[Paper](https://www.ijcai.org/proceedings/2022/0586.pdf)] [[Code](https://github.com/yuchaoli/PST)]
- Attempt: Parameter-Efficient Multi-task Tuning via Attentional Mixtures of Soft Prompts, <ins>EMNLP, 2022</ins> [[Paper](https://aclanthology.org/2022.emnlp-main.446/)] [[Code](https://github.com/AkariAsai/ATTEMPT)]
- On the Effectiveness of Parameter-Efficient Fine-Tuning, <ins>Arxiv, 2022</ins> [[Paper](https://arxiv.org/pdf/2211.15583.pdf)] [[Code](https://github.com/fuzihaofzh/AnalyzeParameterEfficientFinetune)]
- AdaMix: Mixture-of-Adaptations for Parameter-efficient Model Tuning, <ins>ACL, 2022</ins> [[Paper](https://aclanthology.org/2022.emnlp-main.388/)] [[Code](https://github.com/microsoft/AdaMix)]
- Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning, <ins>NIPS, 2022</ins> [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0cde695b83bd186c1fd456302888454c-Abstract-Conference.html)] [[Code](https://github.com/r-three/t-few)]
- Modular and Parameter-Efficient Fine-Tuning for NLP Models, <ins>EMNLP, 2022</ins>, [[Paper](https://aclanthology.org/2022.emnlp-tutorials.5/)]
- Meta-Adapters: Parameter Efficient Few-shot Fine-tuning through Meta-Learning, <ins>AutoML, 2022</ins> [[Paper](https://openreview.net/forum?id=BCGNf-prLg5)]
- AdaMix: Mixture-of-Adaptations for Parameter-efficient Model Tuning, <ins>EMNLP, 2022</ins> [[Paper](https://arxiv.org/abs/2205.12410)] [[Code](https://github.com/microsoft/AdaMix)]
- Exploring extreme parameter compression for pre-trained language models, <ins>ICLR, 2022</ins> [[Paper](https://openreview.net/forum?id=RftryyYyjiG)] 


## Model Compression
- A Survey on Model Compression for Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.07633)]
#### Model Pruning
- Towards Structured Sparsity in Transformers for Efficient Inference, <ins>ICML Workshop, 2023</ins> [[Paper](https://openreview.net/forum?id=c4m0BkO4OL)]
- The Emergence of Essential Sparsity in Large Pre-trained Models: The Weights that Matter, <ins>Arxiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2306.03805)] [[Code](https://github.com/VITA-Group/essential_sparsity)]
- Low-Rank Prune-And-Factorize for Language Model Compression, <ins>Arxiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2306.14152)]
- Knowledge-preserving Pruning for Pre-trained Language Models without Retraining, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.03449)]
- Constraint-aware and Ranking-distilled Token Pruning for Efficient Transformer Inference, <ins>KDD, 2023</ins> [[Paper](https://arxiv.org/abs/2306.14393)]
- LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation, <ins>ICML, 2023</ins>  [[Paper](https://arxiv.org/abs/2306.11222)] [[Code](https://github.com/yxli2123/LoSparse)]
- A Simple and Effective Pruning Approach for Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.11695)] [[Code](https://github.com/locuslab/wanda)]
- Ten Lessons We Have Learned in the New "Sparseland": A Short Handbook for Sparse Neural Network Researchers, <ins>Arxiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2302.02596)]
- Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning, <ins>Arxiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2305.18403)]
- Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.15805)]
- LLM-Pruner: On the Structural Pruning of Large Language Models, <ins>Github, 2023</ins> [[Paper](https://arxiv.org/abs/2305.11627)] [[Code](https://github.com/horseee/LLM-Pruner)]
- SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2301.00774)] [[Code](https://github.com/IST-DASLab/sparsegpt)]
- ZipLM: Hardware-Aware Structured Pruning of Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2302.04089)]
- Sparsity May Cry: Let Us Fail (Current) Sparse Neural Networks Together! <ins>ICLR, 2022</ins> [[Paper](https://openreview.net/forum?id=J6F3lLg4Kdp)] [[Code](https://github.com/VITA-Group/SMC-Bench)]
- Vision Transformer Slimming: Multi-Dimension Searching in Continuous Optimization Space, <ins>CVPR, 2022</ins> [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chavan_Vision_Transformer_Slimming_Multi-Dimension_Searching_in_Continuous_Optimization_Space_CVPR_2022_paper.pdf)] [[Code](https://github.com/Arnav0400/ViT-Slim)]
- Unified Visual Transformer Compression, <ins>ICLR, 2022</ins> [[Paper](https://openreview.net/forum?id=9jsZiUgkCZP)] [[Code](https://github.com/VITA-Group/UVC)]
- From Dense to Sparse: Contrastive Pruning for Better Pre-trained Language Model Compression, <ins>AAAI, 2022</ins> [[Paper](https://arxiv.org/abs/2112.07198)] [[Code](https://github.com/RunxinXu/ContrastivePruning)]
- Visual Transformer Pruning, <ins>KDDW, 2021</ins> [[Paper](https://arxiv.org/abs/2104.08500)] [[Code](https://github.com/Cydia2018/ViT-cifar10-pruning)]
- Accelerating Sparse Deep Neural Networks, <ins>Arxiv, 2021</ins> [[Paper](https://arxiv.org/pdf/2104.08378.pdf)]
- Learning N:M Fine-grained Structured Sparse Neural Networks From Scratch, <ins>ICLR, 2021</ins> [[Paper](https://arxiv.org/abs/2102.04010)] [[Code](https://github.com/aojunzz/NM-sparsity)]
- To prune, or not to prune: exploring the efficacy of pruning for model compression, <ins>ICLRW, 2018</ins> [[Paper](https://openreview.net/forum?id=S1lN69AT-)] [[Code](https://github.com/IntelLabs/Model-Compression-Research-Package)]


#### Model Quantization
- Data-free quantization aware training for large language models,  <ins>Arxiv, 2023</ins> 
- Q-Diffusion: Quantizing Diffusion Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2302.04304)]
- QuIP: 2-Bit Quantization of Large Language Models With Guarantees, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.13304)]
- The case for 4-bit precision: k-bit Inference Scaling Laws, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2212.09720)]
- Efficiency Pentathlon: A Standardized Arena for Efficiency Evaluation, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.09701)]
- ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.09782)]
- Do Emergent Abilities Exist in Quantized Large Language Models: An Empirical Study, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.08072)] [[Code](https://github.com/rucaibox/quantizedempirical)]
- Self-Distilled Quantization: Achieving High Compression Rates in Transformer-Based Language Models, <ins>ACL, 2023</ins> [[Paper](https://arxiv.org/abs/2307.05972)]
- QIGen: Generating Efficient Kernels for Quantized Inference on Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2307.03738)] [[Code](https://github.com/IST-DASLab/QIGen)]
- SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/pdf/2306.03078)] [[Code](https://github.com/Vahe1994/SpQR)]
- SqueezeLLM: Dense-and-Sparse Quantization, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.07629)]
- Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing <ins>Arxiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2306.12929)]
- AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.00978)] [[Code](https://github.com/mit-han-lab/llm-awq)]
- GPTQ: Accurate Quantization for Generative Pre-trained Transformers, <ins>ICLR, 2023</ins> [[Paper](https://openreview.net/forum?id=tcbBPnfwxS)] [[Code](https://github.com/IST-DASLab/gptq)]
- OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2304.07493)]
- Blockwise Compression of Transformer-based Models without Retraining, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2304.01483)]
- ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers, <ins>NeurlPS, 2022</ins> [[Paper](https://openreview.net/forum?id=f-fVCElZ-G1)]
- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models, <ins>NeurlPS-ENLSP, 2022 </ins>[[Paper](https://arxiv.org/abs/2211.10438)] [[Code](https://github.com/mit-han-lab/smoothquant)]
- GPT3.int8(): 8-bit Matrix Multiplication for Transformers at Scale, <ins>NeurlPS, 2022</ins> [[Paper](https://openreview.net/forum?id=dXiGWqBoxaD)] [[Code](https://doi.org/10.48550/arXiv.2208.07339)]
- Compression of Generative Pre-trained Language Models via Quantization, <ins>ACL, 2022</ins> [[Paper](https://aclanthology.org/2022.acl-long.331.pdf)]
- Towards Efficient Post-training Quantization of Pre-trained Language Models, <ins>NIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2109.15082)]


#### Low-rank Decomposition
- TensorGPT: Efficient Compression of the Embedding Layer in LLMs based on the Tensor-Train Decomposition, <ins>Arxiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2307.00526)]
- Sparse Plus Low Rank Matrix Decomposition: A Discrete Optimization Approach, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2109.12701)] [[Code](https://github.com/NicholasJohnson2020/SparseLowRankSoftware)]
- Compressing Transformers: Features Are Low-Rank, but Weights Are Not!  <ins>AAAI, 2023</ins> [[Paper](https://cs.nju.edu.cn/wujx/paper/AAAI2023_AFM.pdf)]
- Strategies for Applying Low Rank Decomposition to Transformer-Based Models,  <ins>NeurlPS-ENLSP, 2022</ins> [[Paper](https://neurips2022-enlsp.github.io/papers/paper_33.pdf)]
- Numerical Optimizations for Weighted Low-rank Estimation on Language Model,  <ins>EMNLP, 2022</ins> [[Paper](https://aclanthology.org/2022.emnlp-main.91.pdf)] 
- Language Model Compression With Weighted Low-rank Factorization,  <ins>ICLR, 2022</ins> [[Paper](https://openreview.net/pdf?id=uPv9Y3gmAI5)]
- Monarch: Expressive Structured Matrices for Efficient and Accurate Training,  <ins>ICML, 2022</ins> [[Paper](https://proceedings.mlr.press/v162/dao22a/dao22a.pdf)] [[Code](https://github.com/HazyResearch/fly)]
- Compressing Pre-trained Language Models using Progressive Low Rank Decomposition,  <ins>NeurlPS-ENLSP, 2021</ins> [[Paper](https://neurips2021-nlp.github.io/papers/27/CameraReady/Neurips_Workshop_camera_ready.pdf)]
- Kronecker Decomposition for GPT Compression,  <ins>NeurlPS-ENLSP, 2021</ins> [[Paper](https://aclanthology.org/2022.acl-short.24.pdf)]

#### Knowledge Distillation
- Token-Scaled Logit Distillation for Ternary Weight Generative Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.06744)]
- Baby Llama: knowledge distillation from an ensemble of teachers trained on a small dataset with no performance penalty, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.02019)]
- Domain Knowledge Distillation from Large Language Model: An Empirical Study in the Autonomous Driving Domain, <ins>ITSC, 2023</ins> [[Paper](https://arxiv.org/abs/2307.11769)]
- Distilling Large Vision-Language Model with Out-of-Distribution Generalizability, <ins>Arxiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2307.03135)] [[Code](https://github.com/xuanlinli17/large_vlm_distillation_ood)]
- Knowledge Distillation Performs Partial Variance Reduction, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/pdf/2305.17581)]
- Symbolic Chain-of-Thought Distillation: Small Models Can Also "Think" Step-by-Step, <ins>ACL, 2023</ins> [[Paper](https://arxiv.org/abs/2306.14050)]
- GKD: Generalized Knowledge Distillation for Auto-regressive Sequence Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.13649)[
- Less is More: Task-aware Layer-wise Distillation for Language Model Compression, <ins>ICML, 2023</ins> [[Paper](https://arxiv.org/pdf/2210.01351.pdf)]
- Knowledge Distillation of Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.08543)]
- Bridging the Gap between Decision and Logits in Decision-based Knowledge Distillation for Pre-trained Language Models, <ins>ACL, 2021</ins> [[Paper](https://arxiv.org/abs/2306.08909)] [[Code](https://github.com/thunlp-mt/dbkd-plm)]
- Propagating Knowledge Updates to LMs Through Distillation, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.09306)]
- Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes, <ins>ACL, 2023</ins> [[Paper](https://arxiv.org/abs/2305.02301)]
- What Language Reveals about Perception: Distilling Psychophysical Knowledge from Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2302.01308)]
- Specializing Smaller Language Models towards Multi-Step Reasoning, <ins>ICML, 2023</ins> [[Paper](https://aclanthology.org/2022.findings-naacl.169.pdf)] [[Code](https://github.com/FranxYao/FlanT5-CoT-Specialization)]
- Lifting the Curse of Capacity Gap in Distilling Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.12129)] [[Code](https://github.com/genezc/minimoe)]
- KDSTM: Neural Semi-supervised Topic Modeling with Knowledge Distillation, <ins>ICLR, 2022</ins> [[Paper](https://arxiv.org/abs/2307.01878)]
- Improving Neural Topic Models using Knowledge Distillation, <ins>EMNLP, 2022</ins> [[Paper](https://www.aclweb.org/anthology/2020.emnlp-main.137/)] [[Code](https://github.com/ahoho/kd-topic-models)]
- Distilling Multi-Step Reasoning Capabilites of Large Language Models into Smaller Models via Semantic Decompositions,  <ins>Arxiv, 2021</ins> [[Paper](https://arxiv.org/abs/2212.00193)]

#### Efficient Attention
- Fast Causal Attention with Dynamic Sparsity <ins>Arxiv, 2023</ins>[[Paper](https://openreview.net/forum?id=BQEaklwG9P)]
- Sumformer: Universal Approximation for Efficient Transformers, <ins>Arxiv, 2023</ins>[[Paper](https://arxiv.org/abs/2307.02301)]
- FLuRKA: Fast fused Low-Rank & Kernel Attention, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.15799)]
- Awesome-Efficient-Transformers: https://github.com/Edwardlzy/Awesome-Efficient-Transformers
- When to Use Efficient Self Attention? Profiling Text, Speech and Image Transformer Variants, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.08667)] 
- RWKV: Reinventing RNNs for the Transformer Era, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.13048)] 
- HALOS: Hashing Large Output Space for Cheap Inference,  <ins>MLSys, 2022</ins> [[Paper](https://proceedings.mlsys.org/paper/2022/hash/1ff8a7b5dc7a7d1f0ed65aaa29c04b1e-Abstract.html)]
- Reformer: The Efficient Transformer,  <ins>ICLR, 2022</ins> [[Paper](https://openreview.net/forum?id=rkgNKkHtvB)] [[Code](https://github.com/lucidrains/reformer-pytorch)]
- Efficient Transformers: A Survey, <ins>ACM Computing Surveys, 2022</ins> [[Paper](https://dl.acm.org/doi/10.1145/3530811)]
- Hierarchical Transformers Are More Efficient Language Models, <ins>NACCL, 2022</ins> [[Paper](https://aclanthology.org/2022.findings-naacl.117/)] [[Code](https://github.com/lucidrains/hourglass-transformer-pytorch)]
- Rethinking Attention with Performers,  <ins>ICLR, 2021</ins> [[Paper](https://openreview.net/forum?id=Ua6zuk0WRH)] [[Code](https://github.com/lucidrains/performer-pytorch)]
- Scatterbrain: Unifying Sparse and Low-rank Attention,  <ins>NeurlPS, 2021</ins> [[Paper](https://openreview.net/forum?id=SehIKudiIo1)] [[Code](https://github.com/HazyResearch/fly)]
- SMYRF: Efficient Attention using Asymmetric Clustering,  <ins>NeurlPS, 2020</ins> [[Paper](https://dl.acm.org/doi/10.5555/3495724.3496267)] [[Code](https://github.com/giannisdaras/smyrf)]
- Long Range Arena: A Benchmark for Efficient Transformers, <ins>ICLR, 2020</ins> [[Paper](https://openreview.net/forum?id=qVyeW-grC2k)] [[Code](https://github.com/google-research/long-range-arena)]
- Linformer: Self-Attention with Linear Complexity,  <ins>Arxiv, 2020</ins> [[Paper](https://arxiv.org/abs/2006.04768)] [[Code](https://github.com/lucidrains/linformer)]
- A tensorized transformer for language modeling,  <ins>NeurlPS, 2019</ins> [[Paper](https://dl.acm.org/doi/10.5555/3454287.3454487)] [[Code](https://github.com/szhangtju/The-compression-of-Transformer)]
- Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations, <ins>ICML, 2019 </ins> [[Paper](https://arxiv.org/abs/1903.05895)] [[Code](https://github.com/HazyResearch/butterfly)]

## Efficient Inference
-  Quantized matmul for efficient inference of large-scale generative language models.
- https://github.com/huggingface/candle
- Fast Inference from Transformers via Speculative Decoding.
- AdaTape: Foundation model with adaptive computation and dynamic read-and-write, <ins>Google Blog </ins> [[Paper](https://ai.googleblog.com/2023/08/adatape-foundation-model-with-adaptive.html)]
- Fly-Swat or Cannon? Cost-Effective Language Model Choice via Meta-Modeling, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.06077)]
- Incrementally-Computable Neural Networks: Efficient Inference for Dynamic Inputs, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.14988)]
- FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU, <ins>ICML, 2023</ins> [[Paper](https://arxiv.org/abs/2303.06865)] [[Code](https://github.com/FMInference/FlexGen)]
- H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models, <ins>ICML Workshop, 2023</ins> [[Paper](https://arxiv.org/abs/2306.14048)]
- Predictive Pipelined Decoding: A Compute-Latency Trade-off for Exact LLM Decoding, <ins>ICML Workshop, 2023</ins> [[Paper](https://arxiv.org/abs/2307.05908)]
- SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.02628)]
- FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.05176)]
- Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time, <ins>ICML, 2023</ins> [[Paper](https://www.andrew.cmu.edu/user/beidic/)]
- High-throughput Generative Inference of Large Language Models with a Single GPU, <ins>ICML, 2023</ins> [[Paper](https://www.andrew.cmu.edu/user/beidic/)]
- SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification, <ins>Arxiv, 2023</ins>  [[Paper](https://doi.org/10.48550/arXiv.2305.09781)] [[Code](https://github.com/flexflow/FlexFlow/tree/inference)]
- An Efficient Sparse Inference Software Accelerator for Transformer-based Language Models on CPUs, <ins>Arxiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2306.16601)]
- Efficiently Scaling Transformer Inference.
- DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale

## Efficient Training
- Optimizing transformer-based machine translation model for single GPU training: a hyperparameter ablation study, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.06017)]
- Skill-it! A Data-Driven Skills Framework for Understanding and Training Language Models, <ins>ICML Workshop, 2023</ins> [[Paper](https://arxiv.org/abs/2307.14430)]
- Scaling TransNormer to 175 Billion Parameters, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.14995)] [[Code](https://github.com/OpenNLPLab/TransnormerLLM)]
- Efficient Training of Language Models using Few-Shot Learning, <ins>ICML, 2023</ins> [[Paper](https://openreview.net/forum?id=SpFIO5Mdso)]
- InRank: Incremental Low-Rank Learning, <ins>ICML, 2023</ins> [[Paper](https://arxiv.org/abs/2306.11250)] [[Code](https://github.com/jiaweizzhao/inrank)]
- Stack More Layers Differently: High-Rank Training Through Low-Rank Updates, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.05695)]
- No Train No Gain: Revisiting Efficient Training Algorithms For Transformer-based Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2307.06440)]
- SparseProp: Efficient Sparse Backpropagation for Faster Training of Neural Networks at the Edge, <ins>ICML, 2023</ins> [[Paper](https://openreview.net/forum?id=JSTp7NiuYi)] [[Code](https://github.com/IST-DASLab/sparseprop)]
- A Survey on Efficient Training of Transformers, <ins>IJCAI, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2302.01107)]
- SNT: Sharpness-Minimizing Network Transformation for Fast Compression-friendly Pretraining, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.04526)] 
- Training Large Language Models Efficiently with Sparsity and Dataflow, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2304.05511)]
- LLM-QAT: Data-Free Quantization Aware Training for Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.17888)] 
- On Efficient Training of Large-Scale Deep Learning Models: A Literature Review, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2304.03589)]
- Survey on Efficient Training of Large Neural Networks, <ins>IJCAI, 2022</ins> [[Paper](https://www.ijcai.org/proceedings/2022/769)]
- Compute-Efficient Deep Learning: Algorithmic Trends and Opportunities, <ins>JMLR, 2023</ins> [[Paper](https://www.jmlr.org/papers/volume24/22-1208/22-1208.pdf)]
- FATE-LLM: https://github.com/FederatedAI/FATE-LLM/releases/tag/v1.2.0
## Efficient Long-Text Training/Inference
- Extending Context Window of Large Language Models via Positional Interpolation (arxiv 2023)
- Memorizing Transformers (ICLR 2022)
- Augmenting Language Models with Long-Term Memory (arxiv 2023)
- Unlimiformer: Long-Range Transformers withUnlimited Length Input (arxiv 2023)
- SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification (arxiv 2023)
- Parallel Context Windows for Large Language Models (ACL 2023)
- Structured Prompting: Scaling In-Context Learning to 1,000 Examples (arxiv 2023)
- Naive Bayes-based Context Extension (Github: https://github.com/bojone/NBCE)
- LongNet: Scaling Transformers to 1,000,000,000 Tokens (arxiv 2023)
- Efficient Long-Text Understanding with Short-Text Models (Published in TACL 2023, will be presented in ACL 2023)
- Focused Transformer: Contrastive Training for Context Scaling (arxiv 2023)
- Lost in the Middle: How Language Models Use Long Contexts (arxiv 2023)
- Landmark Attention: Random-Access Infinite Context Length for Transformers (arxiv 2023)
- A Length-Extrapolatable Transformer (ACL 2023)
- Lost in the Middle: How Language Models Use Long Contexts (arxiv 2023)

## Efficient Framework
### General Framework
- DeepSpeed: https://arxiv.org/abs/2207.00032
- [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII)
- [mosec](https://github.com/mosecorg/mosec)
### LLM Specific Framework
- FasterTransformer: https://github.com/NVIDIA/FasterTransformer/
- DeepSpeed-Chat: https://arxiv.org/abs/2308.01320
- Megatron-LM: https://arxiv.org/abs/1909.08053
- [vLLM](https://github.com/vllm-project/vllm)
- [Text generation inference](https://github.com/huggingface/text-generation-inference)
- [OpenLLM](https://github.com/bentoml/OpenLLM)
- [MLC LLM](https://github.com/mlc-ai/mlc-llm)
- [OpenLLM](https://github.com/bentoml/OpenLLM)
- [skypilot](https://github.com/skypilot-org/skypilot)
- [ray-llm](https://github.com/ray-project/ray-llm)
