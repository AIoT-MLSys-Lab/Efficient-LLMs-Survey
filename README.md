# Efficient GPT: Survey, Papers, Benchmarks, and Open-Source Resources.  

### ToDo (05/10/2023) 
* Keep adding papers into the list.
* Put the papers under their correct categories. Create new categories if the papers do not fit the current ones.
* Add the link of the open source codes (e.g., github repo) if the paper has one.
* Add datasets, benchmarks used by the papers.

## What is Efficient GPT About?

## Survey
* Efficient GPT: A Survey, 2023.
* Efficient GPT for Mobile Computing: Challenges and Opportunities, 2023.

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

## Open LLM
- Open LLM Leaderboard: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
- Awesome-LLM: https://github.com/Hannibal046/Awesome-LLM
- KoLA: Carefully Benchmarking World Knowledge of Large Language Models, [[Paper](https://paperswithcode.com/paper/kola-carefully-benchmarking-world-knowledge)] [[Code](https://github.com/thu-keg/kola)]
- Benchmarking LLM Inference Efficiency: https://ml.energy/leaderboard/?__theme=light

## Efficient Fine-Tuning

#### Data Efficient
- LIMA: Less Is More for Alignment,  <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.11206)]
- Data-Efficient Finetuning Using Cross-Task Nearest Neighbors, <ins>ACL, 2023</ins> [[paper](https://doi.org/10.48550/arXiv.2212.00196)][[Code](https://github.com/allenai/data-efficient-finetuning)]
- Self-Instruct: Aligning Language Model with Self Generated Instructions, <ins>ACL, 2023</ins> [[paper](https://doi.org/10.48550/arXiv.2212.10560)] [[Code](https://github.com/yizhongw/self-instruct)]
- Data Selection for Fine-tuning Large Language Models Using Transferred Shapley Values, <ins>ACL SRW, 2023</ins> [[paper](https://doi.org/10.48550/arXiv.2306.10165)] [[Code](https://github.com/stephanieschoch/ts-dshapley)]

#### Memory Efficient
- Full Parameter Fine-tuning for Large Language Models with Limited Resources, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.09782)] [[Code](https://github.com/OpenLMLab/LOMO)]
- Fine-Tuning Language Models with Just Forward Passes, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.17333)] [[Code](https://github.com/princeton-nlp/MeZO)]

#### Parameter Efficient
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
#### Model Pruning
- The Emergence of Essential Sparsity in Large Pre-trained Models: The Weights that Matter, <ins>Arxiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2306.03805)] [[Code](https://github.com/VITA-Group/essential_sparsity)]
- Low-Rank Prune-And-Factorize for Language Model Compression, <ins>Arxiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2306.14152)]
- Constraint-aware and Ranking-distilled Token Pruning for Efficient Transformer Inference, <ins>KDD, 2023</ins> [[Paper](https://arxiv.org/abs/2306.14393)]
- LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation, <ins>ICML, 2023</ins>  [[Paper](https://arxiv.org/abs/2306.11222)] [[Code](https://github.com/yxli2123/LoSparse)]
- A Simple and Effective Pruning Approach for Large Language Models, <ins>ACL SRW, 2023</ins> [[Paper](https://arxiv.org/abs/2306.11695)] [[Code](https://github.com/locuslab/wanda)]
- Ten Lessons We Have Learned in the New "Sparseland": A Short Handbook for Sparse Neural Network Researchers, <ins>Arxiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2302.02596)]
- Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning, <ins>Arxiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2305.18403)]
- Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.15805)]
- LLM-Pruner: On the Structural Pruning of Large Language Models, <ins>Github, 2023</ins> [[Paper](https://arxiv.org/abs/2305.11627)] [[Code](https://github.com/horseee/LLM-Pruner)]
- SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2301.00774)] [[Code](https://github.com/IST-DASLab/sparsegpt)]
- ZipLM: Hardware-Aware Structured Pruning of Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2302.04089)]
- Vision Transformer Slimming: Multi-Dimension Searching in Continuous Optimization Space, <ins>CVPR, 2022</ins> [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chavan_Vision_Transformer_Slimming_Multi-Dimension_Searching_in_Continuous_Optimization_Space_CVPR_2022_paper.pdf)] [[Code](https://github.com/Arnav0400/ViT-Slim)]
- Unified Visual Transformer Compression, <ins>ICLR, 2022</ins> [[Paper](https://openreview.net/forum?id=9jsZiUgkCZP)] [[Code](https://github.com/VITA-Group/UVC)]
- From Dense to Sparse: Contrastive Pruning for Better Pre-trained Language Model Compression, <ins>AAAI, 2022</ins> [[Paper](https://arxiv.org/abs/2112.07198)] [[Code](https://github.com/RunxinXu/ContrastivePruning)]
- Visual Transformer Pruning, <ins>KDDW, 2021</ins> [[Paper](https://arxiv.org/abs/2104.08500)] [[Code](https://github.com/Cydia2018/ViT-cifar10-pruning)]
- Accelerating Sparse Deep Neural Networks, 2021, [[Paper](https://arxiv.org/pdf/2104.08378.pdf)]
- To prune, or not to prune: exploring the efficacy of pruning for model compression, <ins>ICLRW, 2018</ins> [[Paper](https://openreview.net/forum?id=S1lN69AT-)] [[Code](https://github.com/IntelLabs/Model-Compression-Research-Package)]


#### Model Quantization
- SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/pdf/2306.03078)] [[Code](https://github.com/Vahe1994/SpQR)]
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
- SqueezeLLM: Dense-and-Sparse Quantization, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.07629)]


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
- Distilling Multi-Step Reasoning Capabilites of Large Language Models into Smaller Models via Semantic Decompositions,  <ins>Arxiv, 2021</ins> [[Paper](https://arxiv.org/abs/2212.00193)]

#### Efficient Attention
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
- FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.05176)]
- Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time, <ins>ICML, 2023</ins> [[Paper](https://www.andrew.cmu.edu/user/beidic/)]
- High-throughput Generative Inference of Large Language Models with a Single GPU, <ins>ICML, 2023</ins> [[Paper](https://www.andrew.cmu.edu/user/beidic/)]
- SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification, <ins>Arxiv, 2023</ins>  [[Paper](https://doi.org/10.48550/arXiv.2305.09781)] [[Code](https://github.com/flexflow/FlexFlow/tree/inference)]
- An Efficient Sparse Inference Software Accelerator for Transformer-based Language Models on CPUs, <ins>Arxiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2306.16601)]

## Efficient Training
- SparseProp: Efficient Sparse Backpropagation for Faster Training of Neural Networks at the Edge, <ins>ICML, 2023</ins> [[Paper](https://openreview.net/forum?id=JSTp7NiuYi)] [[Code](https://github.com/IST-DASLab/sparseprop)]
- A Survey on Efficient Training of Transformers, <ins>IJCAI, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2302.01107)]
- SNT: Sharpness-Minimizing Network Transformation for Fast Compression-friendly Pretraining, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.04526)] 
- Training Large Language Models Efficiently with Sparsity and Dataflow, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2304.05511)]
- LLM-QAT: Data-Free Quantization Aware Training for Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.17888)] 
- On Efficient Training of Large-Scale Deep Learning Models: A Literature Review, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2304.03589)]
- Survey on Efficient Training of Large Neural Networks, <ins>IJCAI, 2022</ins> [[Paper](https://www.ijcai.org/proceedings/2022/769)]
- Compute-Efficient Deep Learning: Algorithmic Trends and Opportunities, <ins>JMLR, 2023</ins> [[Paper](https://www.jmlr.org/papers/volume24/22-1208/22-1208.pdf)]
- FATE-LLM: https://github.com/FederatedAI/FATE-LLM/releases/tag/v1.2.0
