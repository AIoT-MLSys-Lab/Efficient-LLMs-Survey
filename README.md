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
- [Efficient Instruction Tuning](#Efficient-Instruction-Tuning)
- [Parameter-Efficient Fine-Tuning](#Parameter-Efficient-Fine-Tuning)
- [Model Compression](#Model-Compression)
  - [Model Pruning](#Model-Pruning)
  - [Knowledge Distillation](#Knowledge-Distillation)
  - [Low-rank Decomposition](#Low-rank-Decomposition)
  - [Quantization](#Quantization)
  - [Efficient Attention](#Efficient-Attention)
- [Efficient Inference](#Efficient-Inference)
- [Efficient Training](#Efficient-Training)

## Efficient Instruction Tuning
- LIMA: Less Is More for Alignment,  <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.11206)] 
- Self-Instruct: Aligning Language Model with Self Generated Instructions, <ins>ACL, 2023</ins> [[paper]( https://doi.org/10.48550/arXiv.2212.10560)] [[Code](https://github.com/yizhongw/self-instruct)]

## Parameter-Efficient Fine-Tuning
- QLORA: Efficient Finetuning of Quantized LLMs, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.14314)]
- Compacter: Efficient Low-Rank Hypercomplex Adapter Layers, <ins>NeurlPS, 2023</ins> [[Paper](https://openreview.net/forum?id=bqGK5PyI6-N)] [[Code](https://github.com/rabeehk/compacter)]
- LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention, <ins>Arxiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2303.16199)] [[Code](https://github.com/ZrrSkywalker/LLaMA-Adapter)]
- DyLoRA: Parameter-Efficient Tuning of Pretrained Models using Dynamic Search-Free Low Rank Adaptation, <ins>EACL, 2023</ins> [[Paper](https://aclanthology.org/2023.eacl-main.239/)] [[Code](https://github.com/huawei-noah/KD-NLP/tree/main/DyLoRA)]
- Parameter-efficient Fine-tuning Design Spaces, <ins>ICLR, 2023</ins> [[Paper](https://openreview.net/forum?id=XSRSWxyJIC)] [[Code](https://github.com/amazon-science/peft-design-spaces)]
- PEFT: State-of-the-art Parameter-Efficient Fine-Tuning (PEFT) methods, <ins>Github, 2022</ins> [[Code](https://github.com/huggingface/peft)]
- Parameter-efficient Fine-tuning of Large-scale Pre-trained Language Models, <ins>Nature Machine Intelligence, 2023</ins> [[Paper](https://doi.org/10.1038/s42256-023-00626-4)] [[Code](https://github.com/thunlp/OpenDelta)]
- LoRA: Low-Rank Adaptation of Large Language Models, <ins>ICLR, 2022</ins> [[Paper](https://openreview.net/forum?id=nZeVKeeFYf9)] [[Code](https://github.com/microsoft/LoRA)]
- Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning, <ins>NeurlPS, 2022</ins> [[Paper](https://openreview.net/forum?id=rBCvMG-JsPd)] [[Code](https://github.com/r-three/t-few)]
- Parameter-Efficient Sparsity for Large Language Models Fine-Tuning, <ins>IJCAI, 2022</ins> [[Paper](https://www.ijcai.org/proceedings/2022/0586.pdf)] [[Code](https://github.com/yuchaoli/PST)]
- Attempt: Parameter-Efficient Multi-task Tuning via Attentional Mixtures of Soft Prompts, <ins>EMNLP, 2022</ins> [[Paper](https://aclanthology.org/2022.emnlp-main.446/)] [[Code](https://github.com/AkariAsai/ATTEMPT)]

## Model Compression
#### Model Pruning
- LLM-Pruner: On the Structural Pruning of Large Language Models, <ins>Github, 2023</ins> [[Code](https://github.com/horseee/LLM-Pruner)]
- SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2301.00774)] [[Code](https://github.com/IST-DASLab/sparsegpt)]
- ZipLM: Hardware-Aware Structured Pruning of Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2302.04089)]
- Vision Transformer Slimming: Multi-Dimension Searching in Continuous Optimization Space, <ins>CVPR, 2022</ins> [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chavan_Vision_Transformer_Slimming_Multi-Dimension_Searching_in_Continuous_Optimization_Space_CVPR_2022_paper.pdf)] [[Code](https://github.com/Arnav0400/ViT-Slim)]
- Unified Visual Transformer Compression, <ins>ICLR, 2022</ins> [[Paper](https://openreview.net/forum?id=9jsZiUgkCZP)] [[Code](https://github.com/VITA-Group/UVC)]
- From Dense to Sparse: Contrastive Pruning for Better Pre-trained Language Model Compression, <ins>AAAI, 2022</ins> [[Paper](https://arxiv.org/abs/2112.07198)] [[Code](https://github.com/RunxinXu/ContrastivePruning)]
- Visual Transformer Pruning, <ins>KDDW, 2021</ins> [[Paper](https://arxiv.org/abs/2104.08500)] [[Code](https://github.com/Cydia2018/ViT-cifar10-pruning)]
- To prune, or not to prune: exploring the efficacy of pruning for model compression, <ins>ICLRW, 2018</ins> [[Paper](https://openreview.net/forum?id=S1lN69AT-)] [[Code](https://github.com/IntelLabs/Model-Compression-Research-Package)]
#### Knowledge Distillation
- Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes, <ins>ACL, 2023</ins> [[Paper](https://arxiv.org/abs/2305.02301)]
- What Language Reveals about Perception: Distilling Psychophysical Knowledge from Large Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2302.01308)]
- Specializing Smaller Language Models towards Multi-Step Reasoning, <ins>ICML, 2023</ins> [[Paper](https://aclanthology.org/2022.findings-naacl.169.pdf)] [[Code](https://github.com/FranxYao/FlanT5-CoT-Specialization)]
- Lifting the Curse of Capacity Gap in Distilling Language Models, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.12129)] [[Code](https://github.com/genezc/minimoe)]
- Distilling Multi-Step Reasoning Capabilites of Large Language Models into Smaller Models via Semantic Decompositions,  <ins>Arxiv, 2021</ins> [[Paper](https://arxiv.org/abs/2212.00193)]

#### Low-rank Decomposition
- Compressing Transformers: Features Are Low-Rank, but Weights Are Not!  <ins>AAAI, 2023</ins> [[Paper](https://cs.nju.edu.cn/wujx/paper/AAAI2023_AFM.pdf)]
- Strategies for Applying Low Rank Decomposition to Transformer-Based Models,  <ins>NeurlPS-ENLSP, 2022</ins> [[Paper](https://neurips2022-enlsp.github.io/papers/paper_33.pdf)]
- Numerical Optimizations for Weighted Low-rank Estimation on Language Model,  <ins>EMNLP, 2022</ins> [[Paper](https://aclanthology.org/2022.emnlp-main.91.pdf)] 
- Language Model Compression With Weighted Low-rank Factorization,  <ins>ICLR, 2022</ins> [[Paper](https://openreview.net/pdf?id=uPv9Y3gmAI5)]
- Monarch: Expressive Structured Matrices for Efficient and Accurate Training,  <ins>ICML, 2022</ins> [[Paper](https://proceedings.mlr.press/v162/dao22a/dao22a.pdf)] [[Code](https://github.com/HazyResearch/fly)]
- Compressing Pre-trained Language Models using Progressive Low Rank Decomposition,  <ins>NeurlPS-ENLSP, 2021</ins> [[Paper](https://neurips2021-nlp.github.io/papers/27/CameraReady/Neurips_Workshop_camera_ready.pdf)]
- Kronecker Decomposition for GPT Compression,  <ins>NeurlPS-ENLSP, 2021</ins> [[Paper](https://aclanthology.org/2022.acl-short.24.pdf)]

#### Quantization
- AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.00978)] [[Code](https://github.com/mit-han-lab/llm-awq)]
- GPTQ: Accurate Quantization for Generative Pre-trained Transformers, <ins>ICLR, 2023</ins> [[Paper](https://openreview.net/forum?id=tcbBPnfwxS)] [[Code](https://github.com/IST-DASLab/gptq)]
- OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2304.07493)]
- ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers, <ins>NeurlPS, 2022</ins> [[Paper](https://openreview.net/forum?id=f-fVCElZ-G1)]
- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models, <ins>NeurlPS-ENLSP, 2022 </ins>[[Paper](https://arxiv.org/abs/2211.10438)] [[Code](https://github.com/mit-han-lab/smoothquant)]
- GPT3.int8(): 8-bit Matrix Multiplication for Transformers at Scale, <ins>NeurlPS, 2022</ins> [[Paper](https://openreview.net/forum?id=dXiGWqBoxaD)] [[Code](https://doi.org/10.48550/arXiv.2208.07339)]
- Compression of Generative Pre-trained Language Models via Quantization, <ins>ACL, 2022</ins> [[Paper](https://aclanthology.org/2022.acl-long.331.pdf)]
- Towards Efficient Post-training Quantization of Pre-trained Language Models, <ins>NIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2109.15082)]

#### Efficient Attention
- RWKV: Reinventing RNNs for the Transformer Era, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.13048)] 
- HALOS: Hashing Large Output Space for Cheap Inference,  <ins>MLSys, 2022</ins> [[Paper](https://proceedings.mlsys.org/paper/2022/hash/1ff8a7b5dc7a7d1f0ed65aaa29c04b1e-Abstract.html)]
- Reformer: The Efficient Transformer,  <ins>ICLR, 2022</ins> [[Paper](https://openreview.net/forum?id=rkgNKkHtvB)] [[Code](https://github.com/lucidrains/reformer-pytorch)]
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

## Efficient Training

- A Survey on Efficient Training of Transformers, <ins>IJCAI, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2302.01107)]
- Training Large Language Models Efficiently with Sparsity and Dataflow, <ins>Arxiv, 2023</ins> [[Paper](https://arxiv.org/abs/2304.05511)]
