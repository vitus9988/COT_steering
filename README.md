# CoT Steering without Prompting

This repository aims to develop **Chain-of-Thought (CoT) Steering** based on the *CoT without Prompting* paradigm.  
We focus on improving the model’s latent reasoning ability **without additional training**, by leveraging **Test-Time Scaling** techniques.

## 🔍 Overview

Traditional CoT methods rely on explicit prompts or handcrafted examples to elicit step-by-step reasoning.  
This project explores how CoT dynamics can be *steered internally* during inference without such external guidance, offering a scalable and prompt-free approach to reasoning.

Key objectives:
- Steer latent CoT behavior without prompt engineering
- Apply test-time modulation techniques to enhance reasoning
- Evaluate controllability and performance across reasoning tasks

## 📁 Repository Structure

```
.
├── configs/          # Experiment configurations
├── src/              # Core implementation
├── eval/             # Evaluation scripts and benchmarks
├── scripts/          # Training / inference scripts
└── README.md         # Project documentation
```

## ⚙️ Installation

```bash
git clone https://github.com/your-username/cot-steering.git
cd cot-steering
pip install -r requirements.txt
```

> 📝 Requires Python 3.8+ and PyTorch.

## 🚀 Usage

Run inference with CoT Steering enabled:

```bash
python src/inference.py --config configs/example.yaml
```

## 📊 Evaluation

We evaluated the effectiveness of **CoT Steering without Prompting** on the **2025 Korean CSAT (수능) Language Section**, using the model `FuseAI/FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview`.

Through CoT Steering, we achieved a significant improvement in performance without any additional training.

### 📈 Performance Comparison

| Model                                      | Score (Language Section) |
|-------------------------------------------|---------------------------|
| HyperClovaX     | 61                        |
| gpt 4o       | 75                        |
| deepseek r1     | 78                        |
| O1 mini      | 78                        |
| FuseO1-DeepSeekR1-QwQ-SkyT1-32B(Base line)            | 67                    |
| FuseO1-DeepSeekR1-QwQ-SkyT1-32B(COT Steering)            | **84**                    |
| O1 preview             | 97                    |

The baseline model achieved a score in the 60s range. After applying our CoT Steering mechanism, the model reached a score of **84**, demonstrating the potential of test-time reasoning modulation in high-stakes language comprehension tasks.

We enhanced the model’s latent reasoning capability through **CoT Steering**, enabling notable performance gains without additional training.

Notably, while most comparison baselines utilize models ranging from **100B to 685B parameters**, our approach achieved competitive performance using a **33B-parameter** model. This result highlights the effectiveness and efficiency of CoT Steering, demonstrating its potential to unlock strong reasoning abilities even in relatively smaller models.

While steering can be applied either at the **token level** or in the **latent space** or using a **potential function**, we observed no significant performance difference between the two approaches.
Given this, we adopted the **token-level steering method**, which offers greater `flexibility` and `computational efficiency`, making it more **practical for real-world deployment**.




## 📌 Citation

Coming soon.

## 🧑‍💻 Contributors

- Seugyoo Lee (@DopeorNope-Lee)


## 📄 License

This project is licensed under the MIT License.
