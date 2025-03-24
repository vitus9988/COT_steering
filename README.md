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
| FuseO1-DeepSeekR1-QwQ-SkyT1-32B            | **84**                    |
| O1 preview             | 97                    |

The baseline model achieved a score in the 60s range. After applying our CoT Steering mechanism, the model reached a score of **84**, demonstrating the potential of test-time reasoning modulation in high-stakes language comprehension tasks.

Further evaluations on different subjects and reasoning benchmarks are in progress.

## 📌 Citation

Coming soon.

## 🧑‍💻 Contributors

- Your Name (@your_handle)
- Additional contributors welcome!

## 📄 License

This project is licensed under the MIT License.
