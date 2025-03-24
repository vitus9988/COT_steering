# CoT Steering without Prompting

**Chain-of-Thought reasoning without prompting** is an effective decoding strategy that expands the model’s search space by sampling from the **top-k tokens** without relying on explicit prompts. This approach facilitates the discovery of latent reasoning paths internal to the model, making it a promising direction for structured reasoning tasks.

However, existing implementations of this method often suffer from **critical deviations** from the original paper. These include incorrect modifications to the decoding process, improper handling of aggregation mechanisms, and failure to preserve the intended search semantics. To address these limitations, we re-implemented the method faithfully to the original formulation.

Moreover, we extend it by introducing steering tokens as a mechanism to explicitly condition the model’s reasoning trajectory. This narrows the search space in a controlled way, enabling the model to more reliably follow structured CoT paths. While this steering can theoretically be applied in the latent space, we instead apply it at the **token level**, leveraging the autoregressive nature of the model to inject **constraints directly into the decoding process**.

To perform unconstrained CoT decoding without steering, simply set the `STEERING_TOKEN` to an `empty string ('')`.

## 🧠 Why CoT is a Search Problem

At its core, **CoT reasoning is a search problem**—the model explores multiple candidate trajectories to find an optimal reasoning path that leads to the correct answer. However, due to biases acquired during pretraining or fine-tuning, the model’s default search behavior is often limited to a narrow and suboptimal region of the solution space.

By applying our method, which balances **diversity** and **structured control**, we are able to recover richer reasoning trajectories and guide the model toward more effective CoT outputs, without any additional training.


## 📁 Repository Structure

```
.
├── template/         # Answer parsing Template Documentation
├── scripts/          # COT scripts & Examples
└── README.md         # Project documentation
```

## ⚙️ Installation

```bash
git clone https://github.com/your-username/cot-steering.git
cd cot-steering
pip install -r requirements.txt
```

> 📝 Requires Python 3.11+.

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

Chen, H., Feng, Y., Liu, Z., Yao, W., Prabhakar, A., Heinecke, S., ... & Wang, H. (2024). Language models are hidden reasoners: Unlocking latent reasoning capabilities via self-rewarding. *arXiv preprint arXiv:2411.04282*.

Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J., & Tian, Y. (2024). Training large language models to reason in a continuous latent space. *arXiv preprint arXiv:2412.06769*.

Muennighoff, N., Yang, Z., Shi, W., Li, X. L., Fei-Fei, L., Hajishirzi, H., ... & Hashimoto, T. (2025). S1: Simple test-time scaling. *arXiv preprint arXiv:2501.19*.

Rodriguez, P., Blaas, A., Klein, M., Zappella, L., Apostoloff, N., Cuturi, M., & Suau, X. (2024). Controlling language and diffusion models by transporting activations. *arXiv preprint arXiv:2410.23054*.

Snell, C., Lee, J., Xu, K., & Kumar, A. (2024). Scaling LLM test-time compute optimally can be more effective than scaling model parameters. *arXiv preprint arXiv:2408.03314*.

Wang, B., Min, S., Deng, X., Shen, J., Wu, Y., Zettlemoyer, L., & Sun, H. (2022). Towards understanding chain-of-thought prompting: An empirical study of what matters. *arXiv preprint arXiv:2212.10001*.

Wang, X., & Zhou, D. (2024). Chain-of-thought reasoning without prompting. *arXiv preprint arXiv:2402.10200*.

Zhang, Z., Zhang, A., Li, M., & Smola, A. (2022). Automatic chain of thought prompting in large language models. *arXiv preprint arXiv:2210.03493*.

Zhao, S., Brekelmans, R., Makhzani, A., & Grosse, R. (2024). Probabilistic inference in language models via twisted sequential Monte Carlo. *arXiv preprint arXiv:2404.17546*.


## 🧑‍💻 Contributors

- Seugyoo Lee (@DopeorNope-Lee)


## 📄 License

This project is licensed under the MIT License.
