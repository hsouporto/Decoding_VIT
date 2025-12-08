# Decoding Vision Transformer Variations for Image Classification <small> - A Guide to Performance and Usability</small>





## Abstract
With the introduction of transformers, the Vision Transformer (ViT) architecture has become one of the leading approaches for image classification. This rapid expansion has produced a wide range of architectural variations, making systematic analysis essential to understand their behaviour, performance, and usability.

This work reviews, analyses, and benchmarks ViT variants using a clear taxonomy, comparing pure transformer models, hybrid CNN–ViT architectures, and traditional CNNs. The comparison highlights key trade-offs between accuracy, computational complexity, and design choices, providing insights that help researchers identify the most suitable architecture for their specific tasks.

## 🔑 Keywords
- Visual Transformers  
- Transformer  
- Complexity




## 🖼 Model Relation Diagram

![Diagram](images/diagram.png)


## 🛠 Usage / Dependencies

The project requires the following Python packages:

pip install -r requirements.txt

```text
certifi==2022.12.7
charset-normalizer==2.1.1
colorama==0.4.6
contourpy==1.2.0
cycler==0.12.1
einops==0.7.0
filelock==3.9.0
fonttools==4.47.0
fsspec==2023.4.0
graphviz==0.20.1
idna==3.4
Jinja2==3.1.2
kiwisolver==1.4.5
MarkupSafe==2.1.3
matplotlib==3.8.2
mpmath==1.3.0
networkx==3.0
numpy==1.24.1
opencv-python==4.9.0.80
packaging==23.2
Pillow==9.3.0
pydicom==2.4.4
pyparsing==3.1.1
python-dateutil==2.8.2
requests==2.28.1
six==1.16.0
sympy==1.12
torch==2.1.2+cu118
torchaudio==2.1.2+cu118
torchvision==0.16.2+cu118
torchviz==0.0.2
tqdm==4.66.1
typing_extensions==4.4.0
urllib3==1.26.13
```


## 📦 Model Weights & Checkpoints

All pre-trained weights and checkpoints for the models presented in the benchmark tables are available for download. These include CNNs, Transformers, and Hybrid CNN–ViT models. Use them to reproduce experiments or fine-tune on your datasets.

- **Download all weights:** [Google Drive Folder](https://drive.google.com/drive/folders/1rJ8rSHNXU3y4IqndCBFuM8-JCU-Bo4we?usp=sharing)  
  This folder contains organized subfolders for each model type with respective pre-trained checkpoints and configuration files.



## Authors
**João Montrezol**  
**Hugo S. Oliveira**  
**Hélder P. Oliveira**
---
