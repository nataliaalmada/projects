# Trabalho de Conclusão de Curso
- Universidade Federal de Roraima
- Bacharelado em Ciência da Computação
- Autora: Natália Ribeiro de Almada
- Sob Orientação do Profesor Dr. Herbert Oliveira Rocha

## Projeto para Detecção de Quedas de Idosos

Este projeto tem como objetivo utilizar dois modelos existentes (YOLO-NAS e YOLOv8) para comparar seus desempenhos, inicialmente com o dataset original fornecido com os modelos e, posteriormente, modificados para comportarem um dataset autoral, permitindo avaliação de performance.
# Dataset Autoral: 
```bash

├── 156 vídeos                  ← Imagens de treino/validação/teste
│   ├── 53 vídeos de idosos como "Não Queda"/
│   └── 103 vídeos de idosos como "Queda"
│ └──  Ao todo, 648 frames      ← Divisão de treino=70% /validação=20% /teste = 10%
10 vídeos que não passaram por nenhuma fase do treinamento para serem testados. 
Classes: "fall" e "person"
````
## YOLOv8 com Dataset Autoral:
Projeto executado em Windows 11

P2_YOLOv8
   ```bash

│
├── dataset/                 ← Imagens de treino/validação/teste
│   ├── images/
│   └── data.yaml
│
├── videos_testes_validacao/← Vídeos de validação
├── videos_teste/           ← Vídeos de teste (não passaram pelo treinamento em nenhum momento)
├── runs/
│   └── detect/
│       └── predict/        ← Saídas da inferência
│           ├── queda/
│           └── naoQueda/
│
├── treinamento_yolo_com_relatorio.ipynb ← Notebook com pipeline completo
└── requirements.txt         ← Dependências do projeto

```


### 🚀 Funcionalidades

- Treinamento e validação de modelo YOLOv8 com dados próprios.
- Avaliação com vídeos externos.
- Classificação automática de vídeos de teste:
  - Vídeos com detecção de **fall** → `videos_saida/queda/`
  - Vídeos **sem detecção de fall** → `videos_saida/naoQueda/`
- Envio automático de alerta via **Telegram** com vídeo anexo.
- Geração de gráficos (IoU, Loss, Accuracy, Histogramas etc.) salvos em PNG.

---

### 🛠️ Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/colocaroresto.git
   cd VisionFall
2. Utilize o notebook disponível em [Notebook no Colab](https://colab.research.google.com/drive/1Z2qt6rKFA-6tgDqdjNlNJybMGVslDUxy
).


## Execução
1. Crie e ative um ambiente virtual (Windows):
    ```bash
    python -m venv venv
    .\venv\Scripts\activate

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt

3. Inicie o Notebook
   ```bash
   jupyter notebook

5. Execute as células

## YOLO-NAS com Dataset Autoral:
Projeto executado em WSL2 através do Windows 11

P2_YOLO_NAS
   ```bash

│
├── datasetNAS/ # Dataset customizado
│ ├── train/
│ ├── valid/
│ └── test/
│
├── videos_teste/ # Vídeos para teste com o modelo treinado
├── videos_saida/
│ ├── queda/ # Vídeos onde quedas foram detectadas
│ └── naoQueda/ # Vídeos sem detecção de queda
│
├── checkpoints/ # Pesos salvos do modelo treinado
└── fall-detection-yolo-nas-train-predict.ipynb
```


### 🚀 Funcionalidades

- Treinamento e validação de modelo YOLOv8 com dados próprios.
- Avaliação com vídeos externos.
- Classificação automática de vídeos de teste:
  - Vídeos com detecção de **fall** → `videos_saida/queda/`
  - Vídeos **sem detecção de fall** → `videos_saida/naoQueda/`
- Envio automático de alerta via **Telegram** com vídeo anexo.
- Geração de gráficos (IoU, Loss, Accuracy, Histogramas etc.) salvos em PNG.

---
### ✅ Pré-requisitos

### WSL2 no Windows
Siga o guia oficial:
- https://learn.microsoft.com/pt-br/windows/wsl/install
Você deve instalar:
- **Ubuntu**
- **WSL2 como padrão**
- Instale o Miniconda no WSL acitando os termos e permita que ele adicione ao PATH.

### 🛠️ Instalação

1.  Crie o ambiente Conda para o projeto
   ```bash
conda create -n yolonas_env python=3.10 -y
conda activate yolonas_env
```
3.  Clone o repositório:
   ```bash
   cd /mnt/c/Users/SEU_USUARIO/Desktop
git clone https://github.com/SEU_USUARIO/P2_YOLO_NAS.git
cd P2_YOLO_NAS
````
2. Utilize o notebook disponível em [Notebook no Colab](https://drive.google.com/file/d/1U3GGm3UOQDEh31S0PYxg6wcgJLgvzSU0/view?usp=sharing
). 
3. Instale as dependÇencias
   ```bash
   pip install -r requirements.txt
📓 Como executar o notebook
1. Inicie o Jupyter no WSL2
 ```bash

conda activate yolonas_env
jupyter notebook
 ```
Ele mostrará uma URL como:

http://localhost:8888/?token=...
Copie e cole no navegador do Windows.

2. Abra o notebook
Abra o arquivo:

 ```bash
fall-detection-yolo-nas-train-predict.ipynb
 ```
2. Execute as células

📦 Treinamento com seu próprio dataset
Certifique-se de que seu dataset está no formato YOLOv5/v8:

📹 Testes com Vídeos
Coloque até 10 vídeos de idoso caindo.mp4 em:

videos_teste/
-Após o teste, o sistema criará:

````bash
videos_saida/
├── queda/
├── naoQueda/
````
### 📲 Integração com Telegram (opcional para ambos os modelos)
### ✅ Pré-requisitos

### Ter pelo menos uma conta no [Telegram](https://web.telegram.org)
1. Crie um app no [Telegram](https://my.telegram.org) 
configure as seguintes informações no código
```bash
api_id = 'SEU_API_ID'
api_hash = 'SEU_API_HASH'
phone = '+55xxxxxxxxx'
dest_user_id = SEU_ID
dest_access_hash = SEU_ACCESS_HASH
```

# YOLOv8 Modelo de Referência
- Disponível em [GitHub](https://github.com/Tech-Watt/Fall-Detection)
seguindo tutorial do [YouTube](https://youtu.be/wrhfMF4uqj8?si=bgvQIP8dlM2cwJRm)- Tech Watt

# YOLO-NAS Modelo de Referência
- Modelo Disponível em [Kaggle](https://www.kaggle.com/code/stpeteishii/fall-detection-yolo-nas-train-predict) ou Com pequenas modificações necessárias para execução na máquina no WSL2  disponível no [Colab](https://drive.google.com/file/d/1k1Nu0BolRzG4UkjND6e6O2NIybvUXsCc/view?usp=sharing)
- Dataset Disponível em [Kaggle](https://www.kaggle.com/datasets/elwalyahmad/fall-detection)
- Coco, pesos pré-treinados disponíveis em [Hugging Face](https://huggingface.co/bdsqlsz/YOLO_NAS/blob/main/yolo_nas_l_coco.pth)
