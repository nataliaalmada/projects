from pathlib import Path
#Para verificar se todas as imagens do dataset tinha uma label correspondente 
#para ter certeza de que todas as imagens que foram selecionadas e marcadas serão lidas pelo modelo
# Caminhos para as pastas
image_dir = Path("C:/Users/Nat Natalia/Desktop/TCC2/dataset/images/all")
label_dir = Path("C:/Users/Nat Natalia/Desktop/TCC2/dataset/labels/all")

# Lista de imagens
image_files = list(image_dir.glob("*.jpg"))

# Verificar se cada imagem tem seu .txt
missing_labels = []
for img in image_files:
    txt_name = img.with_suffix('.txt').name
    if not (label_dir / txt_name).exists():
        missing_labels.append(img.name)

# Mostrar o resultado
if missing_labels:
    print("Imagens sem arquivo .txt correspondente:\n")
    for name in missing_labels:
        print(name)
else:
    print("Todas as imagens têm arquivo .txt correspondente.")
