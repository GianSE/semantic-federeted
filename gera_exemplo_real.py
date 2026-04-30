import torch
import matplotlib.pyplot as plt
import numpy as np
from data import get_federated_dataloaders
from model_autoencoder import build_autoencoder
import os

def generate_real_example():
    print("Carregando dados reais do CIFAR-10...")
    # Usar latent_dim=64 como exemplo
    latent_dim = 64
    device = torch.device("cpu")
    
    # Carregar dados reais e procurar por um carro (classe 1)
    loaders, test_loader = get_federated_dataloaders("cifar10", num_clients=1, batch_size=1, test_batch_size=1, seed=42)
    
    print("Buscando o segundo pássaro no dataset...")
    count = 0
    for img, label in test_loader:
        if label.item() == 2: # 2 é a classe 'bird'
            count += 1
            if count == 3: # Pegar o terceiro pássaro
                break
    
    # Inicializar o modelo e otimizador
    model = build_autoencoder("cifar10", latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    recon_loss_fn = torch.nn.MSELoss()
    
    # "Mini-treino" rápido para demonstração (overfitting em 1 imagem)
    print("Ajustando pesos para demonstração técnica...")
    model.train()
    for _ in range(150):
        optimizer.zero_grad()
        z, recon = model(img)
        loss = recon_loss_fn(recon, img)
        loss.backward()
        optimizer.step()
    
    model.eval()
    # Obter o vetor latente real após o ajuste
    with torch.no_grad():
        z, recon = model(img)
    
    # Preparar para plotagem
    img_np = img.squeeze().permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    recon_np = recon.squeeze().permute(1, 2, 0).numpy()
    recon_np = (recon_np - recon_np.min()) / (recon_np.max() - recon_np.min())
    
    z_np = z.squeeze().numpy()
    
    # Criar o mosaico técnico
    fig = plt.figure(figsize=(12, 4))
    
    # 1. Imagem Original
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img_np)
    ax1.set_title("Imagem Original (CIFAR-10)")
    ax1.axis("off")
    
    # 2. Embedding (Vetor Latente)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.bar(range(len(z_np)), z_np, color='teal')
    ax2.set_title(f"Embedding (Vetor L={latent_dim})")
    ax2.set_xlabel("Índice do Elemento")
    ax2.set_ylabel("Valor")
    
    # 3. Reconstrução
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(recon_np)
    ax3.set_title("Reconstrução (Decoder)")
    ax3.axis("off")
    
    plt.tight_layout()
    save_path = "docs/paper/mosaico_real.png"
    plt.savefig(save_path, dpi=300)
    print(f"Mosaico real salvo em: {save_path}")
    
    # Mostrar os primeiros 10 valores para o usuário
    print("\nExemplo dos primeiros 10 valores do embedding:")
    print(z_np[:10])

if __name__ == "__main__":
    generate_real_example()
