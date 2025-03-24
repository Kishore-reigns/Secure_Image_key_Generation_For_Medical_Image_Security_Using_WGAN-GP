import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, datasets, utils
import os
import kagglehub
import torch.autograd as autograd
import matplotlib.pyplot as plt
import pandas as pd 

data = []

# Function to download multiple datasets
def download_datasets(dataset_list):
    dataset_dirs = [kagglehub.dataset_download(dataset) for dataset in dataset_list]
    return dataset_dirs

# Function to load multiple datasets into a single DataLoader
def load_multiple_datasets(data_dirs, transform, batch_size):
    datasets_list = [datasets.ImageFolder(data_dir, transform=transform) for data_dir in data_dirs]
    combined_dataset = ConcatDataset(datasets_list)
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

# Generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),  # Output (3, 256, 256)
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

# Critic network
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 1, 4, 1, 0, bias=False)
        )
    
    def forward(self, img):
        return self.model(img).view(-1)

# Compute gradient penalty
def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    critic_interpolates = critic(interpolates)
    grad_outputs = torch.ones_like(critic_interpolates, device=device)
    
    gradients = autograd.grad(
        outputs=critic_interpolates, inputs=interpolates,
        grad_outputs=grad_outputs, create_graph=True, retain_graph=True,
        only_inputs=True
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Save generated images
def save_generated_images(generator, epoch, device, num_images=8):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, 100, 1, 1, device=device)
        fake_images = generator(z)
        fake_images = (fake_images + 1) / 2  # Normalize from [-1,1] to [0,1]
        
        os.makedirs("generated_images", exist_ok=True)
        image_path = f"generated_images/epoch_{epoch}.png"
        utils.save_image(fake_images, image_path, normalize=True, nrow=4)
        print(f"Saved generated images at {image_path}")
    generator.train()

# Training loop
def train_deepkeygen(generator, critic, source_loader, transform_loader, num_epochs, lr, device):
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_d = optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))
    lambda_gp = 10
    critic_iterations = 5
    
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(num_epochs):
        for (source_imgs, _), (transform_imgs, _) in zip(source_loader, transform_loader):
            if source_imgs.size(0) != transform_imgs.size(0):
                min_batch_size = min(source_imgs.size(0), transform_imgs.size(0))
                source_imgs = source_imgs[:min_batch_size]
                transform_imgs = transform_imgs[:min_batch_size]
            
            source_imgs, transform_imgs = source_imgs.to(device), transform_imgs.to(device)
            
            for _ in range(critic_iterations):
                z = torch.randn(source_imgs.size(0), 100, 1, 1, device=device)
                fake_imgs = generator(z).detach()
                real_loss = critic(transform_imgs).mean()
                fake_loss = critic(fake_imgs).mean()
                gp = compute_gradient_penalty(critic, transform_imgs, fake_imgs, device)
                critic_loss = fake_loss - real_loss + lambda_gp * gp
                optimizer_d.zero_grad()
                critic_loss.backward()
                optimizer_d.step()
            
            z = torch.randn(source_imgs.size(0), 100, 1, 1, device=device)
            fake_imgs = generator(z)
            generator_loss = -critic(fake_imgs).mean()
            optimizer_g.zero_grad()
            generator_loss.backward()
            optimizer_g.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss D: {critic_loss.item()}, Loss G: {generator_loss.item()}")

        data.append(list(epoch+1,generator_loss.item(),critic_loss.item()))
        
        
        save_generated_images(generator, epoch + 1, device)
        torch.save(generator.state_dict(), f"checkpoints/deepkeygen_epoch_{epoch+1}.pth")
        print(f"Checkpoint saved at epoch {epoch+1}")

# Main script
if __name__ == "__main__":
    print(f"[+]current working directory{os.getcwd()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+]Using device: {device}")
    
    csvpath = "loss.csv"
    
    
    source_datasets = ["raddar/tuberculosis-chest-xrays-montgomery", "masoudnickparvar/brain-tumor-mri-dataset"]
    transform_dataset = "vishalbakshi/hms-hbac-training-spectrogram-images"
    
    source_data_dirs = download_datasets(source_datasets)
    transform_data_dir = kagglehub.dataset_download(transform_dataset)

    print("[+]Datasets downloaded successfully")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    batch_size = 8
    num_epochs = 50
    lr = 0.0002
    
    source_loader = load_multiple_datasets(source_data_dirs, transform, batch_size)
    transform_loader = DataLoader(datasets.ImageFolder(transform_data_dir, transform=transform), batch_size=batch_size, shuffle=True)

    print("[+]Datasets loaded successfully")
    
    generator = Generator().to(device)
    critic = Critic().to(device)

    print("[+]Training begin")
    train_deepkeygen(generator, critic, source_loader, transform_loader, num_epochs, lr, device)
    print("[+]Training ended")


    
    df = pd.DataFrame(data,columns=['epochs','generator_loss','discriminator_loss'])
    df.to_csv(csvpath,index=False)
    print("[+]loss values succesfully stored in csv")
    
