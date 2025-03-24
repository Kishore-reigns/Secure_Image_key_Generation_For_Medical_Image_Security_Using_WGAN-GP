import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os
import kagglehub

# Load datasets directly from local Kaggle folders
def load_datasets(source_data_dir, transform_data_dir, transform, batch_size):
    # Load datasets
    source_dataset = datasets.ImageFolder(source_data_dir, transform=transform)
    transform_dataset = datasets.ImageFolder(transform_data_dir, transform=transform)

    # Create DataLoaders
    source_loader = DataLoader(source_dataset, batch_size=8, shuffle=True)
    transform_loader = DataLoader(transform_dataset, batch_size=8, shuffle=True)

    return source_loader, transform_loader

# Define the Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.residual_blocks = nn.Sequential(
            *[self.residual_block(256) for _ in range(6)]
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.residual_blocks(x)
        x = self.upsample(x)
        return x

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)
        )
        self.fc = nn.Linear(13 * 13, 1)  # Fully connected layer to output a single scalar value

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)  # Apply the fully connected layer
        return x 
# Training Function
def train_deepkeygen(generator, discriminator, source_loader, transform_loader, num_epochs=100, lr=0.0002, device="cpu"):
    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        print("epoch : ", epoch)
        for (source_imgs, _), (transform_imgs, _) in zip(source_loader, transform_loader):
            # Ensure that both batches have the same size
            if source_imgs.size(0) != transform_imgs.size(0):
                min_batch_size = min(source_imgs.size(0), transform_imgs.size(0))
                source_imgs = source_imgs[:min_batch_size]
                transform_imgs = transform_imgs[:min_batch_size]

            source_imgs, transform_imgs = source_imgs.to(device), transform_imgs.to(device)

            # Train Discriminator
            real_labels = torch.ones((source_imgs.size(0), 1)).to(device)
            fake_labels = torch.zeros((source_imgs.size(0), 1)).to(device)

            optimizer_d.zero_grad()
            real_outputs = discriminator(transform_imgs)
            d_real_loss = criterion(real_outputs, real_labels)

            fake_imgs = generator(source_imgs)
            fake_outputs = discriminator(fake_imgs.detach())
            d_fake_loss = criterion(fake_outputs, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            fake_outputs = discriminator(fake_imgs)
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss D: {d_loss.item()}, Loss G: {g_loss.item()}")

    return generator


# Main script
if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("current working directory : ", os.getcwd())

    # Parameters
    print("downloading source domain")
    source_data_dir = kagglehub.dataset_download("raddar/tuberculosis-chest-xrays-montgomery")
    print("downloading Transformation domain")
    transform_data_dir = kagglehub.dataset_download("pankajkumar2002/random-image-sample-dataset")  # Directory for transformation domain images
    batch_size = 8
    num_epochs = 20
    lr = 0.0002
    print("Download complete \nProcessing Starts")
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    print("Loading datasets")
    # Load datasets
    source_loader, transform_loader = load_datasets(source_data_dir, transform_data_dir, transform, batch_size)
    print("datasets Loaded\nModel Initialzation")
    # Model initialization
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    print("Training the model")
    # Train the model
    trained_generator = train_deepkeygen(generator, discriminator, source_loader, transform_loader, num_epochs, lr, device)
    print("Training completed sucessfully")
    # Save the model
    torch.save(trained_generator.state_dict(), "deepkeygen_generator.pth")
    print("saved")
