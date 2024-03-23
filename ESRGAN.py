import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm

# Define the ESRGAN model
class ESRGANGenerator(nn.Module):
    def __init__(self, scale_factor=4):
        super(ESRGANGenerator, self).__init__()
        self.scale_factor = scale_factor
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),

            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),

            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),

            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Define the discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),

            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),

            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),

            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),

            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),

            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),

            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),

            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.main(x)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Initialize the generator and discriminator
generator = ESRGANGenerator().to(device)
discriminator = Discriminator().to(device)

# Define the loss functions and optimizers
criterion_mse = nn.MSELoss()
criterion_bce = nn.BCEWithLogitsLoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.001)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for i, (images, _) in loop:
        images = images.to(device)
        
        # Downsample the images
        downsampled_images = nn.functional.interpolate(images, scale_factor=0.25, mode="bilinear", align_corners=False)
        
        # Train the discriminator
        discriminator.zero_grad()
        real_output = discriminator(images)
        real_label = torch.ones_like(real_output, device=device)
        real_loss = criterion_bce(real_output, real_label)
        
        fake_images = generator(downsampled_images)
        fake_output = discriminator(fake_images.detach())
        fake_label = torch.zeros_like(fake_output, device=device)
        fake_loss = criterion_bce(fake_output, fake_label)
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_d.step()
        
        # Train the generator
        generator.zero_grad()
        fake_images = generator(downsampled_images)
        fake_output = discriminator(fake_images)
        
        pixel_loss = criterion_mse(fake_images, images)
        adversarial_loss = criterion_bce(fake_output, real_label)
        
        g_loss = (pixel_loss +  adversarial_loss)/2
        g_loss.backward()
        optimizer_g.step()
        
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}]")
        loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())
    
    torch.save(generator.state_dict(), "esrgan_generator.pth")
    torch.save(discriminator.state_dict(), "esrgan_discriminator.pth")