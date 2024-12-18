import os
import torch
from tromr.configs import getconfig
from torch.utils.data import DataLoader
from torchvision import transforms
from tromr.dataset import CustomDataset  # Define this based on your data
from tromr.model.tromr_arch import TrOMR  # Replace with the actual model class

def train_model(config_path):
    # Load configuration
    config = getconfig(config_path)

    # Paths from config
    image_dir = config.filepaths.image_dir
    annotation_dir = config.filepaths.annotation_dir
    output_dir = config.filepaths.output_dir

    # Training parameters from config
    batch_size = config.training.batch_size
    epochs = config.training.epochs
    learning_rate = config.training.learning_rate

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to a fixed size
        transforms.ToTensor(),         # Convert to PyTorch tensors
    ])
    train_dataset = CustomDataset(image_dir, annotation_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, Optimizer
    model = TrOMR(config)  # Initialize your model architecture
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()  # Example: Use appropriate loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            images, targets = batch['image'], batch['annotations']
            images, targets = images.to('cuda'), targets.to('cuda')

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Log epoch loss
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:  # Save every 10 epochs
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")
    print(config)

if __name__ == "__main__":
    config_path = '/home/surya/Desktop/musicai/data/Polyphonic-TrOMR/config.yaml'
    train_model(config_path)
   

