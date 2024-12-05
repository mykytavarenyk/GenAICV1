import torch
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torchvision


def train_model(model, train_loader, val_loader, optimizer, num_epochs, device, model_name, logger=None, save_best=True, denorm_params=None, model_update_fn=None, generate_images_flag=False):
    model.to(device)
    best_val_loss = float('inf')

    # Create directories for checkpoints and logs
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs(f'logs/{model_name}', exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f'logs/{model_name}') if logger is None else logger

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(num_epochs):
        model.train()

        # Perform model-specific updates if provided
        if model_update_fn is not None:
            model_update_fn(model, epoch)

        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            images, _ = batch
            images = images.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = model.loss_function(outputs, images)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            progress_bar.set_postfix({'Loss': loss.item()})

        # Calculate average training loss
        train_loss /= len(train_loader.dataset)

        # Validate the model
        val_loss = validate_model(model, val_loader, device)

        # Log losses
        writer.add_scalar('Loss/Train', train_loss, epoch+1)
        writer.add_scalar('Loss/Validation', val_loss, epoch+1)

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}')

        # Learning rate scheduler step
        scheduler.step(val_loss)

        # Save the best model
        if save_best and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'checkpoints/{model_name}_best.pth')
            print(f'Best model saved with validation loss: {best_val_loss:.6f}')

            # Save reconstructed and generated images
            save_reconstructed_images(
                model, images, outputs, epoch, model_name, writer, is_best=True,
                denorm_params=denorm_params, generate_images_flag=generate_images_flag, device=device
            )

        # Save reconstructed and generated images every epoch
        save_reconstructed_images(
            model, images, outputs, epoch, model_name, writer, is_best=False,
            denorm_params=denorm_params, generate_images_flag=generate_images_flag, device=device
        )

    writer.close()


def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images, _ = batch
            images = images.to(device)
            outputs = model(images)
            loss = model.loss_function(outputs, images)
            val_loss += loss.item() * images.size(0)
    val_loss /= len(val_loader.dataset)
    return val_loss


def save_reconstructed_images(model, images, outputs, epoch, model_name, writer, is_best=False, denorm_params=None, generate_images_flag=False, device='cpu'):
    model.eval()
    with torch.no_grad():
        # Save reconstructed images
        reconstructed = model.generate_images(outputs)
        num_images = min(images.size(0), 16)
        orig_images = images[:num_images].cpu()
        recon_images = reconstructed[:num_images].cpu()

        if denorm_params is not None:
            # Denormalize images for visualization
            mean = torch.tensor(denorm_params['mean']).view(1, -1, 1, 1)
            std = torch.tensor(denorm_params['std']).view(1, -1, 1, 1)
            orig_images = orig_images * std + mean
            recon_images = recon_images * std + mean

        # Clamp images to [0, 1]
        orig_images = torch.clamp(orig_images, 0, 1)
        recon_images = torch.clamp(recon_images, 0, 1)

        # Concatenate images side by side
        comparison = torch.cat([orig_images, recon_images], dim=3)  # Concatenate along width
        # Make a grid of images
        img_grid = torchvision.utils.make_grid(comparison, nrow=4, normalize=False, pad_value=1)
        # Save the grid to TensorBoard
        writer.add_image('Reconstructed Images', img_grid, epoch+1)
        # Optionally save the image to disk
        if is_best:
            os.makedirs(f'images/{model_name}', exist_ok=True)
            torchvision.utils.save_image(img_grid, f'images/{model_name}/reconstructions_epoch_{epoch+1}.png')

        # Generate new images if flag is set
        if generate_images_flag:
            generated_images = model.generate_images(outputs=None, num_samples=16, device=device)
            if denorm_params is not None:
                generated_images = generated_images.cpu() * std + mean
            generated_images = torch.clamp(generated_images, 0, 1)
            # Make a grid of generated images
            gen_img_grid = torchvision.utils.make_grid(generated_images, nrow=4, normalize=False, pad_value=1)
            # Save the grid to TensorBoard
            writer.add_image('Generated Images', gen_img_grid, epoch+1)
            # Optionally save the image to disk
            if is_best:
                torchvision.utils.save_image(gen_img_grid, f'images/{model_name}/generated_epoch_{epoch+1}.png')

    model.train()
