import torch
from sklearn.datasets import make_blobs
import torch.nn as nn
import torch.optim as optim

# 3 Layer --> 2 Layer --> 3 Layer
class Autoencoder(nn.Module):
    def __init__(self, input_size=3, bottleneck_size=2, output_size=3):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, bottleneck_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, output_size),  
            nn.ReLU(),  
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def loss_function(self, x, x_hat):
        return nn.MSELoss()(x_hat, x)

if __name__ == "__main__":
    # Generate synthetic blob data
    X_blob, _ = make_blobs(n_samples=100, n_features=3, centers=5, random_state=42)
    X_tensor = torch.FloatTensor(X_blob)
    
    # Example usage
    model = Autoencoder()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training step
    optimizer.zero_grad()
    output_data = model(X_tensor)
    loss = model.loss_function(X_tensor, output_data)
    loss.backward()
    optimizer.step()
    
    print(f"Input shape: {X_tensor.shape}")
    print(f"Output shape: {output_data.shape}")
    print(f"Loss: {loss.item()}")