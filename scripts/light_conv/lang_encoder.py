import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
class EmbeddingLayer(nn.Module):
    def __init__(self, num_classes, embedding_dim=3):
        super(EmbeddingLayer, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_classes, embedding_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        print("lan_embedding shape: ",embedded.shape)
        return embedded

class LightweightConvLayer(nn.Module):
    def __init__(self, num_classes, embedding_dim=3):
        super(LightweightConvLayer, self).__init__()
        self.conv = nn.Conv2d(embedding_dim, num_classes, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(num_classes, 1)
    
    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    # Test EmbeddingLayer
    language_seg_paths = ["./data/Replica/room0/language_features/rgb_300_s.npy"]
    language_fea_paths = [f"./data/Replica/room0/language_features/rgb_{i}_f.npy" for i in range(0,200)]
    print(language_fea_paths[0])
    feature_level = 3
    seg_map1 = torch.from_numpy(np.load(language_seg_paths[0]))
    feature_map = torch.from_numpy(np.load(language_fea_paths[0]))
    seg_map = seg_map1[feature_level:feature_level+1].squeeze(0)
    mask = seg_map != -1
    
    for path in language_fea_paths:
        map = torch.from_numpy(np.load(path))
        print(map.shape)

    plt.imshow(seg_map)
    plt.show()
    print(seg_map.shape)
    print(feature_map.shape)
    
    ## count number of -1 in segmap
    print(torch.sum(seg_map == -1))
    
    embedding_dim = 3
    num_classes = feature_map.shape[0]
    embedding_layer = EmbeddingLayer(num_classes, embedding_dim)
    decoder = LightweightConvLayer(num_classes, embedding_dim)
    embedding_lang = torch.zeros((*seg_map.shape,embedding_dim))
    embedding_lang[mask] = embedding_layer(seg_map[mask].long())
    embedding_lang = embedding_lang.permute(2,0,1)
    embedding_lang = embedding_lang.float()
    
    
    print(embedding_lang.shape)
    predicted_semantic_map = decoder(embedding_lang.unsqueeze(0)).squeeze(0)
    print(predicted_semantic_map.shape)
    # Reshape predicted_semantic_map for loss computation
    predicted_semantic_map_flat = predicted_semantic_map.permute(1, 2, 0)[mask]
    print(predicted_semantic_map_flat.shape)
    # Compute loss
    loss = F.cross_entropy(predicted_semantic_map_flat, seg_map[mask].long(), reduction='sum')

    print(f"Input shape: {seg_map.shape}")
    print(f"Embedding shape: {embedding_lang.shape}")
    print(f"Predicted semantic map shape: {predicted_semantic_map.shape}")
    print(f"Loss: {loss.item()}")