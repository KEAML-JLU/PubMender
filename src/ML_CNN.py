import torch.nn as nn
import torch.nn.functional as F


class CNN_Module(nn.Module):
    
    def __init__(self, n_classes=1199, rows=350):
        super(type(self), self).__init__()
        self.conv1 = nn.Conv1d(200, 256, kernel_size=3)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=4)
        self.conv3 = nn.Conv1d(128, 96, kernel_size=5)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(7776)
        self.bn4 = nn.BatchNorm1d(2000)

        self.lin1 = nn.Linear(7776, 2000)
        self.lin2 = nn.Linear(2000, n_classes)

    def forward(self, features):
        batch_size = features.size(0)
        features = features.transpose(1,2)

        features = F.dropout(features, p=0.6)
        
        features = self.conv1(features)
        features = F.relu(features)
        features = F.dropout(features, p=0.5)
        features = F.max_pool1d(features, kernel_size=2)
        features = self.bn1(features)

        features = self.conv2(features)
        features = F.relu(features)
        features = F.dropout(features, p=0.5)
        features = F.max_pool1d(features, kernel_size=2)
        features = self.bn2(features)

        features = self.conv3(features)
        features = F.relu(features)
        features = F.dropout(features, p=0.5)
        features = features.view(batch_size, -1)
        features = self.bn3(features)

        features = self.lin1(features)
        features = F.relu(features)
        features = F.dropout(features, p=0.5)
        features = self.bn4(features)
        predict = self.lin2(features)
        return predict

class Embedding(nn.Module):
    def __init__(self, emb_model):
        super(type(self), self).__init__()
        self.n_words = emb_model.n_words
        self.embed_size = emb_model.embed_size
        self.embed_layer = emb_model.embed_layer
        self.trainable = emb_model.trainable

    def forward(self, features, normalize=True):
        """
         features is a batch_size * sent_len matrix
        """
        output = self.embed_layer(features)
        if normalize:
            output = self.__normalize_embedding(output)
        if not self.trainable:
            output = Variable(output.data)
        return output

    def __normalize_embedding(self, embed):
        output_norm = torch.norm(embed, 2, 2, keepdim=True)
        embed = embed / output_norm
        return embed

    def get_all_embedding(self):
        return self.embed_layer.weight

    def initialize_embedding(self, init_embed):
        """
         init_embed must be a np.array
        """
        assert type(init_embed) == np.ndarray
        init_embed = torch.Tensor(init_embed)
        self.embed_layer.weight.data = init_embed

    def normalize_all_embedding(self):
        embed = self.embed_layer.weight.data
        embed_norm = torch.norm(embed, 2, 1, keepdim=True)
        embed = embed / embed_norm
        self.embed_layer.weight.data = embed
