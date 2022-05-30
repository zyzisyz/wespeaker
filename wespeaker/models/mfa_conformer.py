import torch
import torch.nn as nn
import torch.nn.functional as F
import wespeaker.models.pooling_layers as pooling_layers

from wespeaker.models.transformer.encoder import ConformerEncoder
from wespeaker.models.transformer.bn import BatchNorm1d
import wespeaker.models.pooling_layers as pooling_layers

class Conformer(torch.nn.Module):
    def __init__(self, n_mels=80, num_blocks=6, output_size=256, embedding_dim=192, input_layer="conv2d2", 
            pos_enc_layer_type="rel_pos"):

        super(Conformer, self).__init__()
        print("input_layer: {}".format(input_layer))
        print("pos_enc_layer_type: {}".format(pos_enc_layer_type))
        self.conformer = ConformerEncoder(input_size=n_mels, num_blocks=num_blocks, 
                output_size=output_size, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type)
        self.pooling = pooling_layers.ASTP(in_dim=output_size*num_blocks)
        self.bn = BatchNorm1d(input_size=output_size*num_blocks*2)
        self.fc = torch.nn.Linear(output_size*num_blocks*2, embedding_dim)
    
    def forward(self, feat):
        lens = torch.ones(feat.shape[0]).to(feat.device)
        lens = torch.round(lens*feat.shape[1]).int()
        x, masks = self.conformer(feat, lens)
        x = x.permute(0, 2, 1)
        x = self.pooling(x)
        x = self.bn(x)
        x = self.fc(x)
        x = x.squeeze(1)
        return x

def MFA_Conformer6(feat_dim, embed_dim, pooling_func='TSTP'):
    model = Conformer()
    return model

 
if __name__ == '__main__':
    model = MFA_Conformer6(feat_dim=80, embed_dim=512, pooling_func='TSTP')
    y = model(torch.rand(2, 200, 80))
    print(y.shape)

    num_params = sum(p.numel() for p in model.parameters())
    print("{} M".format(num_params / 1e6))


