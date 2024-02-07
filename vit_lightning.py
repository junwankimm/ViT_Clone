import torch
import torch.nn.functional as F
import argparse
import wandb
import os
from tqdm import tqdm

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10

from pytorch_lightning.loggers import WandbLogger


import lightning as L



class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, img_size: int = 224):
        super().__init__()
        
        emb_size = in_channels * patch_size ** 2
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))
    
    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b) # CLS Token into bat
        
        # ViT에서 Input은 최종적으로 Projection E Positional Encoding E_pos 에 대해 [CLS; x1E; x2E; ...; xNE] + E_pos로 표현됨
        x = torch.cat([cls_tokens, x], dim=1) + self.positions
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        
        self.qkv = nn.Linear(emb_size, emb_size * 3) # Q,K,V를 하나의 행렬로 
        self.attn_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
    
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # Softmax(QK^T / sqrt(d))V 가 Attention 을 계산하는 방법이니까...
        # MSA는 원래 SA에서 Head를 늘려서 여러개의 Q,K,V를 만들어서 각각의 Head에 대해 Attention을 계산하는 것
        # 아래 계산에서는 한번에 하는것처럼 보이지만 Q,K,V 안에 여러 MSA를 위한 Q,K,V가 들어있는 셈이다. 
        qkv = rearrange(self.qkv(x), 'b n (h d qkv) -> (qkv) b h n d', qkv=3, h=self.num_heads) # 편의를 위해 Q,K,V 한번에, 
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        attention = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) * (self.emb_size ** -0.5)
        attention = F.softmax(attention, dim=-1)
        attention = self.attn_drop(attention)
        
        out = torch.einsum('bhal, bhlv -> bhav', attention, values)
        out = rearrange(out, 'b h n d -> b n (h d)') # Head를 다시 합침. 여기도 사실 Concat인데, einsum을 이용해서 표현함.
        out = self.projection(out) # 원래 MSA 구조가 QKV를 통과한 후에 Projection을 통과하는 구조임.
        
        return out
    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.0):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size)
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size: int = 768, drop_p: float = 0.0, forward_expansion: int = 2, forward_drop_p: float = 0.0, **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size=emb_size, dropout=drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),  # Mean all pathces
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, emb_size: int = 768, forward_expansion: int = 2, **kwargs):
        super().__init__(
            *[TransformerEncoderBlock(emb_size=emb_size, **kwargs) for _ in range(depth)]
        )
class ViT(nn.Sequential):
    def __init__(self, in_channels: int = 3, patch_size: int = 4, forward_expansion: int = 2, img_size: int = 128, depth: int = 12, n_classes: int = 10, **kwargs):
        super().__init__(
            PatchEmbedding(in_channels=in_channels, patch_size=patch_size, img_size=img_size),
            TransformerEncoder(depth=depth, emb_size=3*patch_size**2, forward_expansion=forward_expansion, **kwargs),
            ClassificationHead(emb_size=3*patch_size**2, n_classes=n_classes)
        )
        
class ViTLightning(L.LightningModule): 
    def __init__(self, in_channels: int = 3, patch_size: int = 4, forward_expansion: int = 2, img_size: int = 128, depth: int = 12, n_classes: int = 10, **kwargs):
        super().__init__()
        self.model = ViT(in_channels=in_channels, patch_size=patch_size, forward_expansion=forward_expansion, img_size=img_size, depth=depth, n_classes=n_classes, **kwargs)
    
    def training_step(self, batch, batch_idx):
        img, target = batch
        img = self.model(img)
        loss = F.cross_entropy(img, target)
        
        # wandb.log({'train_loss' : loss})
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
        
        return [optimizer], [lr_scheduler]
    
    def validation_step(self, batch, batch_idx):
        num_correct = 0
        val_avg_loss = 0
        total = 0
        
        img, target = batch
        img = self.model(img)
        loss = F.cross_entropy(img, target)
        
        output = torch.softmax(img, dim=1)
        pred, idx_ = output.max(-1)
        num_correct += torch.eq(target, idx_).sum().item()
        total += target.size(0)
        val_avg_loss += loss
        
        # wandb.log({'val_accuracy' : num_correct / total})
        
    # def test_step(self, batch, batch_idx):
    #     img, target = batch
    #     img = self.model(img)
    #     loss = F.cross_entropy(img, target)
        
    #     wandb.log({'test_loss' : loss})
    
    # def predict_step(self, batch, batch_idx):
    #     img, target = batch
    #     img = self.model(img)
        
    #     return img
        
        
        
        
def main(args):
    
    train_set = CIFAR10(root=args.root,
                        train=True,
                        download=True,
                        transform=Compose([
                            RandomCrop(32, padding=4),
                            RandomHorizontalFlip(),
                            ToTensor(),
                            Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                        ])
    )
    
    test_set = CIFAR10(root=args.root,
                       train=False,
                       download=True,
                       transform=Compose([
                           ToTensor(),
                           Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                       ])
    )
    
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=5)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=5)
    
    model = ViT(img_size=args.img_size, n_classes=args.n_classes, patch_size=args.patch_size, forward_expansion=args.forward_expansion)
    summary(model, train_set[0][0].shape, device='cpu')
    
    os.makedirs(args.log_dir, exist_ok=True)
    wandb.init(project='ViT', name=args.name, config=args)
    print("Start Training")
    
    
    model = ViTLightning(in_channels=3, patch_size=args.patch_size, forward_expansion=args.forward_expansion, img_size=args.img_size, depth=12, n_classes=10)
    
    wandb.finish()
    wandb_logger = WandbLogger(log_model="all", project='ViT', name='lightning')
    trainer = L.Trainer(accelerator='gpu', logger=wandb_logger)
    trainer.fit(model= model, train_dataloaders=train_loader, val_dataloaders=test_loader)

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ViT')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--step_size', type=int, default=100)
    parser.add_argument('--root', type=str, default='./CIFAR10')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--name', type=str, default='./ViT_CIFAR10')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--forward_expansion', type=int, default=2)
    
    args = parser.parse_args()

    main(args)
    
    

    


