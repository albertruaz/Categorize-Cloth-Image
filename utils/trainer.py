import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train_one_epoch(model, dataloader, optimizer, device, epoch_idx=0):
    # ... (기존 train_one_epoch 함수 코드)
    pass

def evaluate(model, dataloader, device, epoch_idx=0):
    # ... (기존 evaluate 함수 코드)
    pass

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.writer = SummaryWriter(log_dir="runs/hierarchical_example")
        
    def train(self):
        # 처음에는 backbone 파라미터 동결
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # primary_head & secondary_heads만 학습
        params_to_update = list(self.model.primary_head.parameters())
        for key in self.model.secondary_heads.keys():
            params_to_update += list(self.model.secondary_heads[key].parameters())

        optimizer = torch.optim.Adam(params_to_update, lr=self.config['initial_lr'])
        best_val_loss = float('inf')

        for epoch in range(self.config['total_epochs']):
            if epoch == self.config['fine_tune_epoch']:
                print("==> Unfreezing the backbone and lowering LR")
                for param in self.model.backbone.parameters():
                    param.requires_grad = True
                optimizer = torch.optim.Adam(self.model.parameters(), 
                                           lr=self.config['fine_tune_lr'])

            train_loss, train_p_acc, train_s_acc = train_one_epoch(
                self.model, self.train_loader, optimizer, self.device, epoch)
            
            val_loss, val_p_acc, val_s_acc = evaluate(
                self.model, self.val_loader, self.device, epoch)

            # 로깅
            self._log_metrics(epoch, train_loss, val_loss, 
                            train_p_acc, train_s_acc, val_p_acc, val_s_acc)

            # 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_hierarchical_cnn.pth")
                print("  [Best model saved]")

        print("Training done!")
        self.writer.close()

    def _log_metrics(self, epoch, train_loss, val_loss, 
                    train_p_acc, train_s_acc, val_p_acc, val_s_acc):
        print(f"[Epoch {epoch+1}/{self.config['total_epochs']}]")
        print(f"  Train loss: {train_loss:.4f} | P-acc: {train_p_acc:.3f}, S-acc: {train_s_acc:.3f}")
        print(f"  Val   loss: {val_loss:.4f}   | P-acc: {val_p_acc:.3f}, S-acc: {val_s_acc:.3f}")

        self.writer.add_scalar("Loss/Train", train_loss, epoch)
        self.writer.add_scalar("Loss/Val", val_loss, epoch)
        self.writer.add_scalar("Acc/Primary_Train", train_p_acc, epoch)
        self.writer.add_scalar("Acc/Secondary_Train", train_s_acc, epoch)
        self.writer.add_scalar("Acc/Primary_Val", val_p_acc, epoch)
        self.writer.add_scalar("Acc/Secondary_Val", val_s_acc, epoch) 