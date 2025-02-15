import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

def train_one_epoch(model, dataloader, optimizer, device, epoch_idx=0):
    model.train()
    ce_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    correct_primary = 0
    correct_secondary = 0

    for batch in tqdm(dataloader, desc=f"Train (Epoch {epoch_idx})", leave=False):
        imgs, primary_labels, pids, secondary_labels_in_local = batch

        imgs = imgs.to(device)
        primary_labels = primary_labels.to(device)
        pids = pids.to(device)
        secondary_labels_in_local = secondary_labels_in_local.to(device)

        primary_logits, feat = model(imgs)

        # Primary loss
        loss_primary = ce_loss_fn(primary_logits, primary_labels)
        pred_p = primary_logits.argmax(dim=1)
        correct_primary += (pred_p == primary_labels).sum().item()

        # Secondary loss
        batch_size = imgs.size(0)
        loss_secondary_sum = 0.0
        correct_s = 0
        for i in range(batch_size):
            pid_i = pids[i].item()
            sub_head = model.secondary_heads[str(pid_i)]
            
            feat_i = feat[i].unsqueeze(0)
            logit_i = sub_head(feat_i)
            label_i = secondary_labels_in_local[i].unsqueeze(0)
            
            loss_i = ce_loss_fn(logit_i, label_i)
            loss_secondary_sum += loss_i
            
            pred_s_i = logit_i.argmax(dim=1)
            if pred_s_i.item() == label_i.item():
                correct_s += 1

        loss_secondary = loss_secondary_sum / batch_size
        loss = loss_primary + loss_secondary

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct_secondary += correct_s
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    primary_acc = correct_primary / total_samples
    secondary_acc = correct_secondary / total_samples

    return avg_loss, primary_acc, secondary_acc


def evaluate(model, dataloader, device, epoch_idx=0):
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    correct_primary = 0
    correct_secondary = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Eval (Epoch {epoch_idx})", leave=False):
            imgs, primary_labels, pids, secondary_labels_in_local = batch

            imgs = imgs.to(device)
            primary_labels = primary_labels.to(device)
            pids = pids.to(device)
            secondary_labels_in_local = secondary_labels_in_local.to(device)

            primary_logits, feat = model(imgs)

            # Primary
            loss_primary = ce_loss_fn(primary_logits, primary_labels)
            pred_p = primary_logits.argmax(dim=1)
            correct_primary += (pred_p == primary_labels).sum().item()

            # Secondary
            batch_size = imgs.size(0)
            loss_secondary_sum = 0.0
            correct_s = 0
            for i in range(batch_size):
                pid_i = pids[i].item()
                sub_head = model.secondary_heads[str(pid_i)]
                
                feat_i = feat[i].unsqueeze(0)
                logit_i = sub_head(feat_i)
                label_i = secondary_labels_in_local[i].unsqueeze(0)
                
                loss_i = ce_loss_fn(logit_i, label_i)
                loss_secondary_sum += loss_i.item()
                
                pred_s_i = logit_i.argmax(dim=1)
                if pred_s_i.item() == label_i.item():
                    correct_s += 1

            loss_secondary = loss_secondary_sum / batch_size
            loss = loss_primary.item() + loss_secondary

            correct_secondary += correct_s
            total_loss += loss * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    primary_acc = correct_primary / total_samples
    secondary_acc = correct_secondary / total_samples

    return avg_loss, primary_acc, secondary_acc


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        # Add input validation
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an instance of nn.Module")
        if not hasattr(model, 'backbone') or not hasattr(model, 'secondary_heads'):
            raise AttributeError("Model must have 'backbone' and 'secondary_heads' attributes")
            
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Add timestamp to log_dir to prevent overwriting
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir=f"runs/hierarchical_example_{timestamp}")

        # 처음에는 backbone 파라미터 동결
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # primary_head & secondary_heads만 학습
        params_to_update = list(model.primary_head.parameters())
        for key in model.secondary_heads.keys():
            params_to_update += list(model.secondary_heads[key].parameters())

        self.optimizer = torch.optim.Adam(params_to_update, lr=config['initial_lr'])

    def train(self):
        best_val_loss = float('inf')
        epochs = self.config['epochs']

        for epoch in range(epochs):
            # Fine-tuning 시작
            if epoch == self.config['fine_tune_epoch']:
                print("==> Unfreezing the backbone and lowering LR")
                for param in self.model.backbone.parameters():
                    param.requires_grad = True
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), 
                    lr=self.config['fine_tune_lr']
                )

            # Train
            train_loss, train_p_acc, train_s_acc = train_one_epoch(
                self.model, self.train_loader, self.optimizer, 
                self.device, epoch_idx=epoch
            )

            # Evaluate
            val_loss, val_p_acc, val_s_acc = evaluate(
                self.model, self.val_loader, self.device, 
                epoch_idx=epoch
            )

            # 로깅
            print(f"[Epoch {epoch+1}/{epochs}]")
            print(f"  Train loss: {train_loss:.4f} | P-acc: {train_p_acc:.3f}, S-acc: {train_s_acc:.3f}")
            print(f"  Val   loss: {val_loss:.4f} | P-acc: {val_p_acc:.3f}, S-acc: {val_s_acc:.3f}")

            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("Acc/Primary_Train", train_p_acc, epoch)
            self.writer.add_scalar("Acc/Secondary_Train", train_s_acc, epoch)
            self.writer.add_scalar("Acc/Primary_Val", val_p_acc, epoch)
            self.writer.add_scalar("Acc/Secondary_Val", val_s_acc, epoch)

            # 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(
                    self.model.state_dict(), 
                    "checkpoints/best_hierarchical_cnn.pth"
                )
                print("  [Best model saved]")

        print("Training completed!")
        self.writer.close() 