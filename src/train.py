"""PyTorch constrained unlearning step."""
import torch
import torch.nn.functional as F

def unlearning_step(model, teacher, forget_batch, retain_batch, optimizer, alpha=1.0, beta=1.0):
    model.train(); optimizer.zero_grad()
    forget_x, forget_y = forget_batch; retain_x, _ = retain_batch
    forget_loss = F.cross_entropy(model(forget_x), forget_y)
    with torch.no_grad(): teacher_logits = teacher(retain_x)
    retain_loss = F.kl_div(F.log_softmax(model(retain_x), -1), F.softmax(teacher_logits, -1), reduction='batchmean')
    loss = -alpha * forget_loss + beta * retain_loss
    loss.backward(); optimizer.step()
    return {'forget_loss': float(forget_loss.detach()), 'retain_distillation': float(retain_loss.detach())}
