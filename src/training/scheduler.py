from torch.optim.lr_scheduler import LambdaLR

def get_warmup_decay(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)