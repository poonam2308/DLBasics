import torch
import torch.nn.functional as F


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        # Ensure the output and target tensors are of the same shape
        outputs = F.softmax(outputs, dim=1)

        # Convert targets to one-hot encoding
        targets = targets.long()
        targets = F.one_hot(targets, num_classes=outputs.size(1))
        targets = targets.permute(0, 3, 1, 2).float()  # Change shape to (batch_size, num_classes, H, W)

        intersection = torch.sum(outputs * targets, dim=(2, 3))
        union = torch.sum(outputs, dim=(2, 3)) + torch.sum(targets, dim=(2, 3))
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - torch.mean(dice_score)  # Dice loss is 1 - Dice score


# Example of using Dice Loss
dice_loss = DiceLoss()
