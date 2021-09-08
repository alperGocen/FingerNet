import torch
from torch import nn


class CenterLoss(nn.Module):
    """Center loss.

       Reference:
       Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

       Args:
         num_classes (int): number of classes.
         feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=99, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, feat_matrix, labels):
        """
        Args:
            feat_matrix: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = feat_matrix.size(0)
        dist_mat = torch.pow(feat_matrix, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
            torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        dist_mat.addmm_(1, -2, feat_matrix, self.centers.t())

        classes = torch.arange(self.num_classes).long()

        if self.use_gpu:
            classes = classes.cuda()

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = dist_mat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

