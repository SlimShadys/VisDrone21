import torch
import torch.nn as nn

MBVersions = {
    'MobileCountx0_5': [16, 32, 64, 128],
    'MobileCountx0_75': [32, 48, 80, 160],
    'MobileCount': [32, 64, 128, 256],
    'MobileCountx1_25': [64, 96, 160, 320],
    'MobileCountx2': [64, 128, 256, 512],
}

class CrowdCounter(nn.Module):
    def __init__(self, gpus, model_name):
        super(CrowdCounter, self).__init__()

        if model_name == 'MobileCount':
            from models.MobileCount import MobileCount as net
        elif model_name == 'MobileCountx1_25':
            from models.MobileCountx1_25 import MobileCount as net
        elif model_name == 'MobileCountx2':
            from models.MobileCountx2 import MobileCount as net

        self.CCN = net(MBVersions[model_name])

        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()

    @property
    def loss(self):
        return self.loss_mse

    def f_loss(self):
        return self.loss_mse

    def forward(self, img, gt_map):
        density_map = self.CCN(img)
        self.loss_mse = self.build_loss(density_map, gt_map)
        return density_map

    def predict(self, img):
        return self.test_forward(img)

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))

    def build_loss(self, density_map, gt_data):
        self.loss_mse = self.loss_mse_fn(density_map.squeeze(), gt_data.squeeze())
        return self.loss_mse

    def test_forward(self, img):
        density_map = self.CCN(img)
        return density_map
