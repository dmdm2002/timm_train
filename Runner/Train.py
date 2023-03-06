import os
import re
import timm
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from Utils.Options import Param
from Utils.DataSet import CustomDataset
from Utils.Displayer import displayer
# from Model.PyramidViG import Pyramid_ViG
# from utils.DataLoader import Loader
# from utils.Displayer import displayer

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class trainer(Param):
    def __init__(self):
        super(trainer, self).__init__()
        os.makedirs(self.OUTPUT_CKP, exist_ok=True)
        os.makedirs(self.OUTPUT_LOSS, exist_ok=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.disp_tr = displayer(["train_acc, train_loss"])
        self.disp_te = displayer(["val_acc, val_loss"])

    def init_weight(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            nn.init.constant(module.bias.data, 0.0)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant(module.bias.data, 0.0)

        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            nn.init.constant(module.bias.data, 0.0)

    def run(self):
        print('--------------------------------------------------')
        print(f'[DEVICE] : {self.device}')
        print('--------------------------------------------------')

        model = timm.create_model('convnext_base', pretrained=True, num_classes=2)

        if self.CKP_LOAD:
            ckp = torch.load(f'{self.OUTPUT_CKP}', map_location=self.device)
            model.load_state_dict(ckp["model_state_dict"])
            epoch = ckp["epoch"] + 1
        else:
            # PyramidViG.apply(self.init_weight)
            epoch = 0

        model.train()

        transform = transforms.Compose(
            [
                transforms.Resize((self.SIZE, self.SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        tr_dataset = CustomDataset(self.DATASET_PATH, self.DATA_STYPE[0], self.DATA_CLS, transform)
        te_dataset = CustomDataset(self.DATASET_PATH, self.DATA_STYPE[1], self.DATA_CLS, transform)

        criterion_CEL = nn.CrossEntropyLoss(label_smoothing=0.1)

        summary = SummaryWriter()

        optim_Adam = optim.Adam(list(model.parameters()), lr=self.LR)

        for ep in range(epoch, self.EPOCH):
            tr_dataloader = DataLoader(dataset=tr_dataset, batch_size=self.BATCHSZ, shuffle=True)
            te_dataloader = DataLoader(dataset=te_dataset, batch_size=self.BATCHSZ, shuffle=True)

            # training Loop
            for idx, (item, label) in enumerate(tqdm.tqdm(tr_dataloader, desc=f'TRAINING EPOCH [{ep} / {self.EPOCH}]')):
                item = item.to(self.device)
                label = label.to(self.device)

                output = model(item)
                loss_CEL = criterion_CEL(output, label)

                """
                loss average 값 구하는 display function
                """
                self.disp_tr.cal_accuray(output, label)
                self.disp_tr.record_loss(loss_CEL)

                optim_Adam.zero_grad()
                loss_CEL.backward()
                optim_Adam.step()

            # PyramidViG.eval()
            with torch.no_grad:
                model.eval()
                for idx, (item, label) in enumerate(tqdm.tqdm(te_dataloader, desc=f'TESTING EPOCH [{ep} / {self.EPOCH}]')):
                    item = item.to(self.device)
                    label = label.to(self.device)

                    output = model(item)
                    loss = criterion_CEL(output, label)

                    self.disp_te.cal_accuray(output, label)
                    self.disp_te.record_loss(loss)

            tr_item_list = self.disp_tr.get_avg(len(tr_dataset), len(tr_dataloader))
            te_item_list = self.disp_te.get_avg(len(te_dataset), len(te_dataloader))

            print(f"===> EPOCH[{ep}/{self.EPOCH}] || train acc : {tr_item_list[0]}   |   train loss : {tr_item_list[1]}"
                  f"   |   test acc : {te_item_list[0]}   |   test loss : {te_item_list[1]} ||")

            summary.add_scalar("train/acc", tr_item_list[0], ep)
            summary.add_scalar("train/loss", tr_item_list[1], ep)

            summary.add_scalar("test/acc", te_item_list[0], ep)
            summary.add_scalar("test/loss", te_item_list[1], ep)

            self.disp_tr.reset()
            self.disp_te.reset()

            torch.save(
                {
                    "PyramidViG_state_dict": model.state_dict(),
                    "optim_AdamW_state_dict": optim_Adam.state_dict(),
                    "epoch": ep,
                },
                os.path.join(f"{self.OUTPUT_CKP}/ckp", f"{epoch}.pth"),
            )