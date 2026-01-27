import os
from collections import OrderedDict
import torch
import torch.nn as nn
from lightning import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.train_utils import make_optimizer


class Base_Module(LightningModule):
    def __init__(self, args):
        super(Base_Module, self).__init__()

        self.args = args

    def init_weights(self, pretrained):
        if pretrained is not None and os.path.exists(pretrained):
            print("=> loading pretrained model {}".format(pretrained))
            pretrained_dict = torch.load(pretrained, map_location="cpu", weights_only=False)
            pretrained_dict = pretrained_dict["state_dict"]

            model_dict = self.state_dict()
            print('Model dict: ', model_dict.keys())
            available_pretrained_dict = {}

            for k, v in pretrained_dict.items():
                print('Pretrained dict: ', k)
                if k in model_dict.keys():
                    if pretrained_dict[k].shape == model_dict[k].shape:
                        available_pretrained_dict[k] = v
                    else:
                        print(pretrained_dict[k].shape)
                        print(model_dict[k].shape)
                if k[7:] in model_dict.keys():
                    if pretrained_dict[k].shape == model_dict[k[7:]].shape:
                        available_pretrained_dict[k[7:]] = v
                k_m = k.replace("encoder.", "")
                if k_m in model_dict.keys():
                    if pretrained_dict[k].shape == model_dict[k_m].shape:
                        available_pretrained_dict[k_m] = v
                k_a = 'model.' + k
                if k_a in model_dict.keys():
                    if pretrained_dict[k].shape == model_dict[k_a].shape:
                        available_pretrained_dict[k_a] = v
                k_a_e = 'model.swinViT.' + k
                if k_a_e in model_dict.keys():
                    if pretrained_dict[k].shape == model_dict[k_a_e].shape:
                        available_pretrained_dict[k_a_e] = v
                    else:
                        print(pretrained_dict[k].shape)
                        print(model_dict[k_a_e].shape)
                k_dec = k.replace("decoder_modules.{}.".format(self.args.task), "")
                if k_dec in model_dict.keys():
                    if pretrained_dict[k].shape == model_dict[k_dec].shape:
                        available_pretrained_dict[k_dec] = v
                    else:
                        print("Decoder mismatch")
                        print(pretrained_dict[k].shape)
                        print(model_dict[k_dec].shape)

            for k, _ in available_pretrained_dict.items():
                print("loading {}".format(k))
            model_dict.update(available_pretrained_dict)
            self.load_state_dict(model_dict, strict=True)
        else:
            print("=> No pretrained model found at {}".format(pretrained))

    def load_weights(self, path):
        if os.path.isfile(path):
            print("=> Loading model from {}".format(path))
            checkpoint = torch.load(path, map_location="cpu")
            try:
                state_dict = checkpoint["state_dict"]
            except KeyError:
                state_dict = checkpoint["model"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "module." in k:
                    name = k[7:]  # remove `module.`
                else:
                    name = k
                print(name)
                new_state_dict[name] = v
            current_state = self.state_dict()
            filtered_state = OrderedDict()
            skipped_keys = []
            for key, value in new_state_dict.items():
                if key not in current_state:
                    skipped_keys.append(key)
                    continue
                if current_state[key].shape != value.shape:
                    skipped_keys.append(key)
                    continue
                filtered_state[key] = value

            if skipped_keys:
                print("=> Skipped loading {} keys (missing or shape mismatch)".format(len(skipped_keys)))
                for key in skipped_keys:
                    print("   - {}".format(key))

            current_state.update(filtered_state)
            self.load_state_dict(current_state, strict=False)
            print("=> trained model loaded")

    def initialize(self, decoder, pred):
        if decoder is not None:
            for m in decoder.modules():

                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(
                        m.weight, mode="fan_in", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        if pred is not None:
            for m in pred.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def configure_optimizers(self):
        optim, kwargs_optimizer = make_optimizer(args=self.args)
        opt = optim(self.parameters(), **kwargs_optimizer)
        scheduler = CosineAnnealingLR(opt, T_max=self.args.epoch)

        return [opt], [scheduler]