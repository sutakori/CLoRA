import torch
import torch.nn as nn
import torch.nn.functional as F

# import logging
# logger = logging.getLogger(__name__)

class CLoraWrapper(nn.Module):
    def __init__(self, peft_model, k, adapter_name='default', lambda_=1):
        super().__init__()
        self.peft_model = peft_model
        self.adapter_name = adapter_name

        self.lambda_ = lambda_

        self.k = k
        self.prepare_regulariation_matrix()

    def prepare_regulariation_matrix(self):
        self.u = nn.ParameterDict({})
        self.v = nn.ParameterDict({})

        for n, v_ in self.peft_model.model.named_modules():
            if not self.peft_model._check_target_module_exists(self.peft_model.peft_config[self.adapter_name], n):
                continue

            v = v_.weight
            # output, k
            self.u[n.replace('.', '-')] = v.new_zeros(v.size(0), self.k) # parameter dict do not allow . in key
            # input, k
            self.v[n.replace('.', '-')] = v.new_zeros(v.size(1), self.k)

        for k, v in self.u.items():
            nn.init.orthogonal_(v)
        for k, v in self.v.items():
            nn.init.orthogonal_(v)

        self.u.requires_grad_(False)
        self.v.requires_grad_(False)

    def compute_reg_loss(self, peft_key, p):
        # input/output, s
        peft_key_ = peft_key.replace('.', '-') # parameter dict do not allow . in key
        target_uv = self.u if 'lora_B' in peft_key_ else self.v

        reg_p = None

        for k, v in target_uv.items():
            if k in peft_key_:
                reg_p = v
                break

        if 'lora_A' in peft_key_:
            # p: r, input
            # reg_p: input, k
            reg_matrix = torch.matmul(p, reg_p)
            reg_loss = torch.norm(reg_matrix, p='fro')**2 / 2
        elif 'lora_B' in peft_key_:
            # p: output, r
            # reg_p: output, k
            reg_matrix = torch.matmul(p.T, reg_p)
            reg_loss = torch.norm(reg_matrix, p='fro')**2 / 2
        else:
            raise NotImplementedError

        return reg_loss

    def forward(self, *args, **kwargs):
        ret = self.peft_model(*args, **kwargs)

        if not self.training:
            return ret

        loss_lm = ret[0]
        ret = ret[1:]

        loss_reg = 0

        reg_paras = [(n,p) for n,p in self.peft_model.named_parameters()
                     if any([t in n for t in self.peft_model.peft_config['default'].target_modules]) and 'lora' in n]

        for n, p in reg_paras:
            loss_reg += self.compute_reg_loss(n, p)

        # logger.info(f'rank {os.environ.get("LOCAL_RANK", 0)} - lm_loss: {loss_lm}, reg_loss: {loss_reg}')

        loss = loss_lm + loss_reg * self.lambda_

        return (loss,) + ret

    def save_pretrained(self, *args, **kwargs):
        merge = kwargs.pop('merge', True)
        if merge:
            merged_model = self.peft_model.merge_and_unload()
            merged_model.save_pretrained(*args, **kwargs)
        else:
            self.peft_model.save_pretrained(*args, **kwargs)
