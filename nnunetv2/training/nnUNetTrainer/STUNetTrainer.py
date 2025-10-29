from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
from torch import nn
import torch.nn.functional as F
from zeta.nn import SSM
from einops import rearrange
import numpy as np
from torch import Tensor
import os
import sys
import time
import copy
import random
import string
def generate_random_id(length=10):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

class STUNetTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000
        self.initial_lr = 1e-4

    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        kernel_sizes = [[3, 3, 3]] * 6
        strides = configuration_manager.pool_op_kernel_sizes[1:]
        if len(strides) > 5:
            strides = strides[:5]
        while len(strides) < 5:
            strides.append([1, 1, 1])
        return STUNet(num_input_channels, num_classes, depth=[1] * 6, dims=[32 * x for x in [1, 2, 4, 8, 16, 16]],
                      pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes,
                      enable_deep_supervision=enable_deep_supervision)
class STUNetTrainer_small_prompt(STUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        kernel_sizes = [[3, 3, 3]] * 6
        strides = configuration_manager.pool_op_kernel_sizes[1:]
        if len(strides) > 5:
            strides = strides[:5]
        while len(strides) < 5:
            strides.append([1, 1, 1])
        return STUNet_prompt(num_input_channels, num_classes, depth=[1] * 6, dims=[16 * x for x in [1, 2, 4, 8, 16, 16]],
                      pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes,
                      enable_deep_supervision=enable_deep_supervision)
class STUNetTrainer_small_prompt_pretrain(STUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        kernel_sizes = [[3, 3, 3]] * 6
        strides = configuration_manager.pool_op_kernel_sizes[1:]
        if len(strides) > 5:
            strides = strides[:5]
        while len(strides) < 5:
            strides.append([1, 1, 1])
        model=STUNet_prompt_pretrain(num_input_channels, num_classes, depth=[1] * 6, dims=[16 * x for x in [1, 2, 4, 8, 16, 16]],
                      pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes,
                      enable_deep_supervision=enable_deep_supervision)
        model.load_pretrained_encoders(
            ct_ckpt_path='/projects/whole_body_PET_CT_segmentation/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset226_uni_seg/STUNetTrainer_small_VIT_cat__nnUNetPlans__3d_fullres/pretrain_1000/best_encoder_ct_epoch_94.pth',
            pet_ckpt_path='/projects/whole_body_PET_CT_segmentation/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset226_uni_seg/STUNetTrainer_small_VIT_cat__nnUNetPlans__3d_fullres/pretrain_1000/best_encoder_pet_epoch_94.pth',
            map_location='cuda'  #
        )
        return model

class STUNet(nn.Module):
    def __init__(self, input_channels, num_classes, depth=[1, 1, 1, 1, 1, 1], dims=[32, 64, 128, 256, 512, 512],
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None, enable_deep_supervision=True):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.final_nonlin = lambda x: x
        self.decoder = Decoder()
        self.decoder.deep_supervision = enable_deep_supervision
        self.upscale_logits = False

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        num_pool = len(pool_op_kernel_sizes)

        assert num_pool == len(dims) - 1

        # encoder
        self.conv_blocks_context = nn.ModuleList()
        stage = nn.Sequential(
            BasicResBlock(input_channels, dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0], use_1x1conv=True),
            *[BasicResBlock(dims[0], dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0]) for _ in
              range(depth[0] - 1)])
        self.conv_blocks_context.append(stage)
        for d in range(1, num_pool + 1):
            stage = nn.Sequential(BasicResBlock(dims[d - 1], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d],
                                                stride=self.pool_op_kernel_sizes[d - 1], use_1x1conv=True),
                                  *[BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d])
                                    for _ in range(depth[d] - 1)])
            self.conv_blocks_context.append(stage)

        # upsample_layers
        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            upsample_layer = Upsample_Layer_nearest(dims[-1 - u], dims[-2 - u], pool_op_kernel_sizes[-1 - u])
            self.upsample_layers.append(upsample_layer)

        # decoder
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool):
            stage = nn.Sequential(BasicResBlock(dims[-2 - u] * 2, dims[-2 - u], self.conv_kernel_sizes[-2 - u],
                                                self.conv_pad_sizes[-2 - u], use_1x1conv=True),
                                  *[BasicResBlock(dims[-2 - u], dims[-2 - u], self.conv_kernel_sizes[-2 - u],
                                                  self.conv_pad_sizes[-2 - u]) for _ in range(depth[-2 - u] - 1)])
            self.conv_blocks_localization.append(stage)

        # outputs
        self.seg_outputs = nn.ModuleList()
        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(nn.Conv3d(dims[-2 - ds], num_classes, kernel_size=1))

        self.upscale_logits_ops = []
        for usl in range(num_pool - 1):
            self.upscale_logits_ops.append(lambda x: x)

    def forward(self, x):
        skips = []
        seg_outputs = []

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.conv_blocks_localization)):
            x = self.upsample_layers[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self.decoder.deep_supervision:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]

class STUNet_prompt(nn.Module):
    def __init__(self,
                 input_channels,
                 num_classes,
                 depth=[1, 1, 1, 1, 1, 1],
                 dims=[32, 64, 128, 256, 512, 512],
                 pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 enable_deep_supervision=True):
        super().__init__()

        self.conv_op = nn.Conv3d
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.final_nonlin = lambda x: x
        self.decoder = Decoder()
        self.decoder.deep_supervision = enable_deep_supervision
        self.upscale_logits = False

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        num_pool = len(pool_op_kernel_sizes)
        assert num_pool == len(dims) - 1

        # ---------------------------
        # Encoder
        # ---------------------------
        self.conv_blocks_context = nn.ModuleList()
        stage0 = nn.Sequential(
            BasicResBlock(input_channels, dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0], use_1x1conv=True),
            *[
                BasicResBlock(dims[0], dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0])
                for _ in range(depth[0] - 1)
            ]
        )
        self.conv_blocks_context.append(stage0)
        for d in range(1, num_pool + 1):
            stage = nn.Sequential(
                BasicResBlock(
                    dims[d - 1],
                    dims[d],
                    self.conv_kernel_sizes[d],
                    self.conv_pad_sizes[d],
                    stride=self.pool_op_kernel_sizes[d - 1],
                    use_1x1conv=True
                ),
                *[
                    BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d])
                    for _ in range(depth[d] - 1)
                ]
            )
            self.conv_blocks_context.append(stage)

        # ---------------------------
        # Upsample layers
        # ---------------------------
        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            if u == 0:
                upsample_layer = Upsample_Layer_nearest(dims[-1 - u] + 2, dims[-2 - u], pool_op_kernel_sizes[-1 - u])
            else:
                upsample_layer = Upsample_Layer_nearest(dims[-1 - u], dims[-2 - u], pool_op_kernel_sizes[-1 - u])
            self.upsample_layers.append(upsample_layer)

        # ---------------------------
        # Decoder blocks (localization)
        # 前 num_pool-1 个阶段采用共享模块
        # ---------------------------
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool - 1):
            stage = nn.Sequential(
                BasicResBlock(dims[-2 - u] * 2, dims[-2 - u],
                              self.conv_kernel_sizes[-2 - u],
                              self.conv_pad_sizes[-2 - u],
                              use_1x1conv=True),
                *[
                    BasicResBlock(dims[-2 - u], dims[-2 - u],
                                  self.conv_kernel_sizes[-2 - u],
                                  self.conv_pad_sizes[-2 - u])
                    for _ in range(depth[-2 - u] - 1)
                ]
            )
            self.conv_blocks_localization.append(stage)

        # ---------------------------
        # ---------------------------
        self.map = {'lymp': 0, 'mela': 1, 'lung': 2, 'brea': 3}
        self.id2task = {v: k for k, v in self.map.items()}
        self.num_tasks = 4

        final_idx = -2 - (num_pool - 1)
        self.conv_blocks_localization_task_specific = nn.ModuleDict()
        for task in self.map.keys():
            stage = nn.Sequential(
                BasicResBlock(dims[final_idx] * 2, dims[final_idx],
                              self.conv_kernel_sizes[final_idx],
                              self.conv_pad_sizes[final_idx],
                              use_1x1conv=True),
                # *[
                #     BasicResBlock(dims[final_idx], dims[final_idx],
                #                   self.conv_kernel_sizes[final_idx],
                #                   self.conv_pad_sizes[final_idx])
                #     for _ in range(depth[final_idx] - 1)
                # ]
                 * [
                     OptimizedResBlock(dims[final_idx], dims[final_idx],
                                   self.conv_kernel_sizes[final_idx],
                                   self.conv_pad_sizes[final_idx])
                     for _ in range(depth[final_idx] - 1)
                 ]
            )
            self.conv_blocks_localization_task_specific[task] = stage

        # ---------------------------
        # ---------------------------
        self.seg_outputs = nn.ModuleList([
            nn.Conv3d(dims[-2 - ds], num_classes, kernel_size=1)
            for ds in range(len(self.conv_blocks_localization))
        ])
        final_idx = -2 - (len(self.pool_op_kernel_sizes) - 1)
        self.seg_outputs_task_specific = nn.ModuleDict()
        for task in self.map.keys():
            self.seg_outputs_task_specific[task] = nn.Conv3d(dims[final_idx], num_classes, kernel_size=1)
        # ---------------------------
        # ---------------------------
        self.upscale_logits_ops = [lambda x: x for _ in range(num_pool - 1)]
        self.leaky = nn.LeakyReLU(inplace=True)

        # prompt logic
        self.intermedia_prompt = nn.Parameter(
            torch.randn(1, 4, 112 // 16, 160 // 32, 128 // 32)
        )

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}
        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]

        self.fusion_layer = StackedFusionConvLayers(
            dims[4] + 4,
            (dims[4] + 4) // 4,
            4,
            3,
            nn.Conv3d,
            self.conv_kwargs,
            nn.InstanceNorm3d, {'eps': 1e-5, 'affine': True},
            nn.Dropout3d, {'p': 0, 'inplace': True},
            nn.LeakyReLU, {'negative_slope': 1e-2, 'inplace': True},
            None, basic_block=ConvDropoutNormNonlin
        )

        # gating layers for "other" prompts
        self.other_gating_list = nn.ModuleList([
            nn.Conv3d(3, 3, kernel_size=1, stride=1, padding=0, bias=True)
            for _ in range(4)
        ])

    def forward(self, x, prompt):
        skips = []
        seg_outputs = []

        # ---------------------------
        # Encoder forward
        # ---------------------------
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
        x = self.conv_blocks_context[-1](x)  # bottom feature

        # ---------------------------
        # Prompt logic
        # ---------------------------
        bs = x.size(0)
        now_prompt = self.intermedia_prompt.repeat(bs, 1, 1, 1, 1)
        dynamic_prompt = self.fusion_layer(torch.cat([x, now_prompt], dim=1))
        task_id = [self.map[batch[0]] for batch in prompt]
        task_id = torch.tensor(task_id, dtype=torch.long, device=x.device)
        task_prompt = dynamic_prompt[torch.arange(bs, device=x.device), task_id, ...].unsqueeze(1)
        other_prompt_indices = []
        for i in range(bs):
            others = [j for j in range(4) if j != task_id[i].item()]
            other_prompt_indices.append(torch.tensor(others, dtype=torch.long, device=x.device))
        other_prompt_indices = torch.stack(other_prompt_indices, dim=0)
        other_prompts = torch.stack(
            [dynamic_prompt[i, other_prompt_indices[i]] for i in range(bs)],
            dim=0
        )
        fused_others_list = []
        for i in range(bs):
            gating_module = self.other_gating_list[task_id[i].item()]
            single_other_prompts = other_prompts[i].unsqueeze(0)
            g_other_i = gating_module(single_other_prompts)
            g_other_i = torch.softmax(g_other_i, dim=1)
            fused_others_i = (g_other_i * single_other_prompts).sum(dim=1, keepdim=True)
            fused_others_list.append(fused_others_i)
        fused_others = torch.cat(fused_others_list, dim=0)

        x = torch.cat([x, task_prompt, fused_others], dim=1)
        # ---------------------------
        # Decoder forward
        # ---------------------------
        num_stages = len(self.upsample_layers)  # num_stages = num_pool
        for u in range(num_stages):
            x = self.upsample_layers[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            if u < num_stages - 1:
                x = self.conv_blocks_localization[u](x)
                out_shared = self.seg_outputs[u](x)
                seg_outputs.append(self.final_nonlin(out_shared))
            else:
                per_stage_output = []
                for b in range(bs):
                    tid = task_id[b].item()
                    task_name = self.id2task[tid]
                    x_b = x[b:b + 1]
                    x_b = self.conv_blocks_localization_task_specific[task_name](x_b)
                    # out_b = self.seg_outputs[u](x_b)
                    out_b = self.seg_outputs_task_specific[task_name](x_b)
                    per_stage_output.append(out_b)
                seg_output_stage = torch.cat(per_stage_output, dim=0)
                seg_outputs.append(self.final_nonlin(seg_output_stage))

        # ---------------------------
        # Deep supervision
        # ---------------------------
        if self.decoder.deep_supervision:
            return tuple(
                [seg_outputs[-1]]
                + [op(s) for op, s in zip(self.upscale_logits_ops[::-1], seg_outputs[:-1][::-1])]
            )
        else:
            return seg_outputs[-1]





class STUNet_(nn.Module):
    def __init__(self,
                 input_channels,
                 num_classes,
                 depth=[1, 1, 1, 1, 1, 1],
                 dims=[32, 64, 128, 256, 512, 512],
                 pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 enable_deep_supervision=True):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        # final nonlin & decoder
        self.final_nonlin = lambda x: x
        self.decoder = Decoder()
        self.decoder.deep_supervision = enable_deep_supervision

        num_pool = len(pool_op_kernel_sizes)
        assert num_pool == len(dims) - 1

        # 1) 并行 Encoder for CT & PET
        self.conv_blocks_context_ct = nn.ModuleList()
        self.conv_blocks_context_pet = nn.ModuleList()

        stage0_ct = nn.Sequential(
            BasicResBlock(1, dims[0],
                          self.conv_kernel_sizes[0],
                          self.conv_pad_sizes[0],
                          use_1x1conv=True),
            *[BasicResBlock(dims[0], dims[0],
                            self.conv_kernel_sizes[0],
                            self.conv_pad_sizes[0])
              for _ in range(depth[0] - 1)]
        )
        stage0_pet = copy.deepcopy(stage0_ct)
        self.conv_blocks_context_ct.append(stage0_ct)
        self.conv_blocks_context_pet.append(stage0_pet)

        # Stages 1..num_pool
        for d in range(1, num_pool + 1):
            block_ct = nn.Sequential(
                BasicResBlock(dims[d - 1], dims[d],
                              self.conv_kernel_sizes[d],
                              self.conv_pad_sizes[d],
                              stride=self.pool_op_kernel_sizes[d - 1],
                              use_1x1conv=True),
                *[BasicResBlock(dims[d], dims[d],
                                self.conv_kernel_sizes[d],
                                self.conv_pad_sizes[d])
                  for _ in range(depth[d] - 1)]
            )
            block_pet = copy.deepcopy(block_ct)
            self.conv_blocks_context_ct.append(block_ct)
            self.conv_blocks_context_pet.append(block_pet)

        self.map = {'lymp': 0, 'mela': 1, 'lung': 2, 'brea': 3}
        self.id2task = {v: k for k, v in self.map.items()}
        self.num_tasks = len(self.map)

        self.intermedia_prompt = nn.Parameter(
            torch.randn(1, 4,
                        112 // 16,
                        160 // 32,
                        128 // 32)
        )
        fusion_in_ch = dims[-1] * 2 + 4
        self.fusion_layer = StackedFusionConvLayers(
            fusion_in_ch,
            fusion_in_ch // 4,
            4, 3,
            nn.Conv3d,
            {
                'kernel_size': conv_kernel_sizes[num_pool],
                'padding': self.conv_pad_sizes[num_pool],
                'stride': 1,
                'dilation': 1,
                'bias': True
            },
            nn.InstanceNorm3d, {'eps': 1e-5, 'affine': True},
            nn.Dropout3d, {'p': 0, 'inplace': True},
            nn.LeakyReLU, {'negative_slope': 1e-2, 'inplace': True},
            None, basic_block=ConvDropoutNormNonlin
        )
        self.other_gating_list = nn.ModuleList([
            nn.Conv3d(self.num_tasks - 1,
                      self.num_tasks - 1,
                      kernel_size=1, stride=1, padding=0, bias=True)
            for _ in range(self.num_tasks)
        ])

        prompt_ch = 2  # task_prompt(1) + fused_others(1)
        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            if u == 0:
                in_ch = dims[-1] * 2 + prompt_ch
            else:
                in_ch = dims[-1 - u]
            out_ch = dims[-2 - u]
            self.upsample_layers.append(
                Upsample_Layer_nearest(in_ch, out_ch, pool_op_kernel_sizes[-1 - u])
            )

        # 4) Localization blocks
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool - 1):
            in_ch = dims[-2 - u] * 3  # upsample + skip(CT+PET)
            stage = nn.Sequential(
                BasicResBlock(in_ch, dims[-2 - u],
                              self.conv_kernel_sizes[-2 - u],
                              self.conv_pad_sizes[-2 - u],
                              use_1x1conv=True),
                *[BasicResBlock(dims[-2 - u], dims[-2 - u],
                                self.conv_kernel_sizes[-2 - u],
                                self.conv_pad_sizes[-2 - u])
                  for _ in range(depth[-2 - u] - 1)]
            )
            self.conv_blocks_localization.append(stage)

        final_idx = -2 - (num_pool - 1)
        self.conv_blocks_localization_task_specific = nn.ModuleDict()
        for task in self.map:
            in_ch = dims[final_idx] * 3
            stage = nn.Sequential(
                BasicResBlock(in_ch, dims[final_idx],
                              self.conv_kernel_sizes[final_idx],
                              self.conv_pad_sizes[final_idx],
                              use_1x1conv=True),
                *[OptimizedResBlock(dims[final_idx], dims[final_idx],
                                    self.conv_kernel_sizes[final_idx],
                                    self.conv_pad_sizes[final_idx])
                  for _ in range(depth[final_idx] - 1)]
            )
            self.conv_blocks_localization_task_specific[task] = stage

        # 5) Segmentation heads
        self.seg_outputs = nn.ModuleList([
            nn.Conv3d(dims[-2 - ds], num_classes, kernel_size=1)
            for ds in range(len(self.conv_blocks_localization))
        ])
        self.seg_outputs_task_specific = nn.ModuleDict()
        for task in self.map:
            self.seg_outputs_task_specific[task] = nn.Conv3d(
                dims[final_idx], num_classes, kernel_size=1
            )

        self.upscale_logits_ops = [lambda x: x for _ in range(num_pool - 1)]
        self.leaky = nn.LeakyReLU(inplace=True)

    def forward(self, x, prompt):
        ct = x[:, 0:1]
        pet = x[:, 1:2]

        # 2) Encoder + skips
        skips = []
        for d in range(len(self.conv_blocks_context_ct) - 1):
            ct = self.conv_blocks_context_ct[d](ct)
            pet = self.conv_blocks_context_pet[d](pet)
            skips.append(torch.cat([ct, pet], dim=1))

        ct = self.conv_blocks_context_ct[-1](ct)
        pet = self.conv_blocks_context_pet[-1](pet)
        x = torch.cat([ct, pet], dim=1)

        # 3) Prompt Logic
        bs = x.size(0)
        now_prompt = self.intermedia_prompt.repeat(bs, 1, 1, 1, 1)
        fusion_input = torch.cat([x, now_prompt], dim=1)
        dynamic_prompt = self.fusion_layer(fusion_input)

        task_id = torch.tensor([self.map[p[0]] for p in prompt],
                               dtype=torch.long, device=x.device)
        task_prompt = dynamic_prompt[torch.arange(bs), task_id].unsqueeze(1)

        fused_others = []
        for i in range(bs):
            others = [t for t in range(self.num_tasks) if t != task_id[i].item()]
            other = dynamic_prompt[i, others]
            single = other.unsqueeze(0)
            gate = self.other_gating_list[task_id[i].item()](single)
            gate = torch.softmax(gate, dim=1)
            fused = (gate * single).sum(dim=1, keepdim=True)
            fused_others.append(fused)
        fused_others = torch.cat(fused_others, dim=0)

        x = torch.cat([x, task_prompt, fused_others], dim=1)

        # 4) Decoder forward
        seg_outputs = []
        for u, up in enumerate(self.upsample_layers):
            x = up(x)
            skip = skips[-(u + 1)]
            x = torch.cat([x, skip], dim=1)
            if u < len(self.upsample_layers) - 1:
                x = self.conv_blocks_localization[u](x)
                seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
            else:
                per_stage = []
                for b in range(bs):
                    tid = task_id[b].item()
                    name = self.id2task[tid]
                    xb = x[b:b + 1]
                    xb = self.conv_blocks_localization_task_specific[name](xb)
                    per_stage.append(self.seg_outputs_task_specific[name](xb))
                seg_outputs.append(torch.cat(per_stage, dim=0))

        if self.decoder.deep_supervision:
            return tuple(
                [seg_outputs[-1]] +
                [op(s) for op, s in zip(self.upscale_logits_ops[::-1],
                                        seg_outputs[:-1][::-1])]
            )
        else:
            return seg_outputs[-1]
class STUNet_prompt_pretrain(STUNet_):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_pretrained_encoders(self, ct_ckpt_path, pet_ckpt_path, map_location='cpu', strict=True):
        """
        """
        ct_state = torch.load(ct_ckpt_path, map_location=map_location)
        pet_state = torch.load(pet_ckpt_path, map_location=map_location)

        for i, block in enumerate(self.conv_blocks_context_ct):
            prefix = f'blocks.{i}.'
            subdict = {
                k[len(prefix):]: v
                for k, v in ct_state.items()
                if k.startswith(prefix)
            }
            block.load_state_dict(subdict, strict=strict)

        for i, block in enumerate(self.conv_blocks_context_pet):
            prefix = f'blocks.{i}.'
            subdict = {
                k[len(prefix):]: v
                for k, v in pet_state.items()
                if k.startswith(prefix)
            }
            block.load_state_dict(subdict, strict=strict)

        print(f"Loaded CT encoder from {ct_ckpt_path}")
        print(f"Loaded PET encoder from {pet_ckpt_path}")


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep_supervision = True

class BasicResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1, stride=1, use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=2):
        """
        Args:
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        return x * avg_out

class OptimizedResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1, stride=1, use_1x1conv=False):
        """
        Args:
        """
        super(OptimizedResBlock, self).__init__()

        branch_channels = output_channels // 2
        self.branch1 = nn.Conv3d(input_channels, branch_channels, kernel_size=1, stride=stride, padding=0)
        self.branch2 = nn.Conv3d(input_channels, branch_channels, kernel_size=3, stride=stride, padding=1)
        self.branch3 = nn.Conv3d(input_channels, branch_channels, kernel_size=5, stride=stride, padding=2)

        self.fuse_conv = nn.Conv3d(branch_channels * 3, output_channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(output_channels, affine=True)
        self.act = nn.LeakyReLU(inplace=True)

        self.ca = ChannelAttention(output_channels, reduction=2)

        if use_1x1conv or input_channels != output_channels or stride != 1:
            self.res_conv = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.res_conv = None

    def forward(self, x):
        residual = x
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.fuse_conv(out)
        out = self.norm(out)
        out = self.act(out)
        out = self.ca(out)
        if self.res_conv is not None:
            residual = self.res_conv(x)
        out += residual
        out = self.act(out)
        return out

class Upsample_Layer_nearest(nn.Module):
    def __init__(self, input_channels, output_channels, pool_op_kernel_size, mode='nearest'):
        super().__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x
class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))
class StackedFusionConvLayers(nn.Module):
    def __init__(self, input_feature_channels, bottleneck_feature_channel, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = copy.deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedFusionConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, bottleneck_feature_channel, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(bottleneck_feature_channel, bottleneck_feature_channel, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 2)] +
              [basic_block(bottleneck_feature_channel, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)]
              ))

    def forward(self, x):
        return self.blocks(x)