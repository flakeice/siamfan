# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.xcorr import xcorr_fast, xcorr_depthwise
from .attention import ChannelSpatialSelfAttention, ChannelCrossAttention, SpatialCrossAttention


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class UPChannelRPN(RPN):
    def __init__(self, anchor_num=5, feature_in=256):
        super(UPChannelRPN, self).__init__()

        cls_output = 2 * anchor_num
        loc_output = 4 * anchor_num

        self.template_cls_conv = nn.Conv2d(feature_in, 
                feature_in * cls_output, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(feature_in, 
                feature_in * loc_output, kernel_size=3)

        self.search_cls_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)


    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls_conv(z_f)
        loc_kernel = self.template_loc_conv(z_f)

        cls_feature = self.search_cls_conv(x_f)
        loc_feature = self.search_loc_conv(x_f)

        cls = xcorr_fast(cls_feature, cls_kernel)
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))
        return cls, loc


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, hidden_kernel_size=5, attention_kernel_size=3):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )
        self.kernel_attn = ChannelSpatialSelfAttention(in_channels, attention_kernel_size)
        self.search_attn = ChannelSpatialSelfAttention(in_channels, attention_kernel_size)

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        kernel = self.kernel_attn(kernel)
        search = self.search_attn(search)
        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out


class MultiRPN(RPN):
    def __init__(self, anchor_num, in_channels, weighted=False):
        super(MultiRPN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('cls'+str(i+2), DepthwiseXCorr(in_channels[i], in_channels[i], 2 * anchor_num, attention_kernel_size=7))
            self.add_module('loc'+str(i+2), DepthwiseXCorr(in_channels[i], in_channels[i], 4 * anchor_num, attention_kernel_size=3))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))
        self.cls_map = [1, 2, 0]
        self.loc_map = [2, 0, 1]
        self.cls_attn = nn.ModuleList([ChannelCrossAttention(c) for c in in_channels])
        self.loc_attn = nn.ModuleList([SpatialCrossAttention(c) for c in in_channels])

    def forward(self, z_fs, x_fs):
        cls = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            cls_xcorr = getattr(self, 'cls'+str(idx))
            z_f = self.cls_attn[idx-2](z_f, z_fs[self.cls_map[idx-2]], z_fs[self.cls_map[idx-2]])
            x_f = self.cls_attn[idx-2](x_f, x_fs[self.cls_map[idx-2]], x_fs[self.cls_map[idx-2]])
            c = cls_xcorr(z_f, x_f)
            cls.append(c)

        loc = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            loc_xcorr = getattr(self, 'loc'+str(idx))
            z_f = self.loc_attn[idx-2](z_f, z_fs[self.loc_map[idx-2]], z_fs[self.loc_map[idx-2]])
            x_f = self.loc_attn[idx-2](x_f, x_fs[self.loc_map[idx-2]], x_fs[self.loc_map[idx-2]])
            l = loc_xcorr(z_f, x_f)
            loc.append(l)

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)
        else:
            return avg(cls), avg(loc)
