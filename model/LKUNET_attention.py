import random
import sys
import time

from model.pke_module import pke_learn

from torch.nn import functional as F
import torch
import torch.nn as nn

from loss.dice_loss import DiceBCELoss, DiceLoss
from loss.triplet_loss import triplet_margin_loss_gor, triplet_margin_loss_gor_one, sos_reg

from common.common_util import remove_borders, sample_keypoint_desc, simple_nms, nms, \
    sample_descriptors
from common.train_util import get_gaussian_kernel, affine_images


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class LkAttentionBlock(nn.Module):
    def __init__(self, inchannels, outchannels, device='cpu', attention=True):
        super().__init__()

        self.sigmoid = torch.nn.Sigmoid()

        self.bi_upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv = torch.nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=1, padding=1)
        
        self.small_filter = torch.nn.Conv2d(outchannels, outchannels, kernel_size=1, stride=1, padding=0)
        self.regular_filter = torch.nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1)
        self.large_filter = torch.nn.Conv2d(outchannels, outchannels, kernel_size=5, stride=1, padding=2)
        
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.attention_g = torch.nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1)
        self.attention_x = torch.nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=2, padding=1)
        self.collapse = torch.nn.Conv2d(outchannels,1, kernel_size=1, stride=1,padding=0)

        self.attention_flag = attention
        self.to(device)
    
    def forward(self, x):
        x = nn.ReLU(inplace=True)(self.conv(x))

        x = x + self.small_filter(x) + self.regular_filter(x) + self.large_filter(x)
        

        if self.attention_flag == False:
          return nn.ReLU(inplace=True)(x)

        conv = nn.ReLU(inplace=True)(x)
        x = self.pool(conv)
        conv = conv*self.bi_upsample(self.sigmoid(self.collapse(torch.nn.ReLU(inplace=True)(self.attention_g(x)+ self.attention_x(conv)))))

        return x, conv
        
class SuperRetina(nn.Module):
    def __init__(self, config=None, device='cpu', n_class=1):
        super().__init__()

        self.PKE_learn = True
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1, d2 = 64, 64, 128, 128, 256, 256, 256
        # Shared Encoder.
        self.lkencoder_1 = LkAttentionBlock(1, c1, device)
        self.lkencoder_2 = LkAttentionBlock(c1, c2, device)
        self.lkencoder_3 = LkAttentionBlock(c2, c3, device)
        self.lkencoder_4 = LkAttentionBlock(c3, c4, device, attention=False)

        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=4, stride=2, padding=0)
        self.convDc = torch.nn.Conv2d(d1, d2, kernel_size=1, stride=1, padding=0)

        self.trans_conv = nn.ConvTranspose2d(d1, d2, 2, stride=2)
        self.trans_conv_2 = nn.ConvTranspose2d(d2, d2, 2, stride=2)

        # Detector Head
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(c3 + c4, c3)
        self.dconv_up2 = double_conv(c2 + c3, c2)
        self.dconv_up1 = double_conv(c1 + c2, c1)

        self.conv_last = nn.Conv2d(c1, n_class, kernel_size=1)

        if config is not None:
            self.config = config

            self.nms_size = config['nms_size']
            self.nms_thresh = config['nms_thresh']
            self.scale = 8

            self.dice = DiceLoss()

            self.kernel = get_gaussian_kernel(kernlen=config['gaussian_kernel_size'],
                                              nsig=config['gaussian_sigma']).to(device)

        self.to(device)

    def network(self, x):
        
        x, conv1 = self.lkencoder_1(x)
        x, conv2 = self.lkencoder_2(x)
        x, conv3 = self.lkencoder_3(x)

        x = self.lkencoder_4(x)

        # Descriptor Head.
        cDa = nn.ReLU(inplace=True)(self.convDa(x))
        cDb = nn.ReLU(inplace=True)(self.convDb(cDa))
        desc = self.convDc(cDb)

        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        desc = self.trans_conv(desc)

        cPa = self.upsample(x)
        cPa = torch.cat([cPa, conv3], dim=1)

        cPa = self.dconv_up3(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv2], dim=1)

        cPa = self.dconv_up2(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv1], dim=1)

        cPa = self.dconv_up1(cPa)

        semi = self.conv_last(cPa)
        semi = torch.sigmoid(semi)

        return semi, desc


    def descriptor_loss(self, detector_pred, label_point_positions, descriptor_pred,
                        affine_descriptor_pred, grid_inverse, affine_detector_pred=None):
        """
        calculate descriptor loss, construct triples on raw images and affine images
        :param detector_pred: output of detector network
        :param label_point_positions: initial label points
        :param descriptor_pred: output of descriptor network
        :param affine_descriptor_pred: output of descriptor network, with affine images as input
        :param grid_inverse: used for inverse affine transformation
        :return: descriptor loss (triplet loss)
        """

        # sample keypoints on initial labels
        # label_descriptors, label_affine_descriptors, label_keypoints = \
        #     sample_descriptors(label_point_positions, descriptor_pred, affine_descriptor_pred, grid_inverse,
        #                        nms_size=self.nms_size, nms_thresh=self.nms_thresh, scale=self.scale)
        #
        # for s, kps in enumerate(label_keypoints):
        #     label_mask = torch.zeros(detector_pred[s].shape).to(detector_pred)
        #     label_mask[0, kps[:, 1].long(), kps[:, 0].long()] = 1
        #     label_mask = F.conv2d(label_mask.unsqueeze(0), self.mask_kernel, stride=1,
        #                           padding=(self.mask_kernel.shape[-1] - 1) // 2)
        #     detector_pred[s][label_mask[0] > 1e-5] = 0
        if not self.PKE_learn:
            detector_pred[:] = 0  # only learn from the initial labels
        detector_pred[label_point_positions == 1] = 10
        descriptors, affine_descriptors, keypoints = \
            sample_descriptors(detector_pred, descriptor_pred, affine_descriptor_pred, grid_inverse,
                               nms_size=self.nms_size, nms_thresh=self.nms_thresh, scale=self.scale,
                               affine_detector_pred=affine_detector_pred)

        # descriptors_tmp = []
        # affine_descriptor_tmp = []
        # for i in range(len(descriptors)):
        #     descriptors_tmp.append(torch.cat((descriptors[i], label_descriptors[i]), -1))
        #     affine_descriptor_tmp.append(torch.cat((affine_descriptors[i], label_affine_descriptors[i]), -1))
        # descriptors = descriptors_tmp
        # affine_descriptors = affine_descriptor_tmp

        positive = []
        negatives_hard = []
        negatives_random = []
        anchor = []
        D = descriptor_pred.shape[1]
        for i in range(len(affine_descriptors)):
            if affine_descriptors[i].shape[1] == 0:
                continue
            descriptor = descriptors[i]
            affine_descriptor = affine_descriptors[i]

            n = affine_descriptors[i].shape[1]
            if n > 1000:  # avoid OOM
                return torch.tensor(0., requires_grad=True).to(descriptor_pred), False

            descriptor = descriptor.view(D, -1, 1)
            affine_descriptor = affine_descriptor.view(D, 1, -1)
            ar = torch.arange(n)

            # random
            neg_index2 = []
            if n == 1:
                neg_index2.append(0)
            else:
                for j in range(n):
                    t = j
                    while t == j:
                        t = random.randint(0, n - 1)
                    neg_index2.append(t)
            neg_index2 = torch.tensor(neg_index2, dtype=torch.long).to(affine_descriptor)

            # hard
            with torch.no_grad():
                dis = torch.norm(descriptor - affine_descriptor, dim=0)
                dis[ar, ar] = dis.max() + 1
                neg_index1 = dis.argmin(axis=1)

            positive.append(affine_descriptor[:, 0, :].permute(1, 0))
            anchor.append(descriptor[:, :, 0].permute(1, 0))
            negatives_hard.append(affine_descriptor[:, 0, neg_index1.long(), ].permute(1, 0))
            negatives_random.append(affine_descriptor[:, 0, neg_index2.long(), ].permute(1, 0))

        if len(positive) == 0:
            return torch.tensor(0., requires_grad=True).to(descriptor_pred), False

        positive = torch.cat(positive)
        anchor = torch.cat(anchor)
        negatives_hard = torch.cat(negatives_hard)
        negatives_random = torch.cat(negatives_random)

        positive = F.normalize(positive, dim=-1, p=2)
        anchor = F.normalize(anchor, dim=-1, p=2)
        negatives_hard = F.normalize(negatives_hard, dim=-1, p=2)
        negatives_random = F.normalize(negatives_random, dim=-1, p=2)

        loss = triplet_margin_loss_gor(anchor, positive, negatives_hard, negatives_random, margin=0.8)

        # can also add sos reg term .
        # reg_term = sos_reg(anchor, positive, KNN=True, k=1, eps=1e-8)
        # if not torch.isnan(reg_term) and reg_term > 0:
        #     loss = loss + 0.1 * reg_term

        return loss, True

    def forward(self, x, label_point_positions=None, value_map=None, learn_index=None):
        """
        In interface phase, only need to input x
        :param x: retinal images
        :param label_point_positions: positions of keypoints on labels
        :param value_map: value maps, used to record history learned geo_points
        :param learn_index: index of input data with detector labels
        :param phase: distinguish dataset
        :return: if training, return loss, else return predictions
        """

        detector_pred, descriptor_pred = self.network(x)
        enhanced_label_pts = None
        enhanced_label = None

        if label_point_positions is not None:
            if self.PKE_learn:
                loss_detector_num = len(learn_index[0])
                loss_descriptor_num = x.shape[0]
            else:
                loss_detector_num = len(learn_index[0])
                loss_descriptor_num = loss_detector_num

            number_pts = 0  # number of learned keypoints
            value_map_update = None
            loss_detector = torch.tensor(0., requires_grad=True).to(x)
            loss_descriptor = torch.tensor(0., requires_grad=True).to(x)

            with torch.no_grad():
                affine_x, grid, grid_inverse = affine_images(x, used_for='detector')
                affine_detector_pred, affine_descriptor_pred = self.network(affine_x)
            loss_cal = self.dice
            if len(learn_index[0]) != 0:
                loss_detector, number_pts, value_map_update, enhanced_label_pts, enhanced_label = \
                    pke_learn(detector_pred[learn_index], descriptor_pred[learn_index],
                              grid_inverse[learn_index], affine_detector_pred[learn_index],
                              affine_descriptor_pred[learn_index], self.kernel, loss_cal,
                              label_point_positions[learn_index], value_map[learn_index],
                              self.config, self.PKE_learn)

            #  For showing PKE process
            if enhanced_label_pts is not None:
                enhanced_label_pts_tmp = label_point_positions.clone()
                enhanced_label_pts_tmp[learn_index] = enhanced_label_pts
                enhanced_label_pts = enhanced_label_pts_tmp
            if enhanced_label is not None:
                enhanced_label_tmp = label_point_positions.clone()
                enhanced_label_tmp[learn_index] = enhanced_label
                enhanced_label = enhanced_label_tmp

            detector_pred_copy = detector_pred.clone().detach()
            # if value_map_update is not None:
            #     # optimize descriptors of recorded points
            #     detector_pred_copy[learn_index][value_map_update >=
            #                                     self.config['VALUE MAP'].getfloat('value_increase_point')] = 1
            #
            affine_x_for_desc, grid_for_desc, grid_inverse_for_desc = affine_images(x, used_for='descriptor')
            _, affine_descriptor_pred_for_desc = self.network(affine_x_for_desc)
            loss_descriptor, descriptor_train_flag = self.descriptor_loss(detector_pred_copy, label_point_positions,
                                                                          descriptor_pred,
                                                                          affine_descriptor_pred_for_desc,
                                                                          grid_inverse_for_desc)

            if self.PKE_learn and len(learn_index[0]) != 0:
                value_map[learn_index] = value_map_update
            loss = loss_detector + loss_descriptor

            return loss, number_pts, loss_detector.cpu().data.sum(), \
                   loss_descriptor.cpu().data.sum(), enhanced_label_pts, \
                   enhanced_label, detector_pred, loss_detector_num, loss_descriptor_num

        return detector_pred, descriptor_pred
