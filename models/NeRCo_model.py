# -*- coding: utf-8 -*-
import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import sobel
import random
import clip
from torchvision.transforms import Resize
import numpy as np
from util import util_save
import torch.nn as nn

'''CLIP code'''
device = "cuda" if torch.cuda.is_available() else "cpu"
CLIP, preprocess = clip.load("RN50", device=device)
torch_resize = Resize([224,224])

real_B=np.array([[0., 1.]])
real_B=torch.Tensor(real_B).to(device)

real_A=np.array([[1., 0.]])
real_A=torch.Tensor(real_A).to(device)

text = clip.tokenize(["low light image", "high light image"]).to(device)

class NeRComodel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'D_A_edge', 'D_B_edge', 'D_A_CLIP', 'D_B_CLIP', 'PreConsis', 'H']
        visual_names_A = ['fake_B']
        
        self.visual_names = visual_names_A
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A_edge', 'D_B_edge', 'D_A', 'D_B', 'Pre', 'H']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'Pre', 'H']

        self.netG_A = networks.define_G(opt.input_nc * 2, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netPre = networks.define_Pre(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netH = networks.define_H(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_A_edge = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B_edge = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.pre_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_A_edge_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_edge_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_A_clip_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_clip_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.

            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netPre.parameters(), self.netG_A.parameters(), self.netG_B.parameters(), self.netH.parameters()), lr=opt.lr,
                betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(),self.netD_A_edge.parameters(), self.netD_B_edge.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
    
    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.pre_A = self.netPre(self.real_A)
        self.H, self.mask = self.netH(self.real_A)

        temp = torch.cat((self.real_A, self.pre_A), 1)
        self.fake_B = self.netG_A(temp * self.mask)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))

        self.fake_A = self.netG_B(self.real_B * self.mask)  # G_B(B)
        self.pre_A1 = self.netPre(self.fake_A)
        temp = torch.cat((self.fake_A, self.pre_A1), 1)
        self.rec_B = self.netG_A(temp)   # G_A(G_B(B))


    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    
    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

        return self.loss_D_A

    def backward_D_A_edge(self):
        self.fake_B_edge = sobel.tran(self.fake_B)
        fake_B_edge = self.fake_B_edge_pool.query(self.fake_B_edge)
        self.real_B_edge = sobel.tran(self.real_B)
        self.loss_D_A_edge = self.backward_D_basic(self.netD_A_edge, self.real_B_edge, fake_B_edge)

        return self.loss_D_A_edge
    
    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        pre_A = self.pre_A_pool.query(self.pre_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A) + self.backward_D_basic(self.netD_B, self.real_A, pre_A)


        return self.loss_D_B
    
    def backward_D_B_edge(self):
        self.fake_A_edge = sobel.tran(self.fake_A)
        fake_A_edge = self.fake_A_edge_pool.query(self.fake_A_edge)
        self.real_A_edge = sobel.tran(self.real_A)
        self.loss_D_B_edge = self.backward_D_basic(self.netD_B_edge, self.real_A_edge, fake_A_edge)

        return self.loss_D_B_edge

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(torch.cat((self.real_B, self.real_B), 1))
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True) + self.criterionGAN(self.netD_B(self.pre_A), True)
        
        self.loss_G_A_edge = self.criterionGAN(self.netD_A_edge(self.fake_B), True)
        self.loss_G_B_edge = self.criterionGAN(self.netD_B_edge(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A

        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        self.loss_PreConsis = self.criterionCycle(self.pre_A, self.real_A) * 4 * lambda_A + self.criterionGAN(self.netD_B(self.pre_A), True)

        """Calculate the loss for Mask Extractor H"""
        self.G_A_real_B_minus_H = self.netG_A(torch.cat((self.real_B - self.H, self.real_B - self.H), 1))
        # self.loss_Bypass = self.criterionCycle(self.real_B, self.real_A - self.H) + self.criterionCycle(self.real_B,self.netG_A(self.real_B + self.H))
        loss_Bypass = self.criterionCycle(self.fake_B, self.real_A + self.H) + self.criterionCycle(self.real_B,
                                                                                                        self.G_A_real_B_minus_H)
        resultA = self.netD_A(self.real_A + self.H)
        loss_D_A_h = self.criterionGAN(resultA, True)
        resultB = self.netD_B(self.real_B - self.H)
        loss_D_B_h = self.criterionGAN(resultB, True)
        # 伪噪声标签
        netH_fake_A_concat_ca, _ = self.netH(self.fake_A)
        loss_self = self.criterionCycle(self.H, self.fake_B - self.real_A) + \
                         self.criterionCycle(self.fake_A, self.real_B - netH_fake_A_concat_ca)
        self.loss_H = loss_Bypass + loss_D_A_h + loss_D_B_h + loss_self

        """Calculate the loss for Mask Extractor CLIP"""
        fake_B_CLIP = torch_resize(self.fake_B_clip_pool.query(self.fake_B))
        rec_B_CLIP = torch_resize(self.fake_B_clip_pool.query(self.rec_B))

        logits_per_image, logits_per_text = CLIP(fake_B_CLIP, text)
        probs = logits_per_image.softmax(dim=-1)
        self.loss_D_A_CLIP = self.criterionCycle(probs, real_B)

        logits_per_image, logits_per_text = CLIP(rec_B_CLIP, text)
        probs = logits_per_image.softmax(dim=-1)
        self.loss_D_A_CLIP += self.criterionCycle(probs, real_B)

        fake_A_CLIP = torch_resize(self.fake_B_clip_pool.query(self.fake_A))
        rec_A_CLIP = torch_resize(self.fake_A_clip_pool.query(self.rec_A))

        logits_per_image, logits_per_text = CLIP(fake_A_CLIP, text)
        probs = logits_per_image.softmax(dim=-1)
        self.loss_D_B_CLIP = self.criterionCycle(probs, real_A)

        logits_per_image, logits_per_text = CLIP(rec_A_CLIP, text)
        probs = logits_per_image.softmax(dim=-1)
        self.loss_D_B_CLIP += self.criterionCycle(probs, real_A)
        
        self.loss_A = self.loss_G_A + self.loss_cycle_A + self.loss_idt_A + self.loss_D_A_CLIP * 0.5 + self.loss_G_A_edge * 0.5
        self.loss_B = self.loss_G_B + self.loss_cycle_B + self.loss_idt_B + self.loss_D_B_CLIP * 0.5 + self.loss_G_B_edge * 0.5
        
        # combined loss and calculate gradients
        self.loss_G = self.loss_A + self.loss_B + self.loss_H * 0.5 + self.loss_PreConsis
        self.loss_G.backward()
        
        return self.loss_G


    def optimize_parameters(self):
        # forward
        self.forward()      # compute fake images and reconstruction images.
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_A_edge, self.netD_B_edge], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        loss_G = self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_A_edge, self.netD_B_edge], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        loss_D_A = self.backward_D_A()      # calculate gradients for D_A
        loss_D_B = self.backward_D_B()      # calculate graidents for D_B
        loss_D_A_edge = self.backward_D_A_edge()
        loss_D_B_edge = self.backward_D_B_edge()
        self.optimizer_D.step()  # update D_A and D_B's weights

        return loss_G + loss_D_A + loss_D_B + loss_D_A_edge + loss_D_B_edge
