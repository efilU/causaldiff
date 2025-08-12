from enum import Enum

import torch
from torch import Tensor
from torch.nn.functional import silu

from .latentnet import *  # 加上.表示相对导入
from .unet import *
from choices import *

from .wide_resnet import BasicBlock
from .wide_resnet import NetworkBlock
from torch import nn, optim, autograd
import numpy as np

# std = 1.
# Flatten类：将输入张量展平为二维张量(batch_size, -1)
class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)

# UnFlatten类：将展平的张量恢复为指定的形状
class UnFlatten(nn.Module):
	def __init__(self, type = '3d'):
		super(UnFlatten, self).__init__()
		self.type = type
	def forward(self, input):
		if self.type == '3d':
			# 恢复为5D张量(batch_size, channels, 1, 1, 1)
			return input.view(input.size(0), input.size(1), 1, 1, 1)
		else:
			# 恢复为4D张量(batch_size, channels, 1, 1)
			return input.view(input.size(0), input.size(1), 1, 1)


# 定义自动编码器的配置参数
@dataclass
class BeatGANsAutoencConfig(BeatGANsUNetConfig):
	# number of style channels
	# 编码器输出通道数，即潜在表示的维度
	enc_out_channels: int = 512
	# 编码器注意力分辨率
	enc_attn_resolutions: Tuple[int] = None
	# 编码器池化方式
	enc_pool: str = 'depthconv'
	# 编码器残差块数量
	enc_num_res_block: int = 2
	# 编码器通道倍增因子
	enc_channel_mult: Tuple[int] = None
	# 是否使用梯度检查点
	enc_grad_checkpoint: bool = False
	# 潜在网络配置
	latent_net_conf: MLPSkipNetConfig = None
	# CLUB隐藏层维度
	club_hidden_dim: int = 64
	# 是否使用CLUB约束
	use_club: bool = False
	# 是否使用一致性约束
	consistency: bool = False
	# 掩码阈值
	mask_threshold: float = None

	# 创建模型实例的方法
	def make_model(self):
		return BeatGANsAutoencModel(self)


# d_LaCIM类：实现LaCIM(Lanent Causal Influence Model)模型
class d_LaCIM(nn.Module):
	def __init__(self,
					in_channel=1,
					zs_dim=256,  # 潜在空间维度
					num_classes=1,
					decoder_type=0,  # 解码器类型
					total_env=2,
					is_cuda=1
					):
		
		super(d_LaCIM, self).__init__()
		# print('model: d_LaCIM, zs_dim: %d' % zs_dim)
		self.in_channel = in_channel
		self.num_classes = num_classes
		self.zs_dim = zs_dim
		self.decoder_type = decoder_type
		self.total_env = total_env
		self.is_cuda = is_cuda
		self.in_plane = zs_dim
		# z变量维度设为潜在空间维度的一半
		self.z_dim = int(round(zs_dim * 0.5))
		# 创建图像编码器
		self.Enc_x = self.get_Enc_x_28()
		self.u_dim = total_env
		print('z_dim is ', self.z_dim)
		# s变量维度为剩余部分
		self.s_dim = int(self.zs_dim - self.z_dim)
		# 存储各环境下的均值和对数方差网络
		self.mean_z = []
		self.logvar_z = []
		self.mean_s = []
		self.logvar_s = []

		# 定义共享层以减少参数数量
		self.shared_s1 = self.Fc_bn_ReLU(self.in_plane, self.in_plane)
		self.shared_s2 = nn.Linear(self.in_plane, self.s_dim)
		self.shared_s3 = self.Fc_bn_ReLU(self.in_plane, self.in_plane)
		self.shared_s4 = nn.Linear(self.in_plane, self.s_dim)
		self.shared_z1 = self.Fc_bn_ReLU(self.in_plane, self.in_plane)
		self.shared_z2 = nn.Linear(self.in_plane, self.z_dim)
		self.shared_z3 = self.Fc_bn_ReLU(self.in_plane, self.in_plane)
		self.shared_z4 = nn.Linear(self.in_plane, self.z_dim)

		# 为每个环境创建特定的均值和方差网络
		for env_idx in range(self.total_env):
			self.mean_z.append(
				nn.Sequential(
					# self.Fc_bn_ReLU(self.in_plane, self.in_plane),
					# nn.Linear(self.in_plane, self.z_dim)
					self.shared_z1,
					self.shared_z2
				)
			)
			self.logvar_z.append(
				nn.Sequential(
					# self.Fc_bn_ReLU(self.in_plane, self.in_plane),
					# nn.Linear(self.in_plane, self.z_dim)
					self.shared_z3,
					self.shared_z4
				)
			)
			self.mean_s.append(
				nn.Sequential(
					# self.shared_s,
					# nn.Linear(self.in_plane, self.s_dim)
					self.shared_s1,
					self.shared_s2
				)
			)
			self.logvar_s.append(
				nn.Sequential(
					# self.shared_s,
					# nn.Linear(self.in_plane, self.s_dim)
					self.shared_s3,
					self.shared_s4
				)
			)

		# 将列表转换为ModuleList以便正确注册参数
		self.mean_z = nn.ModuleList(self.mean_z)
		self.logvar_z = nn.ModuleList(self.logvar_z)
		self.mean_s = nn.ModuleList(self.mean_s)
		self.logvar_s = nn.ModuleList(self.logvar_s)
		
		# prior
		# 先验网络
		self.Enc_u_prior = self.get_Enc_u()
		# 先验均值和对数方差网络
		self.mean_zs_prior = nn.Sequential(
			nn.Linear(32, self.zs_dim))
		self.logvar_zs_prior = nn.Sequential(
			nn.Linear(32, self.zs_dim))
		
		# 解码器网络（用于分类）
		self.Dec_y = self.get_Dec_y()
		# 可学习的alpha参数
		self.alpha = nn.Parameter(torch.FloatTensor([1.0, 1.0]))


	# 获取潜在变量z和s的方法
	def get_zs(self, x, target, env, adv_mode='none'):
		# 通过图像编码器提取特征
		x = self.Enc_x(x)
		# 编码获取均值和方差
		mu, logvar = self.encode(x, env)
		# 重参数化采样
		zs = self.reparametrize(mu, logvar)
		# 分离z和s变量
		z = zs[:, :self.z_dim]
		s = zs[:, self.z_dim:]
		# adversarial evaluate mode
		# if adv_mode == 'z':
		# 	z = z+self.delta
		# elif adv_mode == 's':
		# 	s = s+self.delta
		return z, s

	# 获取预测的分类结果
	def get_pred_y(self, x, env):
		# 通过图像编码器提取特征
		x = self.Enc_x(x)
		# 编码获取均值和方差
		mu, logvar = self.encode(x, env)
		# 重参数化采样
		zs = self.reparametrize(mu, logvar)
		# 分离z和s变量
		z = zs[:, :self.z_dim]
		s = zs[:, self.z_dim:]
		# “合并”z和s
		zs = torch.cat([z, s], dim=1)
		# 通过解码器获取预测结果（仅使用s变量）
		pred_y = self.Dec_y(zs[:, self.z_dim:])
		return pred_y

	# 获取重建图像和预测分类结果
	def get_x_y(self, z, s):
		# 合并z和s变量
		zs = torch.cat([z, s], dim=1)
		# 通过解码器重建图像（使用z，s）
		rec_x = self.Dec_x(zs)
		# 通过解码器获取预测结果（仅使用s变量）
		pred_y = self.Dec_y(zs[:, self.z_dim:])
		# 返回裁剪后的重建图像和预测结果
		# 去除图像边缘的填充区域，get_Enc_x_28()方法？
		return rec_x[:, :, 2:30, 2:30].contiguous(), pred_y

	# 仅通过s变量获取预测结果
	def get_y(self, s):
		return self.Dec_y(s)

	# 通过均值和方差重参数化采样并获取预测结果
	def get_y_by_zs(self, mu, logvar, env):
		# 重参数化采样
		zs = self.reparametrize(mu, logvar)
		# 提取s变量
		s = zs[:, self.z_dim:]
		# 通过解码器获取预测结果
		return self.Dec_y(s)

	# 编码获取均值和方差
	def encode_mu_var(self, x, env_idx=0):
		# 返回指定条件下的均值和对数方差（z和s变量连接）
		return torch.cat([self.mean_z[env_idx](x), self.mean_s[env_idx](x)] ,dim=1), \
				torch.cat([self.logvar_z[env_idx](x), self.logvar_s[env_idx](x)], dim=1)

	# 编码获取先验分布
	def encode_prior(self, x, env_idx):
		# 创建环境索引的one-hot编码
		temp = env_idx * torch.ones(x.size()[0], 1)
		temp = temp.long().to(x.device)#.cuda()
		# 创建浮点形张量
		y_onehot = torch.FloatTensor(x.size()[0], self.total_env).to(x.device)#.cuda()
		# 元素初始化为0
		y_onehot.zero_()
		# 根据temp张量的索引值，将y_onehot张量的特定位置设置为1
		y_onehot.scatter_(1, temp, 1)  # (dim, temp, 填充值)
		# print(env_idx, y_onehot, 'onehot')
		# 通过先验编码器处理环境信息
		u = self.Enc_u_prior(y_onehot)
		#return self.mean_zs_prior(u), self.logvar_zs_prior(u)
		# 生成默认的s变量（随机噪声）
		default_s = torch.randn(x.size(0), self.s_dim).to(x.device)#.cuda()
		# 返回先验分布的均值和对数方差
		return torch.cat([self.mean_zs_prior(u)[:, :self.z_dim], default_s], dim=1), \
				torch.cat([self.logvar_zs_prior(u)[:, :self.z_dim], default_s], dim=1)

	# 解码重建图像（注意：此处Dec_x未定义？？？？）
	def decode_x(self, zs):
		return self.Dec_x(zs)

	# 解码获取分类结果
	def decode_y(self, s):
		return self.Dec_y(s)

	# 重参数化技巧
	def reparametrize(self, mu, logvar):
		# 计算标准差
		std = logvar.mul(0.5).exp_()  # 对数方差的标准差 exp(0.5ln(方差))
		# 根据是否使用cuda生成正态分布噪声
		if self.is_cuda:
			eps = torch.cuda.FloatTensor(std.size()).normal_()
		else:
			eps = torch.FloatTensor(std.size()).normal_()
		# eps = torch.FloatTensor(std.size()).normal_()
		# 应用重参数化公式: z = mu + std * eps
		return eps.mul(std).add_(mu)


	# 创建分类解码器
	def get_Dec_y(self):
		# 返回序列网络，包含全连接层、批归一化、ReLU激活和Softmax
		return nn.Sequential(
			# self.Fc_bn_ReLU(int(self.zs_dim), 512)
			# s变量维度
			self.Fc_bn_ReLU(int(self.zs_dim - self.z_dim), 512),
			self.Fc_bn_ReLU(512, 256),
			nn.Linear(256, self.num_classes),
			nn.Softmax(dim=1),
		)

	# 创建环境先验编码器
	def get_Enc_u(self):
		# 返回一个序列网络，将环境维度映射到32维
		return nn.Sequential(
			self.Fc_bn_ReLU(self.u_dim, 16),
			self.Fc_bn_ReLU(16, 32)
		)

	# 创建图像编码器（基于wide ResNet）
	def get_Enc_x_28(self, depth=28, num_classes=10, widen_factor=10, sub_block1=False, dropRate=0.0, bias_last=True):
		# nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
		# 定义各通道数
		nChannels = [16, 16 * widen_factor, 32 * widen_factor, self.zs_dim]
		# 验证网络深度（为啥是这样？）
		assert ((depth - 4) % 6 == 0)
		n = (depth - 4) / 6
		block = BasicBlock
		# 1st conv before any network block
		# 第一个卷积层
		self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
							padding=1, bias=False)
		# 1st block
		# 第一个卷积块，block参数为基础块的类型（BasicBlock、Bottleneck等）
		self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
		# 可选的子块
		if sub_block1:
			# 1st sub-block
			self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
		# 2nd block
		self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
		# 3rd block
		self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
		# global average pooling and classifier
		self.bn1 = nn.BatchNorm2d(nChannels[3])
		self.relu = nn.ReLU(inplace=True)
		self.fc = nn.Linear(nChannels[3], num_classes, bias=bias_last)
		self.nChannels = nChannels[3]

		# 权重初始化
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear) and not m.bias is None:
				m.bias.data.zero_()
		# 返回序列网络
		return nn.Sequential(
			self.conv1,
			self.block1,
			self.block2,
			self.block3,
			self.bn1,
			self.relu,
			# F.avg_pool2d(out, 8),
		# 	self.Conv_bn_ReLU(self.in_channel, 32),
		# 	nn.MaxPool2d(2),
		# 	self.Conv_bn_ReLU(32, 64),
		# 	nn.MaxPool2d(2),
		# 	self.Conv_bn_ReLU(64, 128),
		# 	nn.MaxPool2d(2),
		# 	self.Conv_bn_ReLU(128, 256),
			
			# 自适应平均池化到1x1？？？要干啥
			nn.AdaptiveAvgPool2d(1),
			Flatten(),
		)

	# 创建全连接+批归一化+ReLU
	def Fc_bn_ReLU(self, in_channels, out_channels):
		layer = nn.Sequential(
			nn.Linear(in_channels, out_channels),
			nn.BatchNorm1d(out_channels),
			nn.ReLU())
		return layer

# 实现互信息对比学习上界(Contrastive Learning Upper bound of Mutual Information)
class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
	'''
		This class provides the CLUB estimation to I(X,Y)
		Method:
			forward() :      provides the estimation with input samples  
			loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
		Arguments:
			x_dim, y_dim :         the dimensions of samples from X, Y respectively
			hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
			x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
	'''
	def __init__(self, x_dim, y_dim, hidden_size):
		super(CLUB, self).__init__()
		# p_mu outputs mean of q(Y|X)
		#print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
		# 输出 q(Y|X)的均值
		self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
									nn.ReLU(),
									nn.Linear(hidden_size//2, y_dim))
		# p_logvar outputs log of variance of q(Y|X)
		# 输出q(Y|x)的对数方差
		self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
									nn.ReLU(),
									nn.Linear(hidden_size//2, y_dim),
									nn.Tanh())

	# 获取均值和对数方差
	def get_mu_logvar(self, x_samples):
		mu = self.p_mu(x_samples)
		logvar = self.p_logvar(x_samples)
		return mu, logvar

	# 前向传播，计算互信息估计	
	def forward(self, x_samples, y_samples): 
		mu, logvar = self.get_mu_logvar(x_samples)
		
		# log of conditional probability of positive sample pairs
		# 正样本对的条件概率对数
		positive = - (mu - y_samples)**2 /2./logvar.exp()  
		
		prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
		y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

		# log of conditional probability of negative sample pairs
		# 负样本对的条件概率对数
		negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

		# 返回正负样本对数概率差的均值
		return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

	# 计算未归一化的对数似然（知识点）
	def loglikeli(self, x_samples, y_samples, index_mask=None): # unnormalized loglikelihood 
		mu, logvar = self.get_mu_logvar(x_samples)
		# 计算对数似然
		ll = (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1)
		# 如果提供了掩码，将对应位置设置为0
		if index_mask is not None:
			ll[index_mask] = 0.
		return ll.mean(dim=0)
	
	# 计算学习损失
	def learning_loss(self, x_samples, y_samples, index_mask=None):
		# 损失为负的对数似然
		return - self.loglikeli(x_samples, y_samples, index_mask)

# BeatGANsAutoencModel类：实现基于BeatGAN的自动编码器模型
class BeatGANsAutoencModel(BeatGANsUNetModel):
	def __init__(self, conf: BeatGANsAutoencConfig):
		super().__init__(conf)
		self.conf = conf

		# print('conf: ')
		# print(conf.model_channels)
		# print(conf.embed_channels)
		# having only time, cond
		
		# 时间嵌入网络，仅处理时间和条件
		self.time_embed = TimeStyleSeperateEmbed(
			time_channels=conf.model_channels,
			time_out_channels=conf.embed_channels,
		)
		# self.time_embed = nn.Sequential(
		# 	linear(self.time_emb_channels, conf.embed_channels),
		# 	nn.SiLU(),
		# 	linear(conf.embed_channels, conf.embed_channels),
		# )

		# self.encoder = BeatGANsEncoderConfig(
		# 	image_size=conf.image_size,
		# 	in_channels=conf.in_channels,
		# 	model_channels=conf.model_channels,
		# 	out_hid_channels=conf.enc_out_channels,
		# 	out_channels=conf.enc_out_channels,
		# 	num_res_blocks=conf.enc_num_res_block,
		# 	attention_resolutions=(conf.enc_attn_resolutions
		# 						or conf.attention_resolutions),
		# 	dropout=conf.dropout,
		# 	channel_mult=conf.enc_channel_mult or conf.channel_mult,
		# 	use_time_condition=False,
		# 	conv_resample=conf.conv_resample,
		# 	dims=conf.dims,
		# 	use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
		# 	num_heads=conf.num_heads,
		# 	num_head_channels=conf.num_head_channels,
		# 	resblock_updown=conf.resblock_updown,
		# 	use_new_attention_order=conf.use_new_attention_order,
		# 	pool=conf.enc_pool,
		# ).make_model()

		# 如果配置了潜在网络，则创建潜在网络
		if conf.latent_net_conf is not None:
			self.latent_net = conf.latent_net_conf.make_model()
		
		# 创建LaCIM模型实例
		self.lacim = d_LaCIM(in_channel=3,
							zs_dim=self.conf.enc_out_channels,
							num_classes=10,
							decoder_type=1,
							total_env=1,)
		# self.conf.use_club = True
		# 如果配置使用CLUB，则创建CLUB实例
		if self.conf.use_club:
			# print('club built!!!')
			self.club = CLUB(x_dim=self.lacim.z_dim, 
							y_dim=self.lacim.s_dim,
							hidden_size=self.conf.club_hidden_dim,)


	# def encode_mu_var(self, x, env_idx=0):
	# 	# print(x.size())
	# 	# print(self.mean_z[env_idx](x).size())
	# 	# print(self.mean_s[env_idx](x).size())
	# 	# print(self.logvar_z[env_idx](x).size())
	# 	# print(self.logvar_s[env_idx](x).size())
	# 	return torch.cat([self.mean_z[env_idx](x), self.mean_s[env_idx](x)] ,dim=1), \
	# 	torch.cat([self.logvar_z[env_idx](x), self.logvar_s[env_idx](x)], dim=1)

	# def encode_prior(self, x, env_idx=0):
	# 	temp = env_idx * torch.ones(x.size()[0], 1)
	# 	temp = temp.long().to(x.device)#.cuda()
	# 	# y_onehot = torch.FloatTensor(x.size()[0], self.args.env_num).to(x.device)#.cuda()
	# 	y_onehot = torch.FloatTensor(x.size()[0], self.total_env).to(x.device)#.cuda()
	# 	y_onehot.zero_()
	# 	y_onehot.scatter_(1, temp, 1)
	# 	# print(env_idx, y_onehot, 'onehot')
	# 	u = self.Enc_u_prior(y_onehot)
	# 	#return self.mean_zs_prior(u), self.logvar_zs_prior(u)
	# 	default_s = torch.randn(x.size(0), self.s_dim).to(x.device)#.cuda()
	# 	return torch.cat([self.mean_zs_prior(u)[:, :self.z_dim], default_s], dim=1), \
	# 			torch.cat([self.logvar_zs_prior(u)[:, :self.z_dim], default_s], dim=1)

	
	# 重参数化方法
	def reparameterize(self, mu, logvar):
		# 计算标准差
		std = logvar.mul(0.5).exp_()
		# 生成cuda上的正态分布噪声
		eps = torch.cuda.FloatTensor(std.size()).normal_()
		# eps = torch.FloatTensor(std.size()).normal_()
		# 应用重参数化公式
		return eps.mul(std).add_(mu)

	# 采样潜在变量z
	def sample_z(self, n: int, device):
		# 确保模型是随机的
		assert self.conf.is_stochastic
		# 返回标准正态分布采样
		return torch.randn(n, self.conf.enc_out_channels, device=device)

	# 噪声转条件（未实现）
	def noise_to_cond(self, noise: Tensor):
		raise NotImplementedError()
		assert self.conf.noise_net_conf is not None
		return self.noise_net.forward(noise)

	# 编码方法
	def encode(self, x):
		# if r is None:
		# 	r = torch.zeros(x.size(0))
		# 通过LaCIM的图像编码器提取特征
		x = self.lacim.Enc_x(x)
		# 编码获取均值和方差（使用环境0）
		mu, logvar = self.lacim.encode_mu_var(x, env_idx=0)
		# 编码获取先验分布
		mu_prior, logvar_prior = self.lacim.encode_prior(x, env_idx=0)
		# mu, logvar = [self.lacim.encode_mu_var(x, env_idx=i) for i in range(self.lacim.total_env)]
		# mu = torch.stack([mu[env_idx][i] for i, env_idx in enumerate(r)])
		# logvar = torch.stack([logvar[env_idx][i] for i, env_idx in enumerate(r)])

		# mu_prior, logvar_prior = [self.lacim.encode_prior(x, env_idx=i) for i in range(self.lacim.total_env)]
		# mu_prior = torch.stack([mu_prior[env_idx][i] for i, env_idx in enumerate(r)])
		# logvar_prior = torch.stack([logvar_prior[env_idx][i] for i, env_idx in enumerate(r)])

		# 重参数化获取条件
		cond = self.reparameterize(mu, logvar)
		# z = zs[:, :self.z_dim]
		# s = zs[:, self.z_dim:]
		# pred_y = self.Dec_y(zs)
		# 返回编码字典结果
		return {'cond': cond, 'mu':mu, 'logvar':logvar, 'mu_prior':mu_prior, 'logvar_prior':logvar_prior}
	
	# 解码获取分类结果
	def dec_y(self, s):
		return self.lacim.Dec_y(s)

	# 获取style空间大小属性
	@property
	def stylespace_sizes(self):
		# 获取所有输入块、中间块和输出块的模板
		modules = list(self.input_blocks.modules()) + list(
			self.middle_block.modules()) + list(self.output_blocks.modules())
		sizes = []
		# 遍历模块，查找ResBlock并获取条件嵌入层的权重形状
		for module in modules:
			if isinstance(module, ResBlock):
				linear = module.cond_emb_layers[-1]
				sizes.append(linear.weight.shape[0])
		return sizes

	# 编码到style空间
	def encode_stylespace(self, x, return_vector: bool = True):
		"""
		encode to style space
		"""
		# 获取所有输入块、中间块和输出块的模板
		modules = list(self.input_blocks.modules()) + list(
			self.middle_block.modules()) + list(self.output_blocks.modules())
		# (n, c)
		# 通过编码器前向传播
		cond = self.encoder.forward(x)
		S = []
		# 遍历模块，查找ResBlock并获取style表示
		for module in modules:
			if isinstance(module, ResBlock):
				# (n, c')
				# 通过条件嵌入层前向传播
				s = module.cond_emb_layers.forward(cond)
				S.append(s)

		# 根据参数决定返回向量还是列表
		if return_vector:
			# (n, sum_c)
			return torch.cat(S, dim=1)
		else:
			return S

	def forward(self,
				x,
				t,
				y=None,
				x_start=None,  # 起始图像
				cond=None,  # 条件向量
				style=None,  # style条件
				noise=None,
				t_cond=None,  # 条件时间步
				**kwargs):
		"""
		Apply the model to an input batch.

		Args:
			x_start: the original image to encode
			cond: output of the encoder
			noise: random noise (to predict the cond)
		"""

		if t_cond is None:
			t_cond = t

		if noise is not None:
			# if the noise is given, we predict the cond from noise
			cond = self.noise_to_cond(noise)  # 通过噪声预测条件

		if cond is None:
			mode = 'train'
			# 如果提供了x，则验证长度一致性
			if x is not None:
				assert len(x) == len(x_start), f'{len(x)} != {len(x_start)}'

			# 编码其实图像获取条件
			tmp = self.encode(x_start)
			cond = tmp['cond']

		if t is not None:
			# 进行时间嵌入
			_t_emb = timestep_embedding(t, self.conf.model_channels)
			_t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)
		else:
			# this happens when training only autoenc
			_t_emb = None
			_t_cond_emb = None

		# 如果配置使用双条件，则通过时间嵌入网络前向传播
		if self.conf.resnet_two_cond:
			res = self.time_embed.forward(
				time_emb=_t_emb,
				cond=cond,
				time_cond_emb=_t_cond_emb,
			)
		else:
			raise NotImplementedError()

		# 根据配置分离时间嵌入和条件嵌入
		if self.conf.resnet_two_cond:
			# two cond: first = time emb, second = cond_emb
			emb = res.time_emb
			cond_emb = res.emb
		else:
			# one cond = combined of both time and cond
			emb = res.emb
			cond_emb = None

		# if cond.requires_grad:
		# 	print('cond_emb grad: ')
		# 	print(autograd.grad(cond_emb, cond, torch.ones_like(cond_emb)))

		# override the style if given
		# 覆盖style条件（如果提供）
		style = style or res.style

		# 验证类别标签和类别数的一致性
		assert (y is not None) == (
			self.conf.num_classes is not None
		), "must specify y if and only if the model is class-conditional"

		if self.conf.num_classes is not None:
			raise NotImplementedError()  # 抛出异常，表示类别功能尚未实现
			# assert y.shape == (x.shape[0], )
			# emb = emb + self.label_emb(y)

		# where in the model to supply time conditions
		# 在模型中提供时间条件的位置
		enc_time_emb = emb
		mid_time_emb = emb
		dec_time_emb = emb
		# where in the model to supply style conditions
		# print('cond in unet_autoenc : ')
		# cond_emb = None
		# print(cond_emb)

		# cond_emb_gen = cond_emb.clone()
		# if mode == 'train' and self.conf.mask_threshold is not None:
		# 	index_mask = np.where(np.random.rand(x_start.size(0)) < self.conf.mask_threshold)
		# 	cond_emb_gen[index_mask] = 0

		# cond_emb = None
		enc_cond_emb = cond_emb
		mid_cond_emb = cond_emb
		dec_cond_emb = cond_emb

		# index_mask = None
		# # index_mask = np.where(np.random.rand(cond_emb.size(0)) <= self.conf.mask_threshold)
		# index_mask = np.where(np.random.rand(cond_emb.size(0)) <= 1.)

		# hs = []
		# 为每个通信倍增因子创建空列表
		hs = [[] for _ in range(len(self.conf.channel_mult))]

		if x is not None:
			# 转换x的数据类型
			h = x.type(self.dtype)

			# input blocks
			k = 0
			for i in range(len(self.input_num_blocks)):
				for j in range(self.input_num_blocks[i]):
					# modify zero arch
					# if k > 0:
					# 	tmp_zero = torch.zeros_like(self.input_blocks[k][0].out_layers[3].weight)
					# 	if torch.equal(tmp_zero, self.input_blocks[k][0].out_layers[3].weight):
					# 		torch.nn.init.normal_(self.input_blocks[k][0].out_layers[3].weight, 0, std)

					# tmp_zero_att = torch.zeros_like(self.input_blocks[4][1].proj_out.weight)
					# if torch.equal(tmp_zero_att, self.input_blocks[4][1].proj_out.weight):
					# 	torch.nn.init.normal_(self.input_blocks[4][1].proj_out.weight, 0, std)
					
					# tmp_zero_att = torch.zeros_like(self.input_blocks[5][1].proj_out.weight)
					# if torch.equal(tmp_zero_att, self.input_blocks[5][1].proj_out.weight):
					# 	torch.nn.init.normal_(self.input_blocks[5][1].proj_out.weight, 0, std)

					# print('enc_cond_emb size : ')
					# print(enc_cond_emb.size())
					
					# 通过输入块前向转播
					h = self.input_blocks[k](h,
											emb=enc_time_emb,
											cond=enc_cond_emb)

					# print(i, j, h.shape)
					# 将输出添加到对应层级的列表中
					hs[i].append(h)
					k += 1
			# 验证处理的块数是否正确
			assert k == len(self.input_blocks)

			# modify zero arch
			# tmp_zero = torch.zeros_like(self.middle_block[0].out_layers[3].weight)
			# if torch.equal(tmp_zero, self.middle_block[0].out_layers[3].weight):
			# 	torch.nn.init.normal_(self.middle_block[0].out_layers[3].weight, 0, std)
			# tmp_zero = torch.zeros_like(self.middle_block[2].out_layers[3].weight)
			# if torch.equal(tmp_zero, self.middle_block[2].out_layers[3].weight):
			# 	torch.nn.init.normal_(self.middle_block[2].out_layers[3].weight, 0, std)
			# tmp_zero = torch.zeros_like(self.middle_block[1].proj_out.weight)
			# if torch.equal(tmp_zero, self.middle_block[1].proj_out.weight):
			# 	torch.nn.init.normal_(self.middle_block[1].proj_out.weight, 0, std)

			# middle blocks
			h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
			# if cond.requires_grad:
			# 	print('h grad: ')
			# 	print(autograd.grad(h, cond, torch.ones_like(h)))
		# 如果没有提供x（仅训练自动编码器时）
		else:
			# no lateral connections
			# happens when training only the autonecoder
			h = None
			hs = [[] for _ in range(len(self.conf.channel_mult))]

		# output blocks 输出块处理
		k = 0
		for i in range(len(self.output_num_blocks)):
			for j in range(self.output_num_blocks[i]):
				# take the lateral connection from the same layer (in reserve)
				# until there is no more, use None
				# 从相同层数获取横向连接（反向）
				try:
					lateral = hs[-i - 1].pop()
					# print(i, j, lateral.shape)
				except IndexError:
					lateral = None
					# print(i, j, lateral)
				
				# modify zero arch
				# tmp_zero = torch.zeros_like(self.output_blocks[k][0].out_layers[3].weight)
				# if torch.equal(tmp_zero, self.output_blocks[k][0].out_layers[3].weight):
				# 	torch.nn.init.normal_(self.output_blocks[k][0].out_layers[3].weight, 0, std)

				# 通过输出块前向传播
				h = self.output_blocks[k](h,
										emb=dec_time_emb,
										cond=dec_cond_emb,
										lateral=lateral,)
				k += 1

		pred = self.out(h)
		# 返回自动编码结果
		return AutoencReturn(pred=pred, cond=cond)


# 定义自动编码器返回值的命名元组
class AutoencReturn(NamedTuple):
	pred: Tensor
	cond: Tensor = None


# 嵌入返回
class EmbedReturn(NamedTuple):
	# style and time
	emb: Tensor = None
	# time only
	time_emb: Tensor = None
	# style only (but could depend on time)
	style: Tensor = None


# 分别处理时间和style嵌入
class TimeStyleSeperateEmbed(nn.Module):
	# embed only style
	def __init__(self, time_channels, time_out_channels):
		super().__init__()
		# 时间嵌入网络
		self.time_embed = nn.Sequential(
			linear(time_channels, time_out_channels),
			nn.SiLU(),
			linear(time_out_channels, time_out_channels),
		)
		# style嵌入（恒等变换）
		self.style = nn.Identity()

	def forward(self, time_emb=None, cond=None, **kwargs):
		if time_emb is None:
			# happens with autoenc training mode
			time_emb = None
		else:
			# 通过时间嵌入网络处理时间嵌入
			time_emb = self.time_embed(time_emb)
		# 通过style嵌入处理条件
		style = self.style(cond)
		# 返回嵌入结果
		return EmbedReturn(emb=style, time_emb=time_emb, style=style)


# 解耦时间和style嵌入
class TimeStyleSeperateEmbed_decouple(nn.Module):
	# embed only style
	def __init__(self, time_channels, time_out_channels):
		super().__init__()
		# 时间嵌入网络
		self.time_embed = nn.Sequential(
			linear(time_channels, time_out_channels),
			nn.SiLU(),
			linear(time_out_channels, time_out_channels),
		)
		# style嵌入（恒等变换：恒等函数，不做任何变换，个人理解：用作定义样式）
		self.style = nn.Identity()

	def forward(self, time_emb=None, cond=None, index_mask=None, **kwargs):
		assert cond is not None
		z = cond[:, :256]
		s = cond[:, 256:]
		s_copy = torch.clone(s)
		# assert index_mask is not None
		if index_mask is not None and len(index_mask[0])>0:
			s_copy[index_mask] *= 0.
		if time_emb is None:
			# happens with autoenc training mode
			time_emb = None
		else:
			# 通过时间嵌入网络处理时间嵌入
			time_emb = self.time_embed(time_emb)
		# 将处理后的s加到时间嵌入上
		time_emb += s_copy
		# 通过style嵌入处理z
		style = self.style(z)
		# 返回嵌入结果
		return EmbedReturn(emb=style, time_emb=time_emb, style=style)
