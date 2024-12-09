import os
import argparse
import numpy as np
import tqdm
import sys
import matplotlib.pyplot as plt
import random
import copy

from torchinfo import summary
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

from dataloader import load_pseudo_label_data, load_data
from clip_custom import clip
from model import Custom_Clip, PromptGenerator
from utils import disable_running_stats, enable_running_stats
from dataset import SingleSourceDataset, MultiSourceDataset
from samplers import RandomDomainSampler
import ot

def arg_parse():
	parser = argparse.ArgumentParser("Training and Evaluation Script", add_help=False)

	# for config
	parser.add_argument(
		"--data_root", type=str, default="/mnt/SSD1/datasets/DG_data/DomainBed/", help="data file path",
	)
	parser.add_argument("--backbone", type=str, default="RN101", help="")
	parser.add_argument("--dataset", type=str, default="ImageCLEF", help="")
	parser.add_argument("--seed", type=int, default=1, help="")

	# for dataloader
	parser.add_argument("--batch_size", type=int, default=30, help="")
	parser.add_argument("--num_workers", type=int, default=4, help="")
	parser.add_argument("--pin_memory", type=bool, default=True, help="")
	parser.add_argument(
		"--threshold",
		type=float,
		default=0.4,
		help="threshold tau for generating pseudo labels",
	)

	# for prompt settings
	parser.add_argument("--M1", type=int, default=16, help="number of classification tokens")
	parser.add_argument("--M2", type=int, default=16, help="number of domain tokens")

	# for training settings
	parser.add_argument("--training_mode", type=str, default="multi-source", help="multi-source, source-combined")
	parser.add_argument("--prompt_iteration", type=int, default=5000, help="")
	parser.add_argument("--prompt_learning_rate", type=float, default=0.005, help="")
	parser.add_argument("--ot_t_weight", type=float, default=0.5, help="")
	parser.add_argument("--t_weight", type=float, default=0.5, help="")
	parser.add_argument("--w_scale", type=float, default=10, help="")
	parser.add_argument("--evaluation_step", type=int, default=50, help="")
	parser.add_argument("--output_dir", type=str, default="outputs", help="")

	parser.add_argument("--OT_clustering", type=int, default=1, help="")
	parser.add_argument("--enhanced_pseudo_label", type=int, default=1, help="")
	parser.add_argument("--entropy_tradeoff", type=float, default=0.0, help="")



	return parser



def entropy_loss(logits):
	p = F.softmax(logits, dim=-1)
	log_p = F.log_softmax(logits, dim=-1)
	loss = -torch.sum(p * log_p, dim=-1)
	return loss.mean()

def args_update(args):
	if args.dataset == "ImageCLEF":
		args.backbone = "RN50"
		args.prompt_iteration = 400
		
	if args.dataset == "S2RDA":
		# Debugging
		args.backbone = "RN50"
		args.prompt_iteration = 0

	if args.dataset == "Office31":
		args.backbone = "RN50"
		args.prompt_iteration = 600

	if args.dataset == "DomainNet":
		args.backbone = "RN101"
		args.prompt_iteration = 4000

	if args.dataset == "OfficeHome":
		args.backbone = "RN50"
		args.prompt_iteration = 1000

	if args.dataset == "PACS":
		args.backbone = "RN18"
		args.prompt_iteration = 800

	if args.dataset == "ViT_ImageCLEF":
		args.backbone = "ViT-B/16"
		args.prompt_iteration = 400
	
	if args.dataset == "ViT_OfficeHome":
		args.backbone = "ViT-B/16"
		args.prompt_iteration = 1000

	if args.dataset == "ViTL_ImageCLEF":
		args.backbone = "ViT-L/14"
		args.prompt_iteration = 400
	
	if args.dataset == "ViTL_OfficeHome":
		args.backbone = "ViT-L/14"
		args.prompt_iteration = 1000


def ema(source, target, count):
	source_dict = source.state_dict()
	target_dict = target.state_dict()
	if count == 1:
		for key in source_dict.keys():
			target_dict[key].data.copy_(source_dict[key].data)
	else:
		for key in source_dict.keys():
			target_dict[key].data.copy_(
				(target_dict[key].data * (count-1) + source_dict[key].data) / count)


def test(target_test_loader, custom_clip_model, text_embeddings, tokenized_prompts, args):
	scale = custom_clip_model.logit_scale.exp()

	correct = 0
	tot = 0
	with torch.no_grad():
		for data, label in target_test_loader:
			tot += args.batch_size
			data = data.to(args.device)
			label = label.to(args.device)

			tot_logits = 0

			img_feature = custom_clip_model.forward_img(data)
			for text_embedding in text_embeddings:
				logits = img_feature @ text_embedding.t()
				tot_logits += logits
			tot_logits /= len(text_embeddings)

			output = torch.argmax(tot_logits, dim=1) 

			correct += (output == label).sum().item()

	return correct / tot


def soft_cross_entropy_loss(predictions, soft_targets):

	# Apply softmax to get predicted probabilities
	log_probs = F.log_softmax(predictions, dim=1)
	
	# Compute cross-entropy loss using soft labels
	loss = -torch.sum(soft_targets * log_probs, dim=1).mean()
	return loss


def calc_distance(A, B):
	distances = (torch.sum(A**2, dim=1, keepdim=True) 
				+ torch.sum(B**2, dim=1)
				- 2 * torch.matmul(A, B.t()))
	return distances


def train(domain_list, classnames, clip_model, preprocess, args):
	n_cls = len(classnames)
	custom_clip_model = Custom_Clip(clip_model)
	
	cos = nn.CosineSimilarity(dim=0, eps=1e-6)
	
	for name, param in custom_clip_model.named_parameters():
		param.requires_grad_(False)
	
	print("Custom_Clip", summary(custom_clip_model))
	best_accs = []

	feature_dim = 1024
	if args.dataset == "DomainNet":
		feature_dim = 512
	print('feature_dim', feature_dim)

	for target_name in domain_list:
		print("*" * 50)
		print("Start training on {}".format(target_name))
		tgt_save_path = os.path.join(args.output_dir, target_name)
		os.makedirs(tgt_save_path, exist_ok=True)
		orig_stdout = sys.stdout
		f = open(tgt_save_path+ "/train.log", "w+")
		sys.stdout = f

	
		source_name_list = domain_list.copy()
		source_name_list.remove(target_name)
		print(source_name_list)
		print(target_name)

		target_path = os.path.join(args.data_root, target_name)

		# if args.enhanced_pseudo_label == 1:
		# 	target_train_dataset = SingleSourceDataset(
		# 		args.data_root, [target_name], preprocess
		# 	)
		# 	target_train_loader = torch.utils.data.DataLoader(
		# 		target_train_dataset,
		# 		batch_size=args.batch_size,
		# 		num_workers=4,
		# 		pin_memory=True,
		# 	)	
		# else:
		# Generate pseudo-labels from CLIP zero-shot predictions with a threshold.
		target_train_loader = load_pseudo_label_data(
			target_name, target_path, preprocess, clip_model, args
		)

		target_test_loader = load_data(target_path, preprocess, args)

		source_train_dataset = MultiSourceDataset(
			args.data_root, source_name_list, preprocess
		)
		
		sampler = RandomDomainSampler(
			source_train_dataset.data, args.batch_size, len(source_name_list)
		)

		source_train_loader = torch.utils.data.DataLoader(
			source_train_dataset,
			batch_size=args.batch_size,
			sampler=sampler,
			num_workers=4,
			pin_memory=True,
		)

		scale = custom_clip_model.logit_scale.exp()
		prompt_learner = PromptGenerator(
			classnames, clip_model, source_name_list, target_name, args, feature_dim
		)

		print("PromptGenerator", summary(prompt_learner))
		tokenized_prompts = prompt_learner.tokenized_prompts

		optimizer = torch.optim.AdamW(
			list(prompt_learner.parameters()), 
			lr=args.prompt_learning_rate
		)
		scheduler = CosineAnnealingLR(optimizer, T_max=args.prompt_iteration)
		# ema_model = copy.deepcopy(prompt_learner)

		for name, param in prompt_learner.named_parameters():
			print(
				f"name: {name}, shape {param.shape}, require grad: {param.requires_grad}"
			)

		# Base prompts
		class_list_tokenize = []
		for i in range(len(classnames)):
			class_list_tokenize.append(f"A photo of a {classnames[i]}")
		text = clip.tokenize(class_list_tokenize).to(args.device)
		
		# Use the running mean to represent the centroid of each domain class
		n_domains = len(source_name_list)
		running_means = torch.zeros(n_domains, n_cls, feature_dim).to(args.device)
		running_count = torch.zeros(n_domains, n_cls).to(args.device)

		best_acc = 0
		pbar = tqdm.tqdm(range(1, args.prompt_iteration + 1))
		for step in pbar:
			source_prompts, target_prompts = prompt_learner()

			try:
				target_data, target_label = next(target_iter)
			except Exception as err:
				target_iter = iter(target_train_loader)
				target_data, target_label = next(target_iter)

			try:
				source_data, source_label, source_domain = next(source_iter)
			except Exception as err:
				source_iter = iter(source_train_loader)
				source_data, source_label, source_domain = next(source_iter)

			target_data = target_data.to(args.device)
			target_label = target_label.to(args.device)
			source_data = source_data.to(args.device)
			source_label = source_label.to(args.device)
			source_domain = source_domain.to(args.device)

			optimizer.zero_grad()
			
			total_loss = 0

			# Training source domains
			source_img_features, source_txt_features, pre_norm_source_img_features = custom_clip_model(
				source_data, source_prompts, tokenized_prompts
			)
			source_logits = source_img_features @ source_txt_features.t()
			source_loss = F.cross_entropy(scale * source_logits, source_label)
			total_loss += source_loss

			
			# If the training mode is set to 'multi-source,' domain labels are available.
			if args.training_mode =='multi-source':
				image_feature_list = []
				text_feature_list = []
				label_list = []
				for source_index in range(n_domains):
					source_prompt_idx = prompt_learner.forward_source(source_index=source_index)
					txt_feature_idx = custom_clip_model.forward_txt(source_prompt_idx, tokenized_prompts)
					text_feature_list.append(txt_feature_idx)
					label_list.append(source_label[source_domain==source_index])
					image_feature_list.append(source_img_features[source_domain==source_index])

					# Use the running mean to represent the centroid of each domain class
					# Store the running mean for each domain class to enable distance-aware pseudo-labeling.
					pre_norm_features = pre_norm_source_img_features[source_domain==source_index]
					unique_cls = torch.unique(label_list[source_index])
					for d_cls in unique_cls:
						cls_features = pre_norm_features[label_list[source_index]==d_cls]
						running_means[source_index][d_cls] = running_means[source_index][ d_cls] * running_count[source_index][ d_cls] + cls_features.sum(0)
						running_count[source_index][ d_cls] += cls_features.shape[0]
						running_means[source_index][ d_cls] /= running_count[source_index][ d_cls]
				
				source_loss_idx = 0
				n_source = 0
				for source_index in range(n_domains):
					source_img_features_idx = image_feature_list[source_index]
					source_txt_features_idx = text_feature_list[source_index]
					source_logits_idx = source_img_features_idx @ source_txt_features_idx.t()
					source_loss_idx += F.cross_entropy(scale * source_logits_idx, label_list[source_index])
					n_source += 1
				total_loss += source_loss_idx / n_source

			# Training target domains
			target_img_features, target_txt_features, pre_norm_target_img_features = custom_clip_model(
				target_data, target_prompts, tokenized_prompts
			)
			target_logits = target_img_features @ target_txt_features.t()

			# Using enhanced pseudo-label
			if args.enhanced_pseudo_label == 1:
				logits = 0
				n_pseudo = 0

				# CLIP zero-shot prediction
				_, text_features = clip_model(target_data, text)
				logits += target_img_features @ text_features.t()
				n_pseudo += 1

				# Using pseudo-label from source-combined
				logits += target_img_features @ source_txt_features.t()
				n_pseudo += 1
				
				if args.training_mode =='multi-source':
					# Compute the distance when all domain classes are observed
					if (running_count>0).sum().sum() == running_count.shape[0]*running_count.shape[1]:
						distance = torch.zeros(running_means.shape[0], target_img_features.shape[0], n_cls).to(args.device)
						for source_index in range(running_means.shape[0]):
							distance[source_index] = calc_distance(pre_norm_target_img_features, running_means[source_index])
						weights = nn.Softmax(dim=0)(-distance*args.w_scale).detach()

						# Distance-aware pseudo-label
						for source_index in range(n_domains):
							source_txt_features_idx = text_feature_list[source_index]
							logits +=  weights[source_index] * (target_img_features @ source_txt_features_idx.t())
						n_pseudo += 1

				logits /= n_pseudo
				probs = (scale *logits).softmax(dim=-1).detach()
				target_cls_loss = soft_cross_entropy_loss(scale * target_logits, probs)
				n_target = 1
			else:
				# BASELINE: Utilize pseudo-labels generated from CLIP zero-shot predictions with a threshold.
				# Cross entropy loss for those that have non -1 labels
				target_cls_loss = F.cross_entropy(
					target_logits[target_label != -1], target_label[target_label != -1]
				)
				n_target = 1
			
			if args.entropy_tradeoff > 0:
				target_entropy_loss = entropy_loss(target_logits[target_label == -1])
				total_loss +=  args.entropy_tradeoff * target_entropy_loss

			# Clustering Reinforcement with Wasserstein Distance
			if args.OT_clustering == 1:
				cost_metric = 1 - target_logits
				BTI = target_img_features.shape[0]
				BTT = target_txt_features.shape[0]
				sample_weight_i = torch.ones(BTI).to(args.device) / BTI
				sample_weight_t = torch.ones(BTT).to(args.device) / BTT
				t_ot_cost = ot.emd2(sample_weight_i, sample_weight_t, cost_metric, numItermax=500000)
				nearest_centroids = cost_metric.min(1)[1]
				target_cls_loss += F.cross_entropy(
					scale * target_logits, nearest_centroids
				)
				n_target += 1
				total_loss += args.ot_t_weight * t_ot_cost
			
			target_loss = target_cls_loss / n_target
			total_loss += args.t_weight * target_loss
			
			
			total_loss.backward()
			optimizer.step()

			if step % (args.prompt_iteration / 20) == 0:
				scheduler.step()

			if args.dataset == "DomainNet":
				if step < 3000:
					continue

			if args.dataset == "DomainNet":
				if step > 3000:
					
					prompt_learner.features[-1] += target_txt_features 
					prompt_learner.features[-2] += source_txt_features 
					prompt_learner.count[-1] += 1
					prompt_learner.count[-2] += 1
					if args.training_mode =='multi-source':
						for source_index in range(n_domains):
							prompt_learner.features[source_index] += text_feature_list[source_index] 
							prompt_learner.count[source_index] += 1
			else:
				if step > 100:
					
					prompt_learner.features[-1] += target_txt_features 
					prompt_learner.features[-2] += source_txt_features
					prompt_learner.count[-1] += 1
					prompt_learner.count[-2] += 1
					if args.training_mode =='multi-source':
						for source_index in range(n_domains):
							prompt_learner.features[source_index] += text_feature_list[source_index] 
							prompt_learner.count[source_index] += 1


			# if prompt_learner.count[-1] > 0:
			# 	ema(prompt_learner, ema_model, prompt_learner.count[-1])

			
			if step% args.evaluation_step:
				continue
			
			
			acc = test(
				target_test_loader,
				custom_clip_model,
				[target_txt_features],
				tokenized_prompts,
				args,
			)

			pbar.set_description(
				f"step: {step}, accuracy: {acc}, target total loss: {target_loss.item()}, classification: {target_cls_loss.item()}"
			)
			if acc > best_acc:
				best_acc = acc
			print(f"Best accuracy so far: {best_acc}, step {step}, target accuracy {acc}")
		

		# Finish training, start evaluation
		torch.save({
			'prompt': prompt_learner.state_dict()}, 
			os.path.join(tgt_save_path, 'last.pth'))

		# # ema_model
		# source_prompts, target_prompts = ema_model()
		# source_txt_features =  custom_clip_model.forward_txt(source_prompts, tokenized_prompts)
		# source_acc = test(
		# 		target_test_loader,
		# 		custom_clip_model,
		# 		[source_txt_features],
		# 		tokenized_prompts,
		# 		args,
		# 	)
		# print('Invariant ema: ', source_acc)

		# target_txt_features =  custom_clip_model.forward_txt(target_prompts, tokenized_prompts)
		# source_acc = test(
		# 		target_test_loader,
		# 		custom_clip_model,
		# 		[target_txt_features],
		# 		tokenized_prompts,
		# 		args,
		# 	)
		# print('target ema: ', source_acc)

		source_prompts, target_prompts = prompt_learner()
		source_txt_features =  custom_clip_model.forward_txt(source_prompts, tokenized_prompts)

		source_acc = test(
				target_test_loader,
				custom_clip_model,
				[source_txt_features],
				tokenized_prompts,
				args,
			)
		print('Invariant last: ', source_acc)

		target_txt_features =  custom_clip_model.forward_txt(target_prompts, tokenized_prompts)
		source_acc = test(
				target_test_loader,
				custom_clip_model,
				[target_txt_features],
				tokenized_prompts,
				args,
			)
		print('target last: ', source_acc)

		target_prompts = prompt_learner.features[-1]/prompt_learner.count[-1]
		source_acc = test(
			target_test_loader,
			custom_clip_model,
			[target_prompts],
			tokenized_prompts,
			args,
		)
		print('average_target_weight: ', source_acc)

		source_prompts = prompt_learner.features[-2] / prompt_learner.count[-2]
		source_acc = test(
			target_test_loader,
			custom_clip_model,
			[source_prompts],
			tokenized_prompts,
			args,
		)
		print('average_source_weight: ', source_acc)
		
		
		acc = test(
				target_test_loader,
				custom_clip_model,
				[source_prompts, target_prompts],
				tokenized_prompts,
				args,
			)
		print('Combined Inv + Target: ', acc)

		if args.training_mode =='multi-source':
			prompt_list = []
			for source_index in range(n_domains):
				# source_prompt_idx = ema_model.forward_source(source_index=source_index)
				
				source_prompt_idx = prompt_learner.features[source_index] / prompt_learner.count[source_index]
				acc = test(
					target_test_loader,
					custom_clip_model,
					[source_prompt_idx],
					tokenized_prompts,
					args,
				)
				print('source: ', source_index, ': ', acc)
				prompt_list.append(source_prompt_idx)

			acc = test(
					target_test_loader,
					custom_clip_model,
					prompt_list,
					tokenized_prompts,
					args,
				)
			print('Combined source: ', acc)

			acc = test(
					target_test_loader,
					custom_clip_model,
					prompt_list + [source_prompts],
					tokenized_prompts,
					args,
				)
			print('Combined Inv + source: ', acc)

			acc = test(
					target_test_loader,
					custom_clip_model,
					prompt_list + [target_prompts],
					tokenized_prompts,
					args,
				)

			print('Combined target + source: ', acc)
		
			acc = test(
					target_test_loader,
					custom_clip_model,
					prompt_list + [source_prompts, target_prompts],
					tokenized_prompts,
					args,
				)

			print('All: ', acc)

		best_accs.append(best_acc)
		print("Best accuracy for each domain:", best_accs, "Average:", np.mean(best_accs))
		sys.stdout = orig_stdout
		f.close()
	
def fix_random_seed(seed_value):
	random.seed(seed_value)
	np.random.seed(seed_value)
	torch.manual_seed(seed_value)

	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed_value)
		torch.cuda.manual_seed(seed_value)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True
	

def main(args):
	args_update(args)
	print(args)
	args.device = "cuda" if torch.cuda.is_available() else "cpu"
	
	fix_random_seed(args.seed)

	model, preprocess = clip.load(args.backbone, device=args.device)
	model.float()
	args.data_root = args.data_root + args.dataset +'/'
	domain_list = os.listdir(args.data_root)

	domain_list = [x for x in domain_list if ".txt" not in x]
	domain_list.sort()
	classnames_path = os.path.join(args.data_root, domain_list[0])

	classnames = os.listdir(classnames_path)
	n_cls = len(classnames)
	classnames.sort()

	args.output_dir = args.output_dir + str(args).replace(", ", "/").replace(
		"'", ""
	).replace("(", "").replace(")", "").replace("Namespace", "")

	print("Output directory:", args.output_dir)
	os.makedirs(args.output_dir, exist_ok=True)

	args.n_cls = n_cls
	train(domain_list, classnames, model, preprocess, args)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		"Training and Evaluation Script", parents=[arg_parse()]
	)
	args = parser.parse_args()
	main(args)
