import os
import copy
import pickle
import shutil

import argparse
import numpy as np
import tqdm
import sys
import matplotlib.pyplot as plt

from torchinfo import summary
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

from dataloader import load_pseudo_label_data, load_data, load_data_meta
from clip_custom import clip
from model_multi import Custom_Clip, PromptGenerator
from utils import disable_running_stats, enable_running_stats
from dataset import MultiSourceDataset, SingleSourceDataset
from samplers import RandomDomainSampler
import ot

def arg_parse():
	parser = argparse.ArgumentParser("Training and Evaluation Script", add_help=False)

	# for config
	parser.add_argument(
		"--data_root",
		type=str,
		default=r"/vast/hvp2011/data/office-31/",
		help="data file path",
	)
	parser.add_argument("--backbone", type=str, default="RN101", help="")
	parser.add_argument("--dataset", type=str, default="ImageCLEF", help="")
	parser.add_argument("--target", type=str, default="Art", help="")
	parser.add_argument("--seed", type=int, default=1, help="")

	# for dataloader
	parser.add_argument("--batch_size", type=int, default=32, help="")
	parser.add_argument("--num_workers", type=int, default=4, help="")
	parser.add_argument("--pin_memory", type=bool, default=True, help="")
	parser.add_argument(
		"--threshold",
		type=float,
		default=0.4,
		help="threshold tau for generating pseudo labels",
	)

	# for prompt settings
	parser.add_argument(
		"--M1", type=int, default=16, help="number of classification tokens"
	)
	parser.add_argument("--M2", type=int, default=16, help="number of domain tokens")

	# for training settings
	parser.add_argument("--prompt_iteration", type=int, default=5000, help="")
	parser.add_argument("--prompt_learning_rate", type=float, default=0.003, help="")
	parser.add_argument("--ot_t_weight", type=float, default=0.5, help="")
	parser.add_argument("--t_weight", type=float, default=0.5, help="")
	parser.add_argument("--output_dir", type=str, default="outputs_multi_baseline", help="")
	parser.add_argument("--w_scale", type=float, default=0.05, help="")
	parser.add_argument("--distance", type=str, default='l2', help="")

	


	parser.add_argument("--self_correct", type=int, default=1, help="")
	parser.add_argument("--inforce_inv", type=int, default=1, help="")
	parser.add_argument("--evaluation_step", type=int, default=50, help="")

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
		
def save_plot(target_name, target_test_loader, custom_clip_model, args):
	scale = custom_clip_model.logit_scale.exp()

	correct = 0
	tot = 0
	feature_train, Y_train, Y_domain_train = [], [], []
	with torch.no_grad():
		for data, label in target_test_loader:
			tot += args.batch_size
			data = data.to(args.device)
			label = label.to(args.device)

			img_feature = custom_clip_model.forward_img(data)
					
			feature_train += img_feature.tolist()
			Y_train += label.tolist()
			print("Train dumped")
		
		fol_name = '/home/long/LA_UDA/LA/plots'
		with open(os.path.join(fol_name, "{}_feature_train.pkl".format(target_name)), "wb") as fp:
			pickle.dump(feature_train, fp)
		with open(os.path.join(fol_name, "{}_Y_train.pkl".format(target_name)), "wb") as fp:
			pickle.dump(Y_train, fp)


def test_w(target_test_loader, custom_clip_model, weight, tokenized_prompts, args):
	scale = custom_clip_model.logit_scale.exp()

	correct = 0
	tot = 0
	with torch.no_grad():
		for data, label in target_test_loader:
			tot += args.batch_size
			data = data.to(args.device)
			label = label.to(args.device)

			tot_logits = 0

			# TODO: test on multiple prompts
			
			img_feature = custom_clip_model.forward_img(data)
			logits = img_feature @ weight.t()
			tot_logits += logits
			output = torch.argmax(tot_logits, dim=1) # % n_cls

			correct += (output == label).sum().item()

		# print("accuracy is: {} with a total of {} data".format(correct / tot, tot))

	return correct / tot

def test(target_test_loader, custom_clip_model, prompt_list, tokenized_prompts, args, running_means=None):
	scale = custom_clip_model.logit_scale.exp()

	correct = 0
	tot = 0
	with torch.no_grad():
		for data, label in target_test_loader:
			tot += args.batch_size
			data = data.to(args.device)
			label = label.to(args.device)

			tot_logits = 0

			
			img_feature, features = custom_clip_model.forward_img_both(data)

			if running_means is not None:
				weights = torch.zeros(running_means.shape[0], features.shape[0], running_means.shape[1]).to(args.device)

				for source_index in range(running_means.shape[0]):
					# print(features.unsqueeze(1).shape, running_means[source_index].unsqueeze(0).shape)
					weights[source_index] = torch.norm((features.unsqueeze(1) - running_means[source_index].unsqueeze(0)), p=2, dim=2)
				weights = nn.Softmax(dim=0)(-weights*args.w_scale)
				# weights /= weights.sum(0, keepdim=True)
				weights = weights.detach()

			# TODO: test on multiple prompts
			for source_index, prompt in enumerate(prompt_list):
				txt_feature = custom_clip_model.forward_txt(prompt, tokenized_prompts)

				logits = img_feature @ txt_feature.t()
				if running_means is not None and source_index < running_means.shape[0]:
					tot_logits += weights[source_index] * logits
				else:
					tot_logits += logits

				

			tot_logits /= len(prompt_list)
			output = torch.argmax(tot_logits, dim=1)

			correct += (output == label).sum().item()

		# print("accuracy is: {} with a total of {} data".format(correct / tot, tot))

	return correct / tot


def make_meta_test(loader, custom_clip_model, num_iter=20):
	datas = []
	labels = []
	with torch.no_grad():
		step = 0
		for data, label in loader:
			step += 1
			if step > num_iter:
				break
			data = data.to(args.device)
			label = label.to(args.device)
			img_feature, features = custom_clip_model.forward_img_both(data)
			datas.append(img_feature.clone().unsqueeze(0))
			labels.append(label.clone().unsqueeze(0))
	return [torch.cat(datas, dim=0), torch.cat(labels, dim=0)]


def test_meta(datas, labels, prompt):

	correct = 0
	tot = datas.shape[0]*datas.shape[1]
	# print(datas.shape)
	# print(labels.shape)
	with torch.no_grad():
		for idx in range(len(datas)):
			logits = datas[idx] @ prompt.t()
			output = torch.argmax(logits, dim=1)
			correct += (output == labels[idx]).sum().item()
	return correct / tot

def soft_cross_entropy_loss(predictions, soft_targets):

	# Apply softmax to get predicted probabilities
	log_probs = F.log_softmax(predictions, dim=1)
	
	# Compute cross-entropy loss using soft labels
	loss = -torch.sum(soft_targets * log_probs, dim=1).mean()
	return loss
	
def _cost_matrix(x, y, p=1):
	"Returns the matrix of $|x_i-y_j|^p$."
	x_col = x.unsqueeze(-2)
	y_lin = y.unsqueeze(-3)
	C = 2-torch.sum((torch.abs(x_col - y_lin)), -1)
	return C


class MLP(nn.Module):
	"""Just  an MLP"""

	def __init__(self, n_inputs, n_outputs):
		super(MLP, self).__init__()
		hparams = {}
		hparams["mlp_width"] = 256
		hparams["mlp_depth"] = 3
		hparams["mlp_dropout"] = 0.0
		self.input = nn.Linear(n_inputs, hparams["mlp_width"])
		self.dropout = nn.Dropout(hparams["mlp_dropout"])
		self.hiddens = nn.ModuleList(
			[
				nn.Linear(hparams["mlp_width"], hparams["mlp_width"])
				for _ in range(hparams["mlp_depth"] - 2)
			]
		)
		self.output = nn.Linear(hparams["mlp_width"], n_outputs)
		self.n_outputs = n_outputs

	def forward(self, x):
		x = self.input(x)
		x = self.dropout(x)
		x = F.relu(x)
		for hidden in self.hiddens:
			x = hidden(x)
			x = self.dropout(x)
			x = F.relu(x)
		x = self.output(x)
		return x


def calc_distance(A, B):
    distances = (torch.sum(A**2, dim=1, keepdim=True) 
                + torch.sum(B**2, dim=1)
                - 2 * torch.matmul(A, B.t()))

    return distances

class SoftCrossEntropyLoss():
	def __init__(self, weights):
		super().__init__()
		self.weights = weights

	def forward(self, y_hat, y):
		p = F.log_softmax(y_hat, 1)
		w_labels = self.weights*y
		loss = -(w_labels*p).sum() / (w_labels).sum()
		return loss

def train(domain_list, target_domain, classnames, clip_model, preprocess, args):
	custom_clip_model = Custom_Clip(clip_model)
	# custom_clip_model = nn.DataParallel(custom_clip_model)
	# custom_clip_model = custom_clip_model.module
	
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

		if target_domain != target_name and args.dataset == "DomainNet":
			continue
		
		print("*" * 50)
		print("Start training on {}".format(target_name))

		if target_domain == target_name:
			tgt_save_path = os.path.join(args.output_dir, target_name)
			os.makedirs(tgt_save_path, exist_ok=True)
			result_path = os.path.join(tgt_save_path, "best_accuracy.txt")
			if os.path.exists(result_path):
				continue
			orig_stdout = sys.stdout
			f = open(tgt_save_path+ "/train.log", "w+")
			sys.stdout = f

		source_name_list = domain_list.copy()
		source_name_list.remove(target_name)
		
		target_path = os.path.join(args.data_root, target_name)

		if args.dataset == "DomainNet":
			target_train_dataset = SingleSourceDataset(
				args.data_root, [target_name], preprocess
			)
			target_train_loader = torch.utils.data.DataLoader(
				target_train_dataset,
				batch_size=args.batch_size,
				num_workers=4,
				pin_memory=True,
			)	
		else:
			target_train_loader = load_pseudo_label_data(
				target_name, target_path, preprocess, clip_model, args
			)

		target_test_loader = load_data(target_path, preprocess, args)
		if target_domain == target_name:
			meta_test = []
			meta_loaders = []
			for ii in source_name_list:
				aa = load_data(os.path.join(args.data_root, ii), preprocess, args)
				meta_test.append(make_meta_test(
								aa, 
								custom_clip_model, num_iter=100))
			
		source_train_dataset = MultiSourceDataset(
			args.data_root, source_name_list, preprocess
		)
		sampler = RandomDomainSampler(
			source_train_dataset.data, args.batch_size*len(source_name_list), len(source_name_list)
		)
		source_train_loader = torch.utils.data.DataLoader(
			source_train_dataset,
			batch_size=args.batch_size*len(source_name_list),
			sampler=sampler,
			num_workers=4,
			pin_memory=True,
		)
		scale = custom_clip_model.logit_scale.exp()
		prompt_learner = PromptGenerator(
			classnames, clip_model, source_name_list, target_name, args
		)
		print("PromptGenerator", summary(prompt_learner))
		tokenized_prompts = prompt_learner.tokenized_prompts

		# save_plot(
		# 		target_name,
		# 		target_test_loader,
		# 		custom_clip_model,
		# 		args,
		# 	)
		# continue

		optimizer = torch.optim.AdamW(
			list(prompt_learner.parameters()), lr=args.prompt_learning_rate
		)

		scheduler = CosineAnnealingLR(optimizer, T_max=args.prompt_iteration)

		for name, param in prompt_learner.named_parameters():
			print(
				f"name: {name}, shape {param.shape}, require grad: {param.requires_grad}"
			)

		n_cls = len(classnames)
		class_list_tokenize = []
		for i in range(n_cls):
			class_list_tokenize.append(f"A photo of a {classnames[i]}")
		text = clip.tokenize(class_list_tokenize).to(args.device)

		n_domains = len(source_name_list)
		running_means = torch.zeros(n_domains+1, n_cls, feature_dim).to(args.device)
		running_count = torch.zeros(n_domains+1, n_cls).to(args.device)
		
		mean_target_txt_features = 0
		mean_count = 0

		best_acc = 0
		n_conflict = 0
		grad_cosine = []
		running_avg_cosine = 0
		pbar = tqdm.tqdm(range(1, args.prompt_iteration + 1))
		for step in pbar:
			_, target_prompts = prompt_learner()

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
			
			if target_domain != target_name:
				continue

			optimizer.zero_grad()

			total_loss = 0 

			# Prepare
			feature_list = []
			text_list = []
			
			for source_index in range(n_domains):
				data = source_data[source_domain==source_index]
				label = source_label[source_domain==source_index]
				source_prompt, target_prompts = prompt_learner(source_index=source_index)
				img_feature, txt_feature, features = custom_clip_model(
					data, source_prompt, tokenized_prompts
				)
				text_list.append(txt_feature)
				feature_list.append(img_feature)
				unique_cls = torch.unique(label)
				for d_cls in unique_cls:
					cls_features = features[label==d_cls]
					running_means[source_index][d_cls] = running_means[source_index][ d_cls] * running_count[source_index][ d_cls] + cls_features.sum(0)
					running_count[source_index][ d_cls] += cls_features.shape[0]
					running_means[source_index][ d_cls] /= running_count[source_index][ d_cls]
					
					running_means[n_domains][d_cls] = running_means[n_domains][ d_cls] * running_count[n_domains][ d_cls] + cls_features.sum(0)
					running_count[n_domains][ d_cls] += cls_features.shape[0]
					running_means[n_domains][ d_cls] /= running_count[n_domains][ d_cls]


			source_loss = 0
			n_source = 0

			invariant_prompts = prompt_learner.forward_invariant()
			invariant_txt = custom_clip_model.forward_txt(invariant_prompts, tokenized_prompts)

			# Forward
			for source_index in range(n_domains):
				data = source_data[source_domain==source_index]
				label = source_label[source_domain==source_index]
				source_img_features = feature_list[source_index]
				source_txt_features = text_list[source_index]
				source_logits = source_img_features @ source_txt_features.t()
				source_loss += F.cross_entropy(scale * source_logits, label)
				n_source += 1

				if args.inforce_inv == 1:
					source_logits = source_img_features @ invariant_txt.t()
					inv_loss = F.cross_entropy(scale * source_logits, label)
					total_loss += inv_loss
			total_loss += source_loss / n_source
			

			target_img_features, target_txt_features, features = custom_clip_model(
				target_data, target_prompts, tokenized_prompts
			)
			target_logits = target_img_features @ target_txt_features.t()

			# distance = torch.zeros(running_means.shape[0], target_img_features.shape[0], n_cls).to(args.device)
			# for source_index in range(running_means.shape[0]):
			# 	if args.distance == 'l2':
			# 		distance[source_index] = calc_distance(features, running_means[source_index])
			# 		# weights = (distance <= distance[-1,...].unsqueeze(0)).float()
			# 		# weights = weights / weights.sum(dim=0, keepdim=True)

			# 		weights = nn.Softmax(dim=0)(-distance*args.w_scale).detach()

				
			# 	else:
			# 		distance[source_index] = features @ running_means[source_index].t()
			# 		# weights = (distance >= distance[-1,...].unsqueeze(0)).float()
			# 		# weights = weights / weights.sum(dim=0, keepdim=True)
			# 		weights = nn.Softmax(dim=0)(distance*args.w_scale).detach()


			with torch.no_grad():
				if step % args.evaluation_step == 0:
					for source_index in range(n_domains):
						for source_index_i in range(n_domains):
							if source_index_i != source_index:
								acc = test_meta(
									meta_test[source_index_i][0],
									meta_test[source_index_i][1],
									text_list[source_index]
								)
								print('source: {} to source {} acc {}'.format(source_index, source_index_i, acc))

			
			# if step % 50 == 0:
			# 	print(distance[:,0,:].permute(1,0))
			
			# if step % 50 == 0:
			# 	print(weights.shape)
			# 	print(weights[:,0,:].permute(1,0))

				# print(running_means)
				
			# Pseudo-label
			logits = 0
			n_pseudo = 0
			# Based prompt from CLIP
			with torch.no_grad():
				image_features, text_features = clip_model(target_data, text)
				logits += image_features @ text_features.t()
				n_pseudo += 1
			
			# Source distillation
			for source_index in range(n_domains):
				source_txt_features = text_list[source_index]
				logits +=  target_img_features @ source_txt_features.t()
				# logits +=  weights[source_index] * (target_img_features @ source_txt_features.t())
			# logits +=  weights[n_domains] * (target_img_features @ invariant_txt.t())
			n_pseudo += 1
			logits /= n_pseudo
			
			probs = (scale *logits).softmax(dim=-1).detach()
			target_cls_loss = soft_cross_entropy_loss(scale * target_logits, probs)
			n_target = 1


			# Self-Correction
			if args.self_correct == 1:
				pseudo_label = target_logits.max(1)[1]
				target_cls_loss += F.cross_entropy(
					scale * target_logits, pseudo_label
				)
				n_target += 1
			target_loss = target_cls_loss / n_target
			total_loss += args.t_weight * target_loss


			target_metric = 1 - target_logits
			BTI = target_img_features.shape[0]
			BTT = target_txt_features.shape[0]
			sample_weight_i = torch.ones(BTI).to(target_img_features.device) / BTI
			sample_weight_t = torch.ones(BTT).to(target_img_features.device) / BTT
			t_ot_cost = ot.emd2(sample_weight_i, sample_weight_t, target_metric, numItermax=500000)
			total_loss += args.ot_t_weight * (t_ot_cost)

			optimizer.zero_grad()	
			total_loss.backward()
			optimizer.step()

			if step % (args.prompt_iteration / 20) == 0:
				scheduler.step()
			

			if args.dataset == "DomainNet":
				if step < 3000:
					if step % 500:
						continue
				else:
					if step% args.evaluation_step:
						continue

			if args.dataset == "DomainNet":
				if step > 3000:
					mean_target_txt_features += target_txt_features
					mean_count +=1
			else:
				if step > 100:
					mean_target_txt_features += target_txt_features
					mean_count +=1

			if (step) % args.evaluation_step:
				continue
			prompt_list = []
 
			for source_index in range(len(source_name_list)):
				source_prompt, _ = prompt_learner(source_index=source_index)
				prompt_list.append(source_prompt)
			prompt_list.append(invariant_prompts)
			acc = test(
					target_test_loader,
					custom_clip_model,
					[target_prompts],
					tokenized_prompts,
					args,
				)
			print('Target: ',acc)


			pbar.set_description(
				f"step: {step}, accuracy: {acc}, target total loss: {target_loss.item()}"
			)
			if acc > best_acc:
				best_acc = acc
			print(f"Best accuracy so far: {best_acc}, step {step}, accuracy {acc}")            
		
		if target_domain != target_name:
			continue
		acc = test_w(
			target_test_loader,
			custom_clip_model,
			mean_target_txt_features/mean_count,
			tokenized_prompts,
			args,
		)
		print('average_weight: ', acc)

		acc = test(
			target_test_loader,
			custom_clip_model,
			prompt_list,
			tokenized_prompts,
			args,
		)
		print('source combined: ', acc)

		acc = test(
			target_test_loader,
			custom_clip_model,
			prompt_list,
			tokenized_prompts,
			args,
			running_means
		)
		print('weight source combined: ', acc)
		
		prompt_list.append(target_prompts)
		acc = test(
			target_test_loader,
			custom_clip_model,
			prompt_list,
			tokenized_prompts,
			args,
			running_means
		)
		print('weight all: ', acc)

		best_accs.append(best_acc)
		print("Best accuracy for each domain:", best_accs, "Average:", np.mean(best_accs))
		print("Number of conflicts:", n_conflict, "Total steps:", args.prompt_iteration, "Conflict rate:", n_conflict/args.prompt_iteration)
		sys.stdout = orig_stdout
		f.close()

			

	
	
	
	
	


def main(args):
	args_update(args)
	print(args)
	args.device = "cuda" if torch.cuda.is_available() else "cpu"

	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	torch.backends.cudnn.deterministic = True

	model, preprocess = clip.load(args.backbone, device=args.device)
	model.float()
	args.data_root = args.data_root + args.dataset +'/'
	domain_list = os.listdir(args.data_root)

	domain_list = [x for x in domain_list if ".txt" not in x]

	classnames_path = os.path.join(args.data_root, domain_list[0])

	classnames = os.listdir(classnames_path)
	n_cls = len(classnames)
	classnames.sort()

	args.output_dir = 'outputs/'+ args.output_dir + str(args).replace(", ", "/").replace(
		"'", ""
	).replace("(", "").replace(")", "").replace("Namespace", "")


	print("Output directory:", args.output_dir)
	os.makedirs(args.output_dir, exist_ok=True)

	args.n_cls = n_cls
	train(domain_list, args.target, classnames, model, preprocess, args)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		"Training and Evaluation Script", parents=[arg_parse()]
	)
	args = parser.parse_args()
	main(args)
