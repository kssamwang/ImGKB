import argparse
import time
import numpy as np
import torch.nn as nn
import torch
import random
from torch import optim
from src.model import KGIB
from src.utils import load_data, generate_batches,  AverageMeter, compute_metrics
from sklearn.model_selection import StratifiedShuffleSplit
from src.loss import Loss
import csv
import warnings
warnings.filterwarnings("ignore")
def seed_torch(seed=23):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

	torch.backends.cudnn.deterministic=True


def train(adj, features, graph_indicator, y, model,optimizer, criterion, beta):
	optimizer.zero_grad()
	output, loss_mi = model(adj, features, graph_indicator)
	loss_train =  criterion(output, y)
	loss_train = loss_train + beta * loss_mi
	loss_train.backward()
	optimizer.step()
	return output, loss_train

def val(adj, features, graph_indicator ,model):
	output, loss_mi = model(adj, features, graph_indicator)
	return output
def test(adj, features, graph_indicator ,model):
	output, loss_mi = model(adj, features, graph_indicator)
	return output


def main(args):

	print("----------------------------------------------starting training------------------------------------------------")

	model = KGIB(features_dim, args.hidden_dim, args.hidden_graphs, args.size_hidden_graphs,  args.nclass, args.max_step, args.num_layers, device).to(device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	criterion = Loss()
	best_acc = 0
	id_auc = []
	for epoch in range(args.epochs):
		start = time.time()
		model.train()
		train_loss = AverageMeter()
		train_auc = AverageMeter()


		# Train for one epoch
		for i in range(n_train_batches):
			output, loss = train(adj_train[i], features_train[i], graph_indicator_train[i], y_train[i], model, optimizer, criterion, args.beta)
			train_loss.update(loss.item(), output.size(0))
			output = nn.functional.softmax(output, dim=1)
			auc, _, _ =compute_metrics(output.data, y_train[i].data)
			train_auc.update(auc,output.size(0))

		# Evaluate on val set
		model.eval()
		val_auc = AverageMeter()
		val_recall = AverageMeter()
		val_f1score = AverageMeter()
		for i in range(n_val_batches):
			output = val(adj_val[i], features_val[i], graph_indicator_val[i], model)
			output = nn.functional.softmax(output, dim=1)
			acc1, recal, f1 = compute_metrics(output.data, y_val[i].data)
			val_auc.update(acc1, output.size(0))
			val_recall.update(recal, output.size(0))
			val_f1score.update(f1, output.size(0))

		# Evaluate on validation set
		# model.eval()
		val_auc1 = AverageMeter()
		val_recall1 = AverageMeter()
		val_f1score1 = AverageMeter()

		for i in range(n_test_batches):
			output = val(adj_test[i], features_test[i], graph_indicator_test[i], model)
			output = nn.functional.softmax(output, dim=1)
			acc1, recal, f1 = compute_metrics(output.data, y_test[i].data)
			val_auc1.update(acc1, output.size(0))
			val_recall1.update(recal, output.size(0))
			val_f1score1.update(f1, output.size(0))

		# Print results
		print('kfold:', '%d' % (Fold_idx),"epoch:", '%03d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss.avg), "train_auc=", "{:.5f}".format(train_auc.avg),
			  "val_auc=", "{:.5f}".format(val_auc.avg), "val_recal=", "{:.5f}".format(val_recall.avg), "val_f1score=", "{:.5f}".format(val_f1score.avg),
			  "test_auc=", "{:.5f}".format(val_auc1.avg), "test_recal=", "{:.5f}".format(val_recall1.avg), "test_f1score=", "{:.5f}".format(val_f1score1.avg),
			  "time=", "{:.5f}".format(time.time() - start))


		with  open('./Results/train.csv', 'a+', newline='', encoding='utf-8') as csvFile:
			writer = csv.writer(csvFile)
			# 先写columns_name
			writer.writerow((args.num_layers, args.max_step, args.hidden_graphs, args.size_hidden_graphs, args.hidden_dim, args.beta, Fold_idx,  epoch + 1,
							 train_loss.avg, train_auc.avg, val_auc.avg.item(), val_recall.avg,
							 val_f1score.avg, val_auc1.avg.item(), val_recall1.avg, val_f1score1.avg, time.time() - start))

		# Remember best auc
		is_best = val_auc.avg > best_acc
		best_acc = max(val_auc.avg, best_acc)
		if is_best:
			best_epoch_auc = epoch + 1
			id_auc.append(best_epoch_auc)
			if len(id_auc) > 10:
				id_auc = id_auc[1:]
			torch.save({
				'epoch': best_epoch_auc,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict(),
			}, 'save_model/model_best{:d}.pth.tar'.format(best_epoch_auc))


	print("finished!", "val_auc=", "{:.5f}".format(best_acc), "best epoch=", "%d" % (best_epoch_auc))

	# Testing auc
	performance1_auc = []
	performance1_recall = []
	performance1_f1 = []
	for best_epoch in id_auc:
		test_auc = AverageMeter()
		test_recall = AverageMeter()
		test_f1score = AverageMeter()
		checkpoint = torch.load('save_model/model_best{:d}.pth.tar'.format(best_epoch))
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		for i in range(n_test_batches):
			output = test(adj_test[i], features_test[i], graph_indicator_test[i], model)
			output = nn.functional.softmax(output, dim=1)
			auc, recal, f1 = compute_metrics(output.data, y_test[i].data)
			test_auc.update(auc, output.size(0))
			test_recall.update(recal, output.size(0))
			test_f1score.update(f1, output.size(0))
		performance1_auc.append(test_auc.avg)
		performance1_recall.append(test_recall.avg)
		performance1_f1.append(test_f1score.avg)
	idx = np.argmax(performance1_auc)
	best_auc = performance1_auc[idx]
	best_recal = performance1_recall[idx]
	best_f1 = performance1_f1[idx]
	print("AUC Loading checkpoint!", "test_auc=", "{:.5f}".format(best_auc), "test_recall=",
		  "{:.5f}".format(best_recal), "test_f1score=", "{:.5f}".format(best_f1))

	with  open('./Results/test.csv', 'a+', newline='', encoding='utf-8') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerow((args.num_layers, args.max_step, args.hidden_graphs, args.size_hidden_graphs, args.hidden_dim,
						 args.beta, Fold_idx, best_auc, best_recal, best_f1))

	L1auc.append(best_auc)
	L1recall.append(best_recal)
	L1f1score.append(best_f1)


if __name__ == "__main__":
	# Argument parser
	parser = argparse.ArgumentParser(description='ImGKB')
	parser.add_argument('--dataset', default='MCF-7', help='Dataset name')
	parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='Initial learning rate')
	parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='Input batch size for training')
	parser.add_argument('--epochs', type=int, default=2, metavar='N', help='Number of epochs to train')
	parser.add_argument('--hidden-graphs', type=int, default=4, metavar='N', help='Number of hidden graphs')
	parser.add_argument('--size-hidden-graphs', type=int, default=6, metavar='N', help='Number of nodes of each hidden graph')
	parser.add_argument('--hidden-dim', type=int, default=72, metavar='N', help='Size of hidden layer of NN')
	parser.add_argument('--feature-dim', type=int, default=10, metavar='N', help='Input size')
	parser.add_argument('--num-layers', type=int, default=2, metavar='N', help='Number of layer of KerGAD')
	parser.add_argument('--penultimate-dim', type=int, default=32, metavar='N', help='Size of penultimate layer of NN')
	parser.add_argument('--max-step', type=int, default=3, metavar='N', help='Max length of walks')
	parser.add_argument('--nclass', type=int, default=2, metavar='N', help='Class number')
	parser.add_argument('--beta', type=float, default=0.3, metavar='beta', help='Compression coefficient')
	parser.add_argument('--normalize', action='store_true', default=True, help='Whether to normalize the kernel values')
	parser.add_argument('--graph_pooling_type', type=str, default='average', choices=["sum", "average"], help='the type of graph pooling (sum/average)')
	parser.add_argument('--n_split', type=int, default=10, help='cross validation')
	parser.add_argument('--seed', type=int, default=23, help='random seed')
	args = parser.parse_args()
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	adj_lst, features_lst, graph_labels = load_data(args.dataset)
	N = len(adj_lst)
	features_dim = features_lst[0].shape[1]
	n_classes = np.unique(graph_labels).size
	args.feature_dim = features_dim
	args.nclass = n_classes

	seed_torch(args.seed)
	skf = StratifiedShuffleSplit(args.n_split, test_size=0.1, train_size=0.9, random_state=args.seed)
	L1auc = list()
	L1recall = list()
	L1f1score = list()

	Fold_idx = 1
	for train_index, test_index in skf.split(np.zeros(len(adj_lst)), graph_labels):

		idx = np.random.permutation(train_index)
		train_index = idx[:int(idx.size * 0.9)].tolist()
		val_index = idx[int(idx.size * 0.9):].tolist()

		adj_train = [adj_lst[i] for i in train_index]
		feats_train = [features_lst[i] for i in train_index]
		label_train = [graph_labels[i] for i in train_index]


		adj_val = [adj_lst[i] for i in val_index]
		feats_val = [features_lst[i] for i in val_index]
		label_val = [graph_labels[i] for i in val_index]

		adj_test = [adj_lst[i] for i in test_index]
		feats_test = [features_lst[i] for i in test_index]
		label_test = [graph_labels[i] for i in test_index]

		adj_train, features_train, graph_pool_lst1, graph_indicator_train, y_train, n_train_batches = generate_batches(
			adj_train, feats_train, label_train, args.batch_size, args.graph_pooling_type, device)

		adj_val, features_val, _, graph_indicator_val, y_val, n_val_batches = generate_batches(
			adj_val, feats_val, label_val, args.batch_size, args.graph_pooling_type, device)

		adj_test, features_test, graph_pool_lst2, graph_indicator_test, y_test, n_test_batches = generate_batches(
			adj_test, feats_test, label_test, args.batch_size, args.graph_pooling_type, device)

		main(args)
		Fold_idx += 1
	print('Optimization finished!', "avg_test_auc=", "{:.5f}".format(np.mean(L1auc)),
		  "avg_test_recall=", "{:.5f}".format(np.mean(L1recall)), "avg_test_f1score=",
		  "{:.5f}".format(np.mean(L1f1score)))


	with  open('./Results/final.csv', 'a+', newline='', encoding='utf-8') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerow((args.num_layers, args.max_step, args.hidden_graphs,
						 args.size_hidden_graphs, args.hidden_dim,
						 args.beta, np.mean(L1auc), np.mean(L1recall),
						 np.mean(L1f1score)))

