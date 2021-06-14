#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 16:04:42 2021

@author: laurent
"""

from dataset import getSets
from viModel import BayesianMnistNet

import numpy as np

import matplotlib.pyplot as plt

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import os

import argparse as args

def saveModels(models, savedir) :
	
	for i, m in enumerate(models) :
		
		saveFileName = os.path.join(savedir, "model{}.pth".format(i))
		
		torch.save({"model_state_dict": m.state_dict()}, os.path.abspath(saveFileName))
	
def loadModels(savedir) :
	
	models = []
	
	for f in os.listdir(savedir) :
		
		model = BayesianMnistNet(p_mc_dropout=None)		
		model.load_state_dict(torch.load(os.path.abspath(os.path.join(savedir, f)))["model_state_dict"])
		models.append(model)
		
	return models

if __name__ == "__main__" :
	
	parser = args.ArgumentParser(description='Train a BNN on Mnist')
	
	parser.add_argument('--filteredclass', type=int, default = 5, choices = [x for x in range(10)], help="The class to ignore during training")
	parser.add_argument('--testclass', type=int, default = 4, choices = [x for x in range(10)], help="The class to test against that is not the filtered class")
	
	
	parser.add_argument('--savedir', default = None, help="Directory where the models can be saved or loaded from")
	parser.add_argument('--notrain', action = "store_true", help="Load the models directly instead of training")
	
	parser.add_argument('--nepochs', type=int, default = 10, help="The number of epochs to train for")
	parser.add_argument('--nbatch', type=int, default = 64, help="Batch size used for training")
	parser.add_argument('--nruntests', type=int, default = 50, help="The number of pass to use at test time for monte-carlo uncertainty estimation")
	parser.add_argument('--learningrate', type=float, default = 5e-3, help="The learning rate of the optimizer")
	parser.add_argument('--numnetworks', type=int, default = 10, help="The number of networks to train to make an ensemble")
	
	args = parser.parse_args()
	plt.rcParams["font.family"] = "serif"
	
	
	train, test = getSets(filteredClass = args.filteredclass)
	train_filtered, test_filtered = getSets(filteredClass = args.filteredclass, removeFiltered = False)
	
	N = len(train)
	
	train_loader = torch.utils.data.DataLoader(train, batch_size=args.nbatch)
	test_loader = torch.utils.data.DataLoader(test, batch_size=args.nbatch)
	
	batchLen = len(train_loader)
	digitsBatchLen = len(str(batchLen))
	
	models = []
	
	# Training or Loading
	if args.notrain :
		
		models = loadModels(args.savedir)
		
	else :
	
		for i in np.arange(args.numnetworks) :
			print("Training model {}/{}:".format(i+1, args.numnetworks))
			
			#Initialize the model
			model = BayesianMnistNet(p_mc_dropout=None) #p_mc_dropout=None will disable MC-Dropout for this bnn, as we found out it makes learning much much slower.
			loss = torch.nn.NLLLoss(reduction='mean') #negative log likelihood will be part of the ELBO
			
			optimizer = Adam(model.parameters(), lr=args.learningrate)
			optimizer.zero_grad()
			
			for n in np.arange(args.nepochs) :
				
				for batch_id, sampl in enumerate(train_loader) :
					
					images, labels = sampl
					
					pred = model(images, stochastic=True)
					
					logprob = loss(pred, labels)
					l = N*logprob
					
					modelloss = model.evalAllLosses()
					l += modelloss
					
					optimizer.zero_grad()
					l.backward()
					
					optimizer.step()
					
					print("\r", ("\tEpoch {}/{}: Train step {"+(":0{}d".format(digitsBatchLen))+"}/{} prob = {:.4f} model = {:.4f} loss = {:.4f}          ").format(
																									n+1, args.nepochs,
																									batch_id+1,
																									batchLen,
																									torch.exp(-logprob.detach().cpu()).item(),
																									modelloss.detach().cpu().item(),
																									l.detach().cpu().item()), end="")
			print("")
			
			models.append(model)
	
	if args.savedir is not None :
		saveModels(models, args.savedir)
	
	
	# Testing
	if args.testclass != args.filteredclass :
		
		train_filtered_seen, test_filtered_seen = getSets(filteredClass = args.testclass, removeFiltered = False)
	
		print("")
		print("Testing against seen class:")
		
		with torch.no_grad() :
		
			samples = torch.zeros((args.nruntests, len(test_filtered_seen), 10))
			
			test_loader = DataLoader(test_filtered_seen, batch_size=len(test_filtered_seen))
			images, labels = next(iter(test_loader))
			
			for i in np.arange(args.nruntests) :
				print("\r", "\tTest run {}/{}".format(i+1, args.nruntests), end="")
				model = np.random.randint(args.numnetworks)
				model = models[model]
				
				samples[i,:,:] = torch.exp(model(images))
		
			print("")
			
			withinSampleMean = torch.mean(samples, dim=0)
			samplesMean = torch.mean(samples, dim=(0,1))
			
			withinSampleStd = torch.sqrt(torch.mean(torch.var(samples, dim=0), dim=0))
			acrossSamplesStd = torch.std(withinSampleMean, dim=0)
			
			print("")
			print("Class prediction analysis:")
			print("\tMean class probabilities:")
			print(samplesMean)
			print("\tPrediction standard deviation per sample:")
			print(withinSampleStd)
			print("\tPrediction standard deviation across samples:")
			print(acrossSamplesStd)
		
			plt.figure("Seen class probabilities")
			plt.bar(np.arange(10), samplesMean.numpy())
			plt.xlabel('digits')
			plt.ylabel('digit prob')	
			plt.ylim([0,1])
			plt.xticks(np.arange(10))
			
			plt.figure("Seen inner and outter sample std")
			plt.bar(np.arange(10)-0.2, withinSampleStd.numpy(), width = 0.4, label="Within sample")
			plt.bar(np.arange(10)+0.2, acrossSamplesStd.numpy(), width = 0.4, label="Across samples")
			plt.legend()
			plt.xlabel('digits')
			plt.ylabel('std digit prob')
			plt.xticks(np.arange(10))
	
	
	
	
	
	print("")
	print("Testing against unseen class:")
	
	with torch.no_grad() :
	
		samples = torch.zeros((args.nruntests, len(test_filtered), 10))
		
		test_loader = DataLoader(test_filtered, batch_size=len(test_filtered))
		images, labels = next(iter(test_loader))
		
		for i in np.arange(args.nruntests) :
			print("\r", "\tTest run {}/{}".format(i+1, args.nruntests), end="")
			model = np.random.randint(args.numnetworks)
			model = models[model]
			
			samples[i,:,:] = torch.exp(model(images))
	
		print("")
		
		withinSampleMean = torch.mean(samples, dim=0)
		samplesMean = torch.mean(samples, dim=(0,1))
		
		withinSampleStd = torch.sqrt(torch.mean(torch.var(samples, dim=0), dim=0))
		acrossSamplesStd = torch.std(withinSampleMean, dim=0)
		
		print("")
		print("Class prediction analysis:")
		print("\tMean class probabilities:")
		print(samplesMean)
		print("\tPrediction standard deviation per sample:")
		print(withinSampleStd)
		print("\tPrediction standard deviation across samples:")
		print(acrossSamplesStd)
		
		plt.figure("Unseen class probabilities")
		plt.bar(np.arange(10), samplesMean.numpy())
		plt.xlabel('digits')
		plt.ylabel('digit prob')
		plt.ylim([0,1])
		plt.xticks(np.arange(10))
		
		plt.figure("Unseen inner and outter sample std")
		plt.bar(np.arange(10)-0.2, withinSampleStd.numpy(), width = 0.4, label="Within sample")
		plt.bar(np.arange(10)+0.2, acrossSamplesStd.numpy(), width = 0.4, label="Across samples")
		plt.legend()
		plt.xlabel('digits')
		plt.ylabel('std digit prob')
		plt.xticks(np.arange(10))
	
	
	
	
	
	print("")
	print("Testing against pure white noise:")
	
	with torch.no_grad() :
	
		l = 1000
		
		samples = torch.zeros((args.nruntests, l, 10))
		
		random = torch.rand((l,1,28,28))
		
		for i in np.arange(args.nruntests) :
			print("\r", "\tTest run {}/{}".format(i+1, args.nruntests), end="")
			model = np.random.randint(args.numnetworks)
			model = models[model]
			
			samples[i,:,:] = torch.exp(model(random))
	
		print("")
		
		withinSampleMean = torch.mean(samples, dim=0)
		samplesMean = torch.mean(samples, dim=(0,1))
		
		withinSampleStd = torch.sqrt(torch.mean(torch.var(samples, dim=0), dim=0))
		acrossSamplesStd = torch.std(withinSampleMean, dim=0)
		
		print("")
		print("Class prediction analysis:")
		print("\tMean class probabilities:")
		print(samplesMean)
		print("\tPrediction standard deviation per sample:")
		print(withinSampleStd)
		print("\tPrediction standard deviation across samples:")
		print(acrossSamplesStd)
		
		plt.figure("White noise class probabilities")
		plt.bar(np.arange(10), samplesMean.numpy())
		plt.xlabel('digits')
		plt.ylabel('digit prob')
		plt.ylim([0,1])
		plt.xticks(np.arange(10))
		
		plt.figure("White noise inner and outter sample std")
		plt.bar(np.arange(10)-0.2, withinSampleStd.numpy(), width = 0.4, label="Within sample")
		plt.bar(np.arange(10)+0.2, acrossSamplesStd.numpy(), width = 0.4, label="Across samples")
		plt.legend()
		plt.xlabel('digits')
		plt.ylabel('std digit prob')
		plt.xticks(np.arange(10))
		
	plt.show()