#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:05:05 2021

@author: laurent
"""

import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision

def getSets(filteredClass = None, removeFiltered = True) :
	"""
	Return a torch dataset
	"""
	
	train = torchvision.datasets.MNIST('./data/', train=True, download=True,
								transform=torchvision.transforms.Compose([
										torchvision.transforms.ToTensor(),
										torchvision.transforms.Normalize((0.1307,), (0.3081,))
								]))

	test = torchvision.datasets.MNIST('./data/', train=False, download=True,
								transform=torchvision.transforms.Compose([
										torchvision.transforms.ToTensor(),
										torchvision.transforms.Normalize((0.1307,), (0.3081,))
								]))
	
	if filteredClass is not None :
		
		train_loader = torch.utils.data.DataLoader(train, batch_size=len(train))
	
		train_labels = next(iter(train_loader))[1].squeeze()
		
		test_loader = torch.utils.data.DataLoader(test, batch_size=len(test))
	
		test_labels = next(iter(test_loader))[1].squeeze()
		
		if removeFiltered : 
			trainIndices = torch.nonzero(train_labels != filteredClass).squeeze()
			testIndices = torch.nonzero(test_labels != filteredClass).squeeze()
		else :
			trainIndices = torch.nonzero(train_labels == filteredClass).squeeze()
			testIndices = torch.nonzero(test_labels == filteredClass).squeeze()
		
		train = torch.utils.data.Subset(train, trainIndices)
		test = torch.utils.data.Subset(test, testIndices)
	
	return train, test

if __name__ == "__main__" :
	
	#test getSets function
	train, test = getSets(filteredClass = 3, removeFiltered = False)
	
	test_loader = torch.utils.data.DataLoader(test, batch_size=len(test))
	
	images, labels = next(iter(test_loader))
	
	print(images.shape)
	print(torch.unique(labels.squeeze()))