import numpy as np
import pickle

def retrieve(name):
	filename = 'info/'+name+'.pkl'
	file = open(filename,"rb")
	customer_list = pickle.load(file)
	return customer_list
