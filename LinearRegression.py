import pandas as pd
import json
import numpy as np
import sys
from sklearn.model_selection import train_test_split
import random

class SimpleLinearRegression:
	def __init__(self):
		#initializes variables
		self.slope_ = None
		self.intercept_ = None
		
	def fit(self, x, y):
		#calculates the mean of the input and labels
		Xmean = np.mean(x)
		ymean = np.mean(y)
		
		#calculate terms needed for slope and intercept of regression line
		numerator = np.sum((x - Xmean) * (y - ymean))
		denominator = np.sum((x - Xmean) ** 2)

		#calculate slope and intercept of regression line
		self.slope_ = numerator / denominator
		self.intercept_ = ymean - self.slope_ * Xmean
		
	def predict(self, x):
		return self.intercept_ + self.slope_ * x