"""
Algorithm based on the paper 'Automatic Chord Recognition from
Audio Using Enhanced Pitch Class Profile' by Kyogu Lee
This script computes 12 dimensional chromagram for chord detection
@author ORCHISAMA
"""

from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from .chromagram import compute_chroma
from tqdm import tqdm
from collections import Counter


class EPCPChordRecognition:

	def __init__(self, x, fs):
		self.chords = ['N','C','D','E','F','G','A','B','Cm','Dm','Em','Fm','Gm','Am','Bm']
		self.templates = [[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], # C
						[0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0], # D
						[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1], # E
						[1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0], # F
						[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1], # G
						[0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], # A
						[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1], # B
						[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], # Cm
						[0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], # Dm
						[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], # Em
						[1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0], # Fm
						[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0], # Gm
						[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], # Am
						[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]] # Bm
		self.chroma_div = None
		self.modus = None

		self.x = x
		self.fs = fs

		#framing audio, window length = 8192, hop size = 1024 and computing PCP
		self.nfft = 8192
		self.hop_size = 1024
		self.nFrames = int(np.round(len(self.x)/(self.nfft-self.hop_size)))

		self.chord_list = [None]*self.nFrames
		self.chord_corr_dict = {}
		self.corr_chroma_dict = {}
		self.dist_matrix = {} # distance of chords of each frame to the template
		self.selection_list = ['mode', 'max_corr']

	def write_to_csv(self, data, csv_filename="chroma.csv", line_limit=0):
		import csv
		
		try:
			lines = len(open(csv_filename).readlines())
		except FileNotFoundError:
			lines = 0

		if lines >= line_limit and line_limit > 0:
			# delete whole file
			open(csv_filename, 'r+').truncate(0)

		chroma_csv = open(csv_filename, "a+", newline="\n")		
		with chroma_csv as file:
			writer = csv.writer(file, delimiter=",")
			writer.writerows([data])
		chroma_csv.close()

	def predict(self, threshold=0.5, selection='mode'):

		if selection in self.selection_list:
			#zero padding to make signal length long enough to have nFrames
			x = np.append(self.x, np.zeros(self.nfft))
			xFrame = np.empty((self.nfft, self.nFrames))
			start = 0   
			chroma = np.empty((12,self.nFrames)) 
			id_chord = np.zeros(self.nFrames, dtype='int32')
			max_cor = np.zeros(self.nFrames)

			for n in tqdm(range(self.nFrames)):
				xFrame[:,n] = x[start:start+self.nfft] 
				start = start + self.nfft - self.hop_size 
				chroma[:,n] = compute_chroma(xFrame[:,n], self.fs)

				chroma[:,n] /= np.max(chroma[:,n])
		
				"""Correlate 12D chroma vector with each of 14 major and minor chords"""
				cor_vec = np.zeros(len(self.templates))
				for ni in range(len(self.templates)):
					# Calculate cosine of EPCP and templates vector to find the maximum batch
					dot_product = np.dot(chroma[:,n], np.array(self.templates[ni]))
					unit_vector = np.linalg.norm(chroma[:,n])*np.linalg.norm(self.templates[ni])
					cor_vec[ni] = dot_product/unit_vector
				
				max_cor[n] = np.max(cor_vec)
				self.corr_chroma_dict[max_cor[n]] = chroma[:,n]
				self.dist_matrix[tuple(chroma[:,n].tolist())] = cor_vec.tolist()
				id_chord[n] =  np.argmax(cor_vec) + 1
		
			#if max_cor[n] < threshold, then no chord is played
			#might need to change threshold value
			id_chord[np.where(max_cor < threshold*np.max(max_cor))] = 0

			for n in range(self.nFrames):
				tmp = self.chords[id_chord[n]]
				self.chord_list[n] = tmp
				self.chord_corr_dict[tmp] = max_cor[n]

			print(self.chord_list)

			if selection == 'mode':
				self.chord_list = list(filter(('N').__ne__, self.chord_list))
				self.modus = Counter(self.chord_list).most_common(1)[0][0]
			elif selection == 'max_corr':
				self.modus = max(self.chord_corr_dict, key=self.chord_corr_dict.get)
			
			corr_key = self.chord_corr_dict[self.modus]
			self.chroma_div = self.corr_chroma_dict[corr_key]
	
			return (self.modus, self.chord_list, self.chroma_div)
		
		else:
			raise Exception("Selection type unknown, use \'mode\' or \'correlation\'")
