import lib.match_templates as matching
import lib.wave as wv
import lib.push_notif as notify
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from jinja2 import FileSystemLoader, Environment
from scipy.io.wavfile import read, write
from sklearn.metrics import classification_report

template_loader = FileSystemLoader(searchpath="./template/")
template_env = Environment(loader=template_loader)
tm = template_env.get_template("log_template")

CHORDS = ['C','D','E','F','G','A','B','Cm','Dm','Em','Fm','Gm','Am','Bm']
NOISY_PATH = "./chords/_past/noisy/"
DESIRED_PATH = "./chords/_past/desired/"
# NOISE_PATH = "./chords/_past/noise_sample/"

NOISY = {}
DESIRED = {}
NOISE = {}

# Load chord data
for chord in CHORDS:
	sr, DESIRED[chord] = read(DESIRED_PATH + "desired_" + chord + ".wav")
	sr, NOISY[chord] = read(NOISY_PATH + "akord_" + chord + ".wav") # read(NOISY_PATH + "akord_" + chord + ".wav")
	NOISY[chord] = (NOISY[chord][:,1] + NOISY[chord][:,0])/2

filters = list(range(50, 300, 50))
delta = .1

acc_csv = open("./accuracy.csv", "w+")
for n_w in filters:
	bundle = []
	result_list = []
	log = open(f"./log_{n_w}.txt", "w+") #open("./documents/logs/log"+str(n_w)+".txt", "w+")
	sq_error = None

	for chord in CHORDS:
		# reading signal
		d = DESIRED[chord]
		x = NOISY[chord]
		k = x.shape[0]
		L = k/(k+1)

		# normalization to [-1,1]
		x = wv.change_range(x, -1, 1)
		d = wv.change_range(d, -1, 1)

		print(f"==================\nProcessing {chord} with filter length {n_w}")
		# RLS noise reduction
		rls = wv.RecursiveLeastSquare(x, d)
		rls.set_params(n_w, delta, L)
		print("Reducing...")
		e = rls.reduce_noise()

		# ensemble square error
		sq_error = np.copy(rls.sq_error()) if chord == "C" else np.vstack((sq_error, rls.sq_error()))

		# chord recognition
		cr = wv.change_range(e, -1, 1)
		epcp = matching.EPCPChordRecognition(cr, sr)
		print("Classifying...")
		result, chord_list, chroma = epcp.predict(threshold=0.5) # classify the chord

		# write chroma and distance in CSV
		epcp.write_to_csv(data=epcp.chroma_div.tolist(), csv_filename=f"./chroma_{n_w}.csv", line_limit=len(CHORDS)) # "./documents/chroma/chroma_"+str(n_w)+".csv"
		epcp.write_to_csv(data=epcp.dist_matrix[tuple(chroma)], csv_filename=f"./dist_{n_w}.csv", line_limit=len(CHORDS))
		
		# data preparation for log.txt
		result_list.append(result)
		bundle.append({'chord':chord, 'result':result, 'chord_list':chord_list, 'chroma': chroma})

	# Evaluation
	report = classification_report(CHORDS, result_list, labels=CHORDS, output_dict=True, zero_division=0)
	accuracy = len([result_list[i] for i in range(len(CHORDS)) if result_list[i] == CHORDS[i]])*100/len(CHORDS)
	additional_metrics = report['macro avg']

	# # MSE
	# mse = np.array([np.mean(sq_error[:, i]) for i in range(0, k)])
	# fig, ax = plt.subplots(1, 1, figsize=(12, 5))
	# ax.plot(mse, linewidth=1)
	# # fig.savefig(f"X:/Finale/assets/mse/{n_w}.png")
	# # fig.clf()
	# plt.show()

	# create log based on template
	acc_csv.write(f"{n_w},{accuracy}")
	msg = tm.render(bundle=bundle, result_list=result_list, accuracy=accuracy, metrics=additional_metrics)

	log.write(msg) # write template
	log.close()

	# notify.pushbullet_notif("RLS is complete", f"Filter size: {n_w}; Accuracy: {accuracy}%")

acc_csv.close()
	
