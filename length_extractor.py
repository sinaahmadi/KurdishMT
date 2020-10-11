# -*- coding: utf-8 -*-
# Sina Ahmadi (ahmadi.sina@outlook.com)
# April 2020
# https://github.com/sinaahmadi/KurdishMT

import json
from nltk.tokenize import sent_tokenize

# Given the directory of the files, select lines (sentences) which contain up to the given length (=number of tokens)
# Used in Section 6.1 of the paper entitled "Towards Machine Translation for the Kurdish Language"
with open("/data_directories.json", "r") as fr:
	dirs = json.load(fr)

def extract_sents(text, n_lower, n_upper):
	temp = list()
	# for i in range(len(sent_tokenize(text))):
	text = text.split("\n")
	for i_r in range(len(text)):
		i = text[i_r]
		if len(i.split()) >= n_lower and len(i.split()) < n_upper:
			temp.append(i_r)

	return temp

def myround(x, base=5):
    return base * round(x/base)

def calcualte_length(text):
	dict_length = dict()
	for i in text.split("\n"):
		# i = i.split()
		if len(i) and len(i) in dict_length:
			dict_length[len(i)] += 1
		else:
			dict_length[len(i)] = 1

	return dict_length


# analyse token & character / line ratios
if False:
	ku = ["data/Ku/tanzil_ku_normalized.txt", "data/Ku/ted_sent_ku_normalized.txt", "data/Ku/wordnet_ku_normalized.txt"]
	en = ["data/En/tanzil_en_lower_tokenized.txt", "data/En/ted_en_lower_tokenized.txt", "data/En/wordnet_en_lower_tokenized.txt"]
	for file in ku+en:
		print(file)
		r_1, r_2 = list(), list()
		with open(file, "r") as f:
			text = f.read()
			d = calcualte_length(text)
			print(len(text.split()) / len(text.split("\n")), "\t&\t", len(text) / len(text.split("\n")))
			# for w in dict(sorted(d.items())): #sorted(d, key=d.get, reverse=True):
			# 	r_1.append(str(w))
			# 	r_2.append(str(d[w]))
			# 	print(w, "\t", d[w])

		# print("\t".join(r_1))
		# print("\t".join(r_2))


ku_ted = "tokenization_models/data/Datasets/Model_2/TED/TED_test_10_ku.txt"
en_ted = "tokenization_models/data/Datasets/Model_2/TED/TED_test_10_en_tokenized.txt"

# Search in the 10% data of TED (model 2)
# find lines within your specific bound
# find the corresponding ones in the other files

# with open(ku_ted, "r") as f:
# 	down, up = 10, 25
# 	f_text = f.read()
# 	selected_lines = list()
# 	for i in extract_sents(f_text, down, up):
# 		selected_lines.append(f_text.split("\n")[i])

# 	with open("data/Datasets/TED_lines/%s"%str(down)+"_"+str(up) + dirs[token][lang][dataset].split("/")[-1], "w") as fw:
# 		fw.write("\n".join(selected_lines))


for token in dirs:
	for lang in dirs[token]:
		for dataset in dirs[token][lang]:
			if lang == "ku" and "test" in dataset and "TED" in dataset:
				print(token, lang, dataset, dirs[token][lang][dataset])
				with open(ku_ted, "r") as f:
					down, up = 75, 100
					f_text = f.read()
					selected_lines = extract_sents(f_text, down, up)
					final_file_ku, final_file_en = list(), list()

					with open(dirs[token][lang][dataset], "r") as f_token:
						tokenized_data = f_token.read().replace("\u200c", "").split("\n")
					
					with open(en_ted, "r") as f_en:
						en_text = f_en.read().split("\n")

					for i in selected_lines:
						final_file_ku.append(tokenized_data[i])
						final_file_en.append(en_text[i])

					with open("data/Datasets/TED_lines/%s"%str(down)+"_"+str(up)+"_" + dirs[token][lang][dataset].split("/")[-1], "w") as fw:
						fw.write("\n".join(final_file_ku))

					with open(("data/Datasets/TED_lines/%s"%str(down)+"_"+str(up)+"_" + dirs[token][lang][dataset].split("/")[-1]).replace("_ku_", "_en_"), "w") as fw:
						fw.write("\n".join(final_file_en))


