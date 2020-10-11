# -*- coding: utf-8 -*-
# Sina Ahmadi (ahmadi.sina@outlook.com)
# October 2019
# https://github.com/sinaahmadi/KurdishMT

import sentencepiece as spm
import glob
from tokenizers import BertWordPieceTokenizer
from nltk.tokenize import WordPunctTokenizer
from mosestokenizer import *
import nltk
import random
import json

def train_bert():
	# https://huggingface.co/transformers/_modules/transformers/tokenization_bert.html

    files = ["Corpora/CS_V0_normalized_sent_per_line.txt",
        "Corpora/AsoSoft_Large_sent_per_line.txt",
        "Corpora/KTC_all_cleaned.txt",
        "Corpora/Lyrics_all_cleaned.txt",
        "Corpora/Tanztil_ku_normalized.txt"]

    vocab_size = 50000
    # Initialize a tokenizer
    tokenizer = BertWordPieceTokenizer(clean_text=True, handle_chinese_chars=False, strip_accents=True, lowercase=False)

    # And then train
    tokenizer.train(
        files,
        vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        limit_alphabet=1000,
        wordpieces_prefix="##",
    )

    tokenizer.save('./', 'ckb-wordpiece_%s'%str(vocab_size))


def test_tokenizer(text, model):
    # # Encode and decode
    # WordPiece = BertWordPieceTokenizer(model, lowercase=False)
    WordPiece = BertWordPieceTokenizer(model, strip_accents=True, clean_text=False, lowercase=False)
    WordPieceEncoder = WordPiece.encode(text)
    # print(WordPieceEncoder)

    # print(WordPieceEncoder.ids)
    # print(WordPieceEncoder.tokens)
    # print(WordPieceEncoder.offsets)
    
    return " ".join(WordPieceEncoder.tokens)




# import nltk

# with open("/Users/sina/My_GitHub/KurdishMT/corpus/Tanzil/stats/TMX/en.txt", "r") as f:
# 	en = f.read().split("\n")

# tokenized_en= list()

# for line in en:
# 	tokenized_en.append( " ".join(nltk.word_tokenize(line)) )

# with open("/Users/sina/My_GitHub/KurdishMT/corpus/Tanzil/stats/TMX/en_tokenized.txt", "w") as f:
# 	f.write("\n".join(tokenized_en))


# # print(WordPiece.decode(WordPieceEncoder.ids))

# if __name__ == '__main__':

def shuffler(source_file, target_file, mode, export_keys=None, import_keys=None):
	# split data 90% and 10% and set the 10% as test set aside
	
	with open(source_file, "r") as f:
		source = {i:j for i, j in enumerate(f.read().split("\n"))}

	with open(target_file, "r") as f:
		target = {i:j for i, j in enumerate(f.read().split("\n"))}

	if import_keys == None:
		keys = list(source.keys())
		random.shuffle(keys)

	else:
		with open(import_keys, "r") as f:
			keys = [int(i) for i in f.read().split()]

	if export_keys != None:
		with open("shuffle_keys/%s_shuffle_keys.txt"%export_keys, "w") as f:
			f.write("\n".join([str(i) for i in keys]))

	source_shuffled, target_shuffled = list(), list()
	for k in keys:
		source_shuffled.append(source[k])
		target_shuffled.append(target[k])

	output_data = dict()

	# if mode == "90-10":
	train_en_90 = source_shuffled[0: round(9 * len(source_shuffled)/10)]
	test_en_90 = source_shuffled[round(9 * len(source_shuffled)/10): ]

	train_ku_90 = target_shuffled[0: round(9 * len(source_shuffled)/10)]
	test_ku_90 = target_shuffled[round(9 * len(source_shuffled)/10): ]

	# output_data = {"en":[train_en_90, test_en_90], "ku": [train_ku_90, test_ku_90]}

	# if mode == "80-10-10":
		# 80%, 10%, 10%
	train_en = train_en_90[0: round(8 * len(train_en_90)/10)]
	val_en = train_en_90[round(8 * len(train_en_90)/10): round(8 * len(train_en_90)/10 + len(train_en_90)/10)]
	test_en = train_en_90[round(8 * len(train_en_90)/10 + len(train_en_90)/10): ]

	train_ku = train_ku_90[0: round(8 * len(train_ku_90)/10)]
	val_ku = train_ku_90[round(8 * len(train_ku_90)/10): round(8 * len(train_ku_90)/10 + len(train_ku_90)/10)]
	test_ku = train_ku_90[round(8 * len(train_ku_90)/10 + len(train_ku_90)/10): ]
	
	output_data = {"90-10": {"en":[train_en_90, test_en_90], "ku": [train_ku_90, test_ku_90]},
					"80-10-10": {"en":[train_en, val_en, test_en], "ku": [train_ku, val_ku, test_ku]}}

	return output_data


def tokenize_english():
	# English tokenizer
	en_datasets_dir = {
		"tanzil_en": "data/En/tanzil_en_lower_tokenized.txt",
		"wordnet_en": "data/En/wordnet_en_lower_tokenized.txt",
		"ted_en": "data/En/ted_en_lower_tokenized.txt"
	}
	for en_data in en_datasets_dir:
		tokenized_text = list()
		# with MosesTokenizer('en') as tokenize:
		with open(en_datasets_dir[en_data], "r") as f:
			text = f.read().split("\n")
			for sentence in text:
				tokenized_text.append(" ".join(nltk.word_tokenize(sentence.lower())))

			with open("data/En/" + en_data + "_lower_tokenized.txt", "w") as fw:
				fw.write("\n".join(tokenized_text))

			
def remove_punctuation(text):
	# puncts = ["،", ",", "؛", ";", "؟", "?", "٪", "%", "ـ", "_", "`", "'", "\""]
	puncts = ["`", "'", "\""]
	for i in puncts:
		text = text.replace(i, "")
	return text


def tokenizer(file_name, model, is_file=False, remove_punc=False):
	# Kurdish tokenizer	
	"""Given a list of sentences in Kurdish, return the tokenized one as text with spaces between tokens """
	if is_file:
		with open(file_name, "r") as f:
			text = f.read().split("\n")

	else:
		text = file_name

	models = {
		"wordpiece": 'tokenization_models/ckb-wordpiece_all_False_50000-vocab.txt',
		"bpe": 'tokenization_models/ckb_bpe_50k.model',
		"unigram": 'tokenization_models/ckb_unigram_50k.model',
		"WordPunct": ""
	}
	
	tokenized_text = list()

	if model == "wordpiece":
		WordPiece = BertWordPieceTokenizer(models[model], strip_accents=False, clean_text=False, lowercase=False)
		for sentence in text:
			WordPieceEncoder = WordPiece.encode(sentence)
			sentence_tokenized = " ".join(WordPieceEncoder.tokens)

			for token in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "##"]:
				sentence_tokenized = sentence_tokenized.replace(token, " ")

			tokenized_text.append(" ".join(sentence_tokenized.split()))
	
	elif model == "WordPunct":
		for sentence in text:
			tokenized_text.append(" ".join(WordPunctTokenizer().tokenize(sentence)))

	else:
		sp = spm.SentencePieceProcessor()
		sp.Load(models[model])

		for sentence in text:
			# print(" ".join( sp.EncodeAsPieces(sentence)).replace("▁", "") )
			tokenized_text.append(" ".join( sp.EncodeAsPieces(sentence)).replace("▁", ""))

	if remove_punc:
		return remove_punctuation("\n".join(tokenized_text))
	else:
		return "\n".join(tokenized_text)


def create_datasets(file_directories):
	# text = "منیش حەزدەکەم وەکو هاوڕێکانم لەو خولانەدا بەشداریی بکەن ، کە لەشاری خورماتوو دەکرێنەوە ، بۆئەوەی فێری شتێکی بەسوود بم"
	# input_sentence = "بە ناوی خوای گەورەومەزن"
	# models = sorted(glob.glob("models/Final_ones/*"))

	input_files = {
		"WordNet": "data/Ku/wordnet_ku_normalized.txt",
		"Tanzil": "data/Ku/tanzil_ku_normalized.txt",
		"TED": "data/Ku/ted_sent_ku_normalized.txt"
	}

	en_tokenized_data_dir = {
		"Tanzil": "data/En/tanzil_en_lower_tokenized.txt",
		"TED": "data/En/ted_en_lower_tokenized.txt",
		"WordNet": "data/En/wordnet_en_lower_tokenized.txt"
	}

	merged_ku = {
			"wordpiece": {"train": [], "val": [], "test": []},
			"bpe": {"train": [], "val": [], "test": []},
			"unigram": {"train": [], "val": [], "test": []},
			"WordPunct": {"train": [], "val": [], "test": []}
			}

	merged_en = {"train": [], "val": [], "test": []}
	
	for file_name in input_files:
		# Uncomment the following if you want to shuffle it again
		# shuffled_output = shuffler(en_tokenized_data_dir[file_name], input_files[file_name], "90-10", export_keys=file_name+"90-10")
		shuffled_output = shuffler(en_tokenized_data_dir[file_name], input_files[file_name], "90-10", import_keys="shuffle_keys/%s90-10_shuffle_keys.txt"%file_name)

		dataset_types = {
			"ku": {
				"train": shuffled_output["80-10-10"]["ku"][0], "val": shuffled_output["80-10-10"]["ku"][1], "test": shuffled_output["80-10-10"]["ku"][2]
			},
			"en": {
				"train": shuffled_output["80-10-10"]["en"][0], "val": shuffled_output["80-10-10"]["en"][1], "test": shuffled_output["80-10-10"]["en"][2]
			}
		}

		# Save the original split data
		# 10%
		with open("data/Datasets/Model_2/"+ file_name + "/" + file_name + "_test_10_ku.txt", "w") as f:
			f.write("\n".join(shuffled_output["90-10"]["ku"][1]))
		with open("data/Datasets/Model_2/"+ file_name + "/" + file_name + "_test_10_en_tokenized.txt", "w") as f:
			f.write("\n".join(shuffled_output["90-10"]["en"][1]))
		# 90%
		with open("data/Datasets/Model_1/"+ file_name + "/" + file_name + "_90_ku.txt", "w") as f:
			f.write("\n".join(shuffled_output["90-10"]["ku"][0]))
		with open("data/Datasets/Model_1/"+ file_name + "/" + file_name + "_90_en_tokenized.txt", "w") as f:
			f.write("\n".join(shuffled_output["90-10"]["en"][0]))

		# Save the Kurdish tokenized ones
		for model in ["wordpiece", "bpe", "unigram", "WordPunct"]:
			tokenized_ku = tokenizer(shuffled_output["90-10"]["ku"][1], model, remove_punc=True)
			with open("data/Datasets/Model_2/" + file_name + "/" +  file_name + "_test_10_ku_tokenized_%s.txt"%model, "w") as f:
				f.write(tokenized_ku)

		# Save the 80-10-10 data
		for lang in dataset_types:
			for d_type in dataset_types[lang]:
				print(d_type, lang, file_name)
				if lang == "en":
					merged_en[d_type].append("\n".join(dataset_types[lang][d_type]))

					with open("data/Datasets/Model_1/" + file_name + "/" +  file_name + "_%s_%s_tokenized.txt"%(d_type, lang), "w") as f:
						f.write("\n".join(dataset_types[lang][d_type]))
				else:
					with open("data/Datasets/Model_1/" + file_name + "/" +  file_name + "_%s_%s.txt"%(d_type, lang), "w") as f:
						f.write("\n".join(dataset_types[lang][d_type]))
					for model in ["wordpiece", "bpe", "unigram", "WordPunct"]:
						tokenized_text = tokenizer(dataset_types[lang][d_type], model, remove_punc=True)
						merged_ku[model][d_type].append(tokenized_text)
						# print(len("\n".join(merged_ku[model][d_type])))
						with open("data/Datasets/Model_1/" + file_name + "/" +  file_name + "_%s_%s_tokenized_%s.txt"%(d_type, lang, model), "w") as f:
							f.write(tokenized_text)


	for d_type in ["train", "val", "test"]:
		# Save the merged datasets (90%)

		# English
		with open("data/Datasets/Model_1/" + "%s_all_%s_tokenized.txt"%(d_type, "en"), "w") as f:
			f.write("\n".join(merged_en[d_type]))

		# Kurdish
		for model in ["wordpiece", "bpe", "unigram", "WordPunct"]:
			file_directories[model]["ku"][d_type] = "data/Datasets/Model_1/" + "%s_all_%s_tokenized_%s.txt"%(d_type, "ku", model)
			file_directories[model]["en"][d_type] = "data/Datasets/Model_1/" + "%s_all_%s_tokenized.txt"%(d_type, "en")
			with open("data/Datasets/Model_1/" + "%s_all_%s_tokenized_%s.txt"%(d_type, "ku", model), "w") as f:
				f.write("\n".join(merged_ku[model][d_type]))

	return file_directories

initial_file_directories = {
	"wordpiece": {
		"ku":{
				"train": "",
				"val": "",
				"test_1": "",
				"test_2": ""
			},
		"en":{
				"train": "",
				"val": "",
				"test_1": "",
				"test_2": ""
			}
	},
	"bpe": {
		"ku":{
				"train": "",
				"val": "",
				"test_1": "",
				"test_2": ""
			},
		"en":{
				"train": "",
				"val": "",
				"test_1": "",
				"test_2": ""
			}
	},
	"unigram": {
		"ku":{
				"train": "",
				"val": "",
				"test_1": "",
				"test_2": ""
			},
		"en":{
				"train": "",
				"val": "",
				"test_1": "",
				"test_2": ""
			}
	},
	"WordPunct": {
		"ku":{
				"train": "",
				"val": "",
				"test_1": "",
				"test_2": ""
			},
		"en":{
				"train": "",
				"val": "",
				"test_1": "",
				"test_2": ""
			}
	}
}


# tokenize_english()
completed_file_directories = create_datasets(initial_file_directories)

with open('data_directories.json', 'w') as outfile:
	json.dump(completed_file_directories, outfile)


# # 15 06 2020 Error with the removed hamza was fixed in the tokenization
# print(tokenizer(["ئەمەوێ بزانم لەوێ چی ئەکەی", "ڕۆژێک لە ڕۆژان، نەخۆشیەکی ترسناک بوو بە هەڕەشە بۆ سەر ژیانی منداڵان . ",
# 	" شتی تردان و ووتیان '' ئەمە شتێکی ترسناک ؟ وە حکومەتیش زۆر تووڕە و دلگران بوو لەم هەواڵە . "], "wordpiece", False, True))

# print(tokenizer(["ئەمەوێ بزانم لەوێ چی ئەکەی", "ڕۆژێک لە ڕۆژان، نەخۆشیەکی ترسناک بوو بە هەڕەشە بۆ سەر ژیانی منداڵان . ",
# 	" شتی تردان و ووتیان '' ئەمە شتێکی ترسناک ؟ وە حکومەتیش زۆر تووڕە و دلگران بوو لەم هەواڵە . "], "wordpiece", False, False))

