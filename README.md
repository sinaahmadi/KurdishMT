# Kurdish Machine Translation

This repository contains the data and the models described in the paper ["Towards Machine Translation for the Kurdish Language"](https://arxiv.org/abs/2010.06041). Please note that `ku` and `en` refer to [Sorani Kurdish](https://en.wikipedia.org/wiki/Sorani) (=`ckb` in ISO 639-3) and English, respectively.

---

We share technical details of our models here so that future systems can be compared with the current project as follows:

- **Datasets**: there are two sets of datasets, each one containing training, testing and validation datasets. These sets are called `Model 1` and `Model 2`.
- **Tokenization models**: you can use the tokenization models which are trained using the [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) and [SentencePiece](https://github.com/google/sentencepiece). Our models are trained using the following corpora:
	- [Pewan](https://sinaahmadi.github.io/resources/pewan.html)
	- [AsoSoft corpus](https://github.com/AsoSoft/AsoSoft-Text-Corpus)
	- [Kurdish Textbooks Corpus (KTC)](https://sinaahmadi.github.io/resources/ktc.html)
	- [Corpus of the Sorani Kurdish Folkloric Lyrics](https://sinaahmadi.github.io/resources/kurdishfolkcorpus.html)
- **Word Embeddings**: we used the [fastText word vectors for Kurdish](https://fasttext.cc/docs/en/crawl-vectors.html) and GloVe for English.
- **System outputs**: the translation outputs of our best model for the two sets of data are provided in the [output dicectory](https://github.com/sinaahmadi/KurdishMT/tree/master/output).


## See also
Shortly after this project, a set of parallel corpora containing Sorani-Kurmanji, Sorani-English and Kurmanji-English sentences was published. Check it out at [https://github.com/KurdishBLARK/InterdialectCorpus](https://github.com/KurdishBLARK/InterdialectCorpus).

## Cite this paper

If you use any part of the data, please consider citing **[this paper](https://www.aclweb.org/anthology/2020.loresmt-1.12/)** as follows:

	@inproceedings{ahmadi-masoud-2020-towards,
	    title = "Towards Machine Translation for the {K}urdish Language",
	    author = "Ahmadi, Sina  and
	      Masoud, Maraim",
	    booktitle = "Proceedings of the 3rd Workshop on Technologies for MT of Low Resource Languages",
	    month = dec,
	    year = "2020",
	    address = "Suzhou, China",
	    publisher = "Association for Computational Linguistics",
	    url = "https://www.aclweb.org/anthology/2020.loresmt-1.12",
	    pages = "87--98"
	}


## License

[Apache License](https://github.com/sinaahmadi/KurdishMT/blob/master/LICENSE)