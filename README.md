# para-nmt-50m

Code to train models from "Pushing the Limits of Paraphrastic Sentence Embeddings with Millions of Machine Translations".

The code is written in python and requires numpy, scipy, theano, and the lasagne libraries.

To get started, run setup.sh to download a pre-trained 600d Trigram-Word model, a pre-trained 4096d BiLTSM Avg. model, and required files such as sample training data and evaluation data. The full 50M paraphrase corpora as well as a filtered, tokenized, 5M paraphrase corpora are available at http://www.cs.cmu.edu/~jwieting.

There is also a demo script that takes the model that you would like to train as a command line argument (check the script to see available choices). Check main/train.py for command line options.

If you use our code, models, or data for your work please cite:

@inproceedings{wieting-17-millions,
        author = {John Wieting and Kevin Gimpel},
        title = {Pushing the Limits of Paraphrastic Sentence Embeddings with Millions of Machine Translations},
        booktitle = {arXiv preprint arXiv:1711.05732},
        year = {2017}
}

@inproceedings{wieting-17-backtrans,
        author = {John Wieting, Jonathan Mallinson, and Kevin Gimpel},
        title = {Learning Paraphrastic Sentence Embeddings from Back-Translated Bitext},
        booktitle = {Proceedings of Empirical Methods in Natural Language Processing},
        year = {2017}
}
