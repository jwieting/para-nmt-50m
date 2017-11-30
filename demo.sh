#load and evaluate pre-trained models
if [ "$1" == "load-trigram-word" ]; then
    sh train.sh -combination_type ngram-word -combination_method concat -loadmodel ../data/ngram-word-concat-40.pickle -margin 0.4 -data ../data/para-nmt-50m-small.txt
elif [ "$1" == "load-bilstm-avg" ]; then
    sh train.sh -model bilstmavg -loadmodel ../data/bilstmavg-4096-40.pickle -axis 2 -dim 2048 -margin 0.4 -data ../data/para-nmt-50m-small.txt

#train paraphrastic sentence embedding models
elif [ "$1" == "wordavg" ]; then
    sh train.sh -model wordaverage -margin 0.4 -data ../data/para-nmt-50m-small.txt
elif [ "$1" == "trigram-avg" ]; then
    sh train.sh -model wordaverage -wordtype 3grams -margin 0.4 -data ../data/para-nmt-50m-small.txt
elif [ "$1" == "lstm-avg" ]; then
    sh train.sh -model lstmavg -margin 0.4 -data ../data/para-nmt-50m-small.txt
elif [ "$1" == "bilstm-avg" ]; then
    sh train.sh -model bilstmavg -margin 0.4 -data ../data/para-nmt-50m-small.txt
elif [ "$1" == "bilstm-max" ]; then
    sh train.sh -model bilstmmax -margin 0.4 -data ../data/para-nmt-50m-small.txt
elif [ "$1" == "ngram-word-concat" ]; then
    sh train.sh -combination_type ngram-word -combination_method concat -margin 0.4 -data ../data/para-nmt-50m-small.txt
elif [ "$1" == "ngram-lstm-concat" ]; then
    sh train.sh -combination_type ngram-lstm -combination_method concat -margin 0.4 -data ../data/para-nmt-50m-small.txt
elif [ "$1" == "word-lstm-concat" ]; then
    sh train.sh -combination_type word-lstm -combination_method concat -margin 0.4 -data ../data/para-nmt-50m-small.txt
elif [ "$1" == "ngram-word-lstm-concat" ]; then
    sh train.sh -combination_type ngram-word-lstm -combination_method concat -margin 0.4 -data ../data/para-nmt-50m-small.txt
else
    echo "$1 not a valid option."
fi
