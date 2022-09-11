
results=(
LASER2
SGPT-125M-weightedmean-msmarco-specb-bitfit
SGPT-125M-weightedmean-msmarco-specb-bitfit-doc
SGPT-125M-weightedmean-msmarco-specb-bitfit-que
SGPT-125M-weightedmean-nli-bitfit
SGPT-5.8B-weightedmean-msmarco-specb-bitfit
SGPT-5.8B-weightedmean-nli-bitfit
all-MiniLM-L6-v2
all-mpnet-base-v2
bert-base-uncased
contriever-base-msmarco
glove.6B.300d
gtr-t5-base
gtr-t5-xxl
komninos
msmarco-bert-co-condensor
sentence-t5-base
sentence-t5-xxl
sgpt-bloom-1b3-nli
sgpt-bloom-7b1-msmarco
sup-simcse-bert-base-uncased
unsup-simcse-bert-base-uncased
)

for i in "${results[@]}"
do
   echo "$i"
   python results_to_csv.py results/$i
done
