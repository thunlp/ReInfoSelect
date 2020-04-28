java -jar utils/RankLib-2.1-patched.jar -train ../features/cknrm_features -ranker 4 -kcv 1 -kcvmd ../checkpoints/ -kcvmn ca -metric2t NDCG@20 -metric2T NDCG@20
java -jar utils/RankLib-2.1-patched.jar -load ../checkpoints/f1.ca -rank ../features/cknrm_features -score f1.score
python utils/gen_trec.py -dev ../data/dev_toy.tsv -res ../results/cknrm_ca.trec
rm f1.score
