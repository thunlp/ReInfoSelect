java -cp RankLib-2.1-patched.jar ciir.umass.edu.features.FeatureManager -input ../features/cknrm_features -output ../features/ -k 2
java -jar utils/RankLib-2.1-patched.jar -train ../features/cknrm_features -ranker 4 -kcv 2 -kcvmd ../checkpoints/ -kcvmn ca -metric2t NDCG@20 -metric2T NDCG@20
java -jar utils/RankLib-2.1-patched.jar -load ../checkpoints/f1.ca -rank ../features/f1.test.cknrm_features -score f1.score
java -jar utils/RankLib-2.1-patched.jar -load ../checkpoints/f2.ca -rank ../features/f2.test.cknrm_features -score f2.score
python utils/gen_trec.py -dev ../data/dev_toy.tsv -res ../results/cknrm_ca.trec
rm f1.score
rm f2.score
