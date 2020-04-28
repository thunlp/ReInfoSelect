import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=list, default=['../results/cknrm.trec', '../results/cknrm_ca.trec'])
    parser.add_argument('output', type=str, default='../results/cknrm_ensemble.trec')
    args = parser.parse_args()

    res = {}
    for trec in args.input:
        with open(trec, 'r') as r:
            for line in r:
                line = line.strip().split()
                qid = line[0]
                did = line[2]
                output = float(line[4])
                if qid not in res:
                    res[qid] = {}
                if did not in res[qid]:
                    res[qid][did] = 0
                res[qid][did] += output

    result_dict = {}
    for qid in res:
        result_dict[qid] = []
        for did in res[qid]:
            result_dict[qid].append((did, res[qid][did]))

    with open('recknrm_ensb.txt', 'w') as writer:
        for qid, values in result_dict.items():
            res = sorted(values, key=lambda x: x[1], reverse=True) # reverse sort by value[1] (output), res=[(docid,score)]
            for rank,value in enumerate(res):
                writer.write(qid+' '+'Q0'+' '+str(value[0])+' '+str(rank+1)+' '+str(value[1])+' '+'default'+'\n')

if __name__ == "__main__":
    main()
