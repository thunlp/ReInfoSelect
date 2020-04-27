import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dev', type=str, default='../data/dev_toy.tsv')
    parser.add_argument('-res', type=str, default='../results/cknrm.trec')
    args = parser.parse_args()

    score_dic = {}
    with open('f1.score', 'r') as r:
        for line in r:
            line = line.strip('\n').split('\t')
            score_dic[line[0] + '$' + line[1]] = line[2]

    outs = {}
    with open(args.dev, 'r') as r:
        qid = ''
        cnt = 0
        for line in r:
            line = line.strip().split('\t')
            if line[3] != qid:
                qid = line[3]
                cnt = 0
                outs[line[3]] = {}
            outs[line[3]][line[4]] = float(score_dic[line[3]+'$'+str(cnt)])
            cnt += 1

    f = open(args.res, 'w')
    for qid in outs:
        ps = {}
        out_idx = sorted(outs[qid].items(), key=lambda x:x[1], reverse=True)
        for i, out in enumerate(out_idx):
            if out[0] not in ps:
                ps[out[0]] = 1
                f.write(' '.join([qid, 'Q0', out[0], str(len(ps)), str(out[1]), 'default']) + '\n')
    f.close()
