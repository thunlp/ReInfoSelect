import json
import re

regex_drop_char = re.compile('[^a-z0-9\s]+')
regex_multi_space = re.compile('\s+')

def raw2tok(s):
    lst = regex_multi_space.sub(' ', regex_drop_char.sub(' ', s.lower())).strip().split()
    return lst

doc_dic = {}
f = open('./data/ClueWeb09/CW09Docs.json')
for line in f:
    line = line.strip('\n')
    single_doc = json.loads(line)
    single_doc['body'] = raw2tok(single_doc['body'])
    single_doc['title'] = raw2tok(single_doc['title'])
    doc_dic[single_doc['id']] = ' '.join(single_doc['title']) + ' ' + ' '.join(single_doc['body'])

query_dic = {}
f = open('./data/ClueWeb09/CW09Queries')
for line in f:
    line = line.strip('\n').split('\t')
    query_dic[line[0]] = line[1]

score = {}
f = open('./data/ClueWeb09/qrels')
for line in f:
    line1 = line.strip('\n').split(' ')
    line = []
    for word in line1:
        if word != '':
            line.append(word)
    score[line[0]+'$'+line[2]] = int(line[3])

for i in range(5):
    print(i)
    train_dic = {}
    val_dic = {}
    test_dic = {}

    f = open('./data/ClueWeb09/CW09.trec')
    for line in f:
        line = line.strip('\n').split(' ')
        if int(line[0])%5 == i:
            if line[0] not in test_dic:
                test_dic[line[0]] = []
            if len(test_dic[line[0]]) == 100:
                continue
            rel_score = score[line[0] + '$' + line[2]] if line[0] + '$' + line[2] in score else 0
            test_dic[line[0]].append([line[2], rel_score, line[4]])
        if (int(line[0])+1)%5 == i:
            if line[0] not in val_dic:
                val_dic[line[0]] = []
            if len(val_dic[line[0]]) == 100:
                continue
            rel_score = score[line[0] + '$' + line[2]] if line[0] + '$' + line[2] in score else 0
            val_dic[line[0]].append([line[2], rel_score, line[4]])
        else:
            if line[0] not in train_dic:
                train_dic[line[0]] = []
            if len(train_dic[line[0]]) == 100:
                continue
            rel_score = score[line[0] + '$' + line[2]] if line[0] + '$' + line[2] in score else 0
            train_dic[line[0]].append([line[2], rel_score])

    def draw(fout, ddic):
        fout = open(fout, 'w')
        for qid in ddic:
            docs = ddic[qid]
            length = len(docs)
            for i in range(length):
                for j in range(length):
                    if docs[i][1] > docs[j][1]:
                        if docs[j][0] not in doc_dic:
                            continue
                        elif docs[i][0] not in doc_dic:
                            continue
                        else:
                            fout.write(query_dic[qid] + '\t' + doc_dic[docs[i][0]] + '\t' + doc_dic[docs[j][0]] + '\n')

    def draw_dev(fout, ddic):
        fout = open(fout, 'w')
        for qid in ddic:
            docs = ddic[qid]
            length = len(docs)
            for i in range(length):
                if docs[i][0] not in doc_dic:
                    continue
                fout.write(query_dic[qid] + '\t' + doc_dic[docs[i][0]] + '\t' + str(docs[i][1]) + '\t' +  qid + '\t' + docs[i][0] + '\t' + docs[i][2] + '\n')# toby change

    draw('cross_validate/train_09_%d.txt' % i, train_dic)
    draw_dev('cross_validate/dev_09_%d.txt' % i, val_dic)
    draw_dev('cross_validate/eval_09_%d.txt' % i, test_dic)
