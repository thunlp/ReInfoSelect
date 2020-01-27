import pytrec_eval

def ndcg(qrels, trec, k):
    with open(qrels, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)

    with open(trec, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, pytrec_eval.supported_measures)
    results = evaluator.evaluate(run)
    for query_id, query_measures in sorted(results.items()):
        pass

    mes = {}
    for measure in sorted(query_measures.keys()):
        mes[measure] = pytrec_eval.compute_aggregated_measure(measure, [query_measures[measure] for query_measures in results.values()])

    metric = 'ndcg_cut_%d' % k
    if metric not in mes:
        print('Depth of NDCG not available.')
        exit()
    ndcg = mes[metric]

    return ndcg

def err(qrels, trec, k):
    res = os.popen('./gdeval.pl -k %d %s %s' % (k, qrels, trec))
    for r in res:
        pass
    scores = r.strip('\n').split(',')
    #ndcg = float(scores[2])
    err = float(scores[3])

    return err
