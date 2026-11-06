"""Development-set grid search for EntailmentBank Task 2 graph retrieval.

Keeps the original retrieval method unchanged:
- single seed
- mutual-nearest-neighbor graph
- problem-aware edge weights
- multi-source Dijkstra expansion
- positive-utility stopping
- content-node budget

Only tunes existing parameters:
- graph neighbors k
- path penalty lambda (RetrievalConfig.edge_penalty)

No API calls and no model training.
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean

from graph_retrieval import (
    CommonGraphRetriever,
    EntailmentBankAdapter,
    RetrievalConfig,
    local_embedder,
)
from prepare_graphs import read_jsonl


STOP_REASONS = [
    'budget_reached',
    'no_reachable_candidate',
    'budget_blocked',
    'no_positive_gain',
    'no_positive_seed',
    'no_content',
    'stopped',
]


def normalize_text(text):
    return ' '.join(re.findall(r'[a-z0-9]+', str(text).lower()))


def sample_signature(sample):
    facts=sample['meta']['triples']
    return (
        normalize_text(sample['question']),
        tuple(sorted(normalize_text(v) for v in facts.values())),
    )


def check_dev_test_overlap(dev_samples, test_path):
    report=dict(
        checked=False,
        dev_samples=len(dev_samples),
        test_samples=None,
        overlapping_ids=[],
        overlapping_question_fact_signatures=0,
    )
    if not test_path:
        return report
    path=Path(test_path)
    if not path.exists():
        report['warning']=f'Test file not found: {path}'
        return report

    test_samples=list(read_jsonl(path,allow_duplicate_ids=True))
    report['checked']=True
    report['test_samples']=len(test_samples)

    dev_ids={x['id'] for x in dev_samples}
    test_ids={x['id'] for x in test_samples}
    report['overlapping_ids']=sorted(dev_ids & test_ids)

    dev_sig={sample_signature(x) for x in dev_samples}
    test_sig={sample_signature(x) for x in test_samples}
    report['overlapping_question_fact_signatures']=len(dev_sig & test_sig)

    if report['overlapping_ids'] or report['overlapping_question_fact_signatures']:
        raise ValueError(
            'Development/test overlap detected. Do not tune on a split that overlaps test. '
            f"IDs={len(report['overlapping_ids'])}, "
            f"signatures={report['overlapping_question_fact_signatures']}"
        )
    return report


def graph_connectivity_stats(graph):
    ids=[n['id'] for n in graph['nodes']]
    adj={nid:set() for nid in ids}
    for e in graph['edges']:
        a,b=e['source'],e['target']
        adj[a].add(b)
        adj[b].add(a)

    seen=set()
    component_sizes=[]
    for nid in ids:
        if nid in seen:
            continue
        stack=[nid]
        seen.add(nid)
        size=0
        while stack:
            u=stack.pop()
            size+=1
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        component_sizes.append(size)

    return dict(
        nodes=len(ids),
        edges=len(graph['edges']),
        components=len(component_sizes),
        isolated_nodes=sum(1 for nid in ids if not adj[nid]),
        largest_component=max(component_sizes) if component_sizes else 0,
        mean_degree=(sum(len(adj[n]) for n in ids)/len(ids) if ids else 0),
    )


def evaluate_one(graph,label,k,penalty,budget,embedder):
    retriever=CommonGraphRetriever(
        graph,
        config=RetrievalConfig(mode='graph',budget=budget,edge_penalty=penalty),
        embedder=embedder,
    )
    result=retriever.retrieve(graph['query'])

    # Gold labels are consulted only after retrieval.
    gold=set(label['gold_supporting_facts'])
    pred=set(result['selected_nodes'])
    hits=len(pred & gold)
    precision=hits/len(pred) if pred else 0
    recall=hits/len(gold)
    f1=2*precision*recall/(precision+recall) if precision+recall else 0

    diag=result.get('diagnostics',{})
    seed_id=result['seed_nodes'][0] if result.get('seed_nodes') else None
    component=set(diag.get('seed_component_content_nodes',[]))
    reachable_gold=len(component & gold)

    candidate_facts={n['id']:n['text'] for n in graph['nodes'] if n['kind']=='content'}
    row=dict(
        id=graph['id'],
        original_id=graph.get('original_id'),
        question=graph['query'],
        k=k,
        edge_penalty=penalty,
        budget=budget,
        proof_depth=label['proof_depth'],
        gold_count=len(gold),
        seed_id=seed_id,
        seed_is_gold=(int(seed_id in gold) if seed_id is not None else None),
        seed_score=diag.get('seed_score'),
        seed_component_content_count=diag.get('seed_component_content_count'),
        reachable_gold_count=(reachable_gold if seed_id is not None else None),
        reachable_gold_ratio=(reachable_gold/len(gold) if seed_id is not None else None),
        stop_reason=diag.get('stop_reason'),
        selected_count=len(pred),
        precision=precision,
        recall=recall,
        f1=f1,
        complete_recall=int(gold<=pred),
        selected_nodes=result['selected_nodes'],
        gold_nodes=sorted(gold),
        candidate_facts=candidate_facts,
        positive_score_content_count=diag.get('positive_score_content_count'),
        iterations=diag.get('iterations'),
        last_reachable_candidates=diag.get('last_reachable_candidates'),
        last_budget_feasible_candidates=diag.get('last_budget_feasible_candidates'),
        last_positive_utility_candidates=diag.get('last_positive_utility_candidates'),
        last_best_utility=diag.get('last_best_utility'),
    )
    return row


def summarize(rows,k,penalty,budget):
    reasons=Counter(r['stop_reason'] for r in rows)
    seed_rows=[r for r in rows if r['seed_is_gold'] is not None]
    return dict(
        k=k,
        edge_penalty=penalty,
        budget=budget,
        samples=len(rows),
        precision=mean(r['precision'] for r in rows),
        recall=mean(r['recall'] for r in rows),
        f1=mean(r['f1'] for r in rows),
        complete_recall=mean(r['complete_recall'] for r in rows),
        selected_count=mean(r['selected_count'] for r in rows),
        seed_gold_rate=(mean(r['seed_is_gold'] for r in seed_rows) if seed_rows else None),
        reachable_gold_ratio=(mean(r['reachable_gold_ratio'] for r in seed_rows) if seed_rows else None),
        **{f'stop_{reason}_rate': reasons.get(reason,0)/len(rows) for reason in STOP_REASONS},
    )


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dev',required=True,
                   help='EntailmentBank Task 2 development/validation JSONL. Never pass test.jsonl here.')
    p.add_argument('--test',default='test.jsonl',
                   help='Optional test JSONL used only for overlap checking; no test labels are used for tuning.')
    p.add_argument('--output',default='retrieval_tuning_dev')
    p.add_argument('--neighbors',nargs='+',type=int,default=[3,5,8])
    p.add_argument('--edge-penalties',nargs='+',type=float,default=[0.03,0.08,0.15])
    p.add_argument('--budget',type=int,default=8)
    p.add_argument('--embedding-model')
    args=p.parse_args()

    if Path(args.dev).resolve()==Path(args.test).resolve():
        p.error('--dev and --test must be different files')
    if any(k<1 for k in args.neighbors):
        p.error('neighbors must be positive')
    if any(x<0 for x in args.edge_penalties):
        p.error('edge penalties must be non-negative')
    if args.budget<1:
        p.error('budget must be positive')

    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    protected=[
        'grid_summary.json','grid_summary.csv','diagnostics.jsonl',
        'stop_reason_summary.json','best_config.json','graph_stats.json',
        'overlap_check.json','run_config.json',
    ]
    if any((out/name).exists() for name in protected):
        p.error('Use a fresh output directory')

    dev_samples=list(read_jsonl(args.dev,allow_duplicate_ids=True))
    if not dev_samples:
        raise ValueError('Empty development set')

    overlap=check_dev_test_overlap(dev_samples,args.test)
    (out/'overlap_check.json').write_text(
        json.dumps(overlap,ensure_ascii=False,indent=2),encoding='utf-8')

    embedder=local_embedder(args.embedding_model)
    adapter=EntailmentBankAdapter()
    all_diagnostics=[]
    summaries=[]
    graph_stats=[]

    for k in args.neighbors:
        built=[]
        per_graph_stats=[]
        for row_no,sample in enumerate(dev_samples,1):
            graph=adapter.build(sample,embedder,k)
            label=adapter.labels(sample)
            instance_id=f"{sample['id']}::row{row_no:04d}"
            graph.update(id=instance_id,original_id=sample['id'])
            label.update(id=instance_id,original_id=sample['id'])
            built.append((graph,label))
            per_graph_stats.append(graph_connectivity_stats(graph))

        graph_stats.append(dict(
            k=k,
            samples=len(built),
            mean_nodes=mean(x['nodes'] for x in per_graph_stats),
            mean_edges=mean(x['edges'] for x in per_graph_stats),
            mean_components=mean(x['components'] for x in per_graph_stats),
            mean_isolated_nodes=mean(x['isolated_nodes'] for x in per_graph_stats),
            mean_largest_component=mean(x['largest_component'] for x in per_graph_stats),
            mean_degree=mean(x['mean_degree'] for x in per_graph_stats),
        ))

        for penalty in args.edge_penalties:
            rows=[]
            for graph,label in built:
                result=evaluate_one(
                    graph,label,k,penalty,args.budget,embedder)
                rows.append(result)
                all_diagnostics.append(result)
            summaries.append(summarize(rows,k,penalty,args.budget))

    # Pre-fixed selection: macro F1, then complete recall, then fewer selected nodes.
    best=max(
        summaries,
        key=lambda x:(x['f1'],x['complete_recall'],-x['selected_count'])
    )
    best_config=dict(
        neighbors=best['k'],
        edge_penalty=best['edge_penalty'],
        budget=args.budget,
        embedding_model=args.embedding_model,
        selection_rule='max macro F1; tie -> max complete_recall; tie -> min selected_count',
        development_summary=best,
    )

    with (out/'diagnostics.jsonl').open('w',encoding='utf-8') as f:
        for row in all_diagnostics:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')

    with (out/'grid_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    (out/'grid_summary.json').write_text(
        json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'best_config.json').write_text(
        json.dumps(best_config,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'graph_stats.json').write_text(
        json.dumps(graph_stats,ensure_ascii=False,indent=2),encoding='utf-8')

    stop_summary=[
        {k:v for k,v in row.items() if k in {
            'k','edge_penalty','budget','samples','seed_gold_rate','reachable_gold_ratio',
            *[f'stop_{r}_rate' for r in STOP_REASONS]
        }}
        for row in summaries
    ]
    (out/'stop_reason_summary.json').write_text(
        json.dumps(stop_summary,ensure_ascii=False,indent=2),encoding='utf-8')

    (out/'run_config.json').write_text(
        json.dumps(dict(
            dev=str(Path(args.dev).resolve()),
            test_for_overlap_check=str(Path(args.test).resolve()) if args.test else None,
            neighbors=args.neighbors,
            edge_penalties=args.edge_penalties,
            budget=args.budget,
            embedding_model=args.embedding_model,
            overlap_check=overlap,
        ),ensure_ascii=False,indent=2),encoding='utf-8')

    print(json.dumps(dict(
        development_samples=len(dev_samples),
        overlap_check=overlap,
        grid=summaries,
        best_config=best_config,
    ),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
