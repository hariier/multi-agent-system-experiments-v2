"""Offline retrieval experiments. Uses the SAME retriever as Agent2, no API calls."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from graph_retrieval import CommonGraphRetriever, RetrievalConfig, local_embedder
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graphs',default='prepared_graphs/task2_graphs.jsonl')
    p.add_argument('--labels',default='prepared_graphs/task2_labels.jsonl')
    p.add_argument('--output',default='retrieval_results')
    p.add_argument('--modes',nargs='+',default=['graph','topk','static_graph','full'],
                   choices=['graph','topk','static_graph','full','none'])
    p.add_argument('--budgets',nargs='+',type=int,default=[5,8,12])
    p.add_argument('--edge-penalty',type=float,default=0.15,
                   help='lambda in graph/static_graph path utility; ignored by topk/full')
    p.add_argument('--embedding-model')
    args=p.parse_args()

    if args.edge_penalty < 0:
        p.error('--edge-penalty must be non-negative')

    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    protected=['results.jsonl','metrics.csv','summary.json','diagnostics_summary.json','run_config.json']
    if any((out/name).exists() for name in protected):
        p.error('Use a fresh output directory')

    graphs=list(read_jsonl(args.graphs))
    labels={x['id']:x for x in read_jsonl(args.labels)}
    manifest_path=Path(args.graphs).parent/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if len(graphs)!=manifest['entailment_samples']:
            raise ValueError('Incomplete preprocessing output')
        if args.embedding_model != manifest.get('embedding_model'):
            raise ValueError('Embedding model differs from preprocessing')
    else:
        raise ValueError('Preprocessing manifest missing; input may be incomplete')

    if not graphs or any(b<1 for b in args.budgets):
        raise ValueError('Empty graphs or invalid budgets')
    if {g['id'] for g in graphs}!=set(labels):
        raise ValueError('Graph/label IDs must match exactly')

    embedder=local_embedder(args.embedding_model)
    records=[]

    with (out/'results.jsonl').open('w',encoding='utf-8') as f:
        for g in graphs:
            expected=g.get('build_config',{}).get('feature_backend','tfidf')
            if ('embeddings' in expected) != bool(embedder):
                raise ValueError('Use the same embedding backend as preprocessing')

            retriever=CommonGraphRetriever(g,embedder=embedder)
            for mode in args.modes:
                budgets=([max(args.budgets)] if mode in {'full','none'} else args.budgets)
                for budget in budgets:
                    retriever.config=RetrievalConfig(
                        mode=mode,
                        budget=budget,
                        edge_penalty=args.edge_penalty,
                    )
                    result=retriever.retrieve(g['query'])

                    # Labels are accessed only AFTER retrieval.
                    label=labels[g['id']]
                    gold=set(label['gold_supporting_facts'])
                    pred=set(result['selected_nodes'])
                    hits=len(pred & gold)
                    precision=hits/len(pred) if pred else 0
                    recall=hits/len(gold)
                    f1=2*precision*recall/(precision+recall) if precision+recall else 0

                    diag=result.get('diagnostics',{})
                    component=set(diag.get('seed_component_content_nodes',[]))
                    reachable_gold=len(component & gold)
                    seed_id=result['seed_nodes'][0] if result.get('seed_nodes') else None

                    row=dict(
                        id=g['id'],
                        mode=mode,
                        budget=budget if mode not in {'full','none'} else 0,
                        edge_penalty=args.edge_penalty,
                        proof_depth=label['proof_depth'],
                        gold_count=len(gold),
                        selected_count=len(pred),
                        precision=precision,
                        recall=recall,
                        f1=f1,
                        complete_recall=int(gold<=pred),
                        non_gold_count=len(pred-gold),
                        context_characters=len(result['context']),
                        seed_id=seed_id,
                        seed_is_gold=(int(seed_id in gold) if seed_id is not None else None),
                        seed_component_content_count=diag.get('seed_component_content_count'),
                        reachable_gold_count=reachable_gold if seed_id is not None else None,
                        reachable_gold_ratio=(reachable_gold/len(gold) if seed_id is not None else None),
                        stop_reason=diag.get('stop_reason'),
                    )
                    records.append(row)
                    f.write(json.dumps(dict(result=result,evaluation=row),ensure_ascii=False)+'\n')

    with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summaries=[]
    diagnostic_summaries=[]
    groups=sorted({(r['mode'],r['budget']) for r in records})
    for mode,budget in groups:
        rows=[r for r in records if r['mode']==mode and r['budget']==budget]
        summary=dict(
            mode=mode,
            budget=budget,
            edge_penalty=args.edge_penalty,
            samples=len(rows),
            **{key:mean(r[key] for r in rows)
               for key in ['precision','recall','f1','complete_recall','selected_count']}
        )
        summaries.append(summary)

        graph_rows=[r for r in rows if r['seed_id'] is not None]
        reason_counts=Counter(r['stop_reason'] for r in rows)
        diagnostic_summaries.append(dict(
            mode=mode,
            budget=budget,
            samples=len(rows),
            seed_gold_rate=(mean(r['seed_is_gold'] for r in graph_rows) if graph_rows else None),
            mean_reachable_gold_ratio=(mean(r['reachable_gold_ratio'] for r in graph_rows) if graph_rows else None),
            stop_reason_counts=dict(reason_counts),
            stop_reason_rates={k: reason_counts.get(k,0)/len(rows) for k in STOP_REASONS},
        ))

    (out/'summary.json').write_text(
        json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'diagnostics_summary.json').write_text(
        json.dumps(diagnostic_summaries,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'run_config.json').write_text(
        json.dumps(dict(
            graphs=str(Path(args.graphs).resolve()),
            labels=str(Path(args.labels).resolve()),
            modes=args.modes,
            budgets=args.budgets,
            edge_penalty=args.edge_penalty,
            embedding_model=args.embedding_model,
            preprocessing_manifest=manifest,
        ),ensure_ascii=False,indent=2),encoding='utf-8')

    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
