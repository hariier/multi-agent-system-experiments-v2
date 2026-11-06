"""Preprocess both datasets; labels are written to a separate file."""
import argparse
import json
from pathlib import Path
from graph_retrieval import CourseGraphAdapter, EntailmentBankAdapter, local_embedder


def read_jsonl(path, allow_duplicate_ids=False):
    seen=set()
    with Path(path).open(encoding='utf-8-sig') as f:
        for line_no,line in enumerate(f,1):
            if not line.strip(): continue
            item=json.loads(line)
            if item['id'] in seen and not allow_duplicate_ids: raise ValueError(f'Duplicate id at line {line_no}')
            seen.add(item['id'])
            yield item


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--course',default='knowledge_data/计算机网络课程知识.json')
    p.add_argument('--entailment',default='test.jsonl')
    p.add_argument('--output',default='prepared_graphs')
    p.add_argument('--embedding-model',help='Existing local sentence-transformers model directory')
    p.add_argument('--neighbors',type=int,default=3)
    args=p.parse_args()
    if args.neighbors<1: p.error('neighbors must be positive')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    paths=[out/'course_graph.json',out/'task2_graphs.jsonl',out/'task2_labels.jsonl',out/'manifest.json']
    if any(path.exists() for path in paths):
        p.error('Output already exists. Use a new --output directory to preserve previous experiments.')
    embedder=local_embedder(args.embedding_model)
    course=CourseGraphAdapter().build(args.course,embedder,args.neighbors)
    graphs=[]; labels=[]; adapter=EntailmentBankAdapter()
    # Multiple proof instances may share a question id. Preserve every row.
    for row,sample in enumerate(read_jsonl(args.entailment, allow_duplicate_ids=True),1):
        graph=adapter.build(sample,embedder,args.neighbors)
        label=adapter.labels(sample)
        instance_id=f"{sample['id']}::row{row:04d}"
        graph.update(id=instance_id,original_id=sample['id'])
        label.update(id=instance_id,original_id=sample['id'])
        graphs.append(graph); labels.append(label)
    if not graphs: raise ValueError('Empty dataset')
    paths[0].write_text(json.dumps(course,ensure_ascii=False,indent=2),encoding='utf-8')
    with paths[1].open('w',encoding='utf-8') as gf, paths[2].open('w',encoding='utf-8') as lf:
        for graph,label in zip(graphs,labels):
            gf.write(json.dumps(graph,ensure_ascii=False)+'\n')
            lf.write(json.dumps(label,ensure_ascii=False)+'\n')
    manifest=dict(entailment_samples=len(graphs),unique_question_ids=len({g['original_id'] for g in graphs}),course_nodes=len(course['nodes']),course_edges=len(course['edges']),
                  neighbors=args.neighbors,embedding_model=args.embedding_model,feature_backend=course['build_config']['feature_backend'],
                  input_course=str(Path(args.course).resolve()),input_entailment=str(Path(args.entailment).resolve()))
    paths[3].write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
