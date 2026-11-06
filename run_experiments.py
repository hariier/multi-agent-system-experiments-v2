"""统一实验入口：多模型、三份数据、基线/消融、原子检查点与汇总。

先用 --dry-run 检查规模，用 --mock 检查整个流程；只有 --live 才调用API。
成功样本自动跳过，失败样本需 --retry-errors 才重试。不要同时运行两个进程
写入同一个输出目录，入口用独占锁防止重复付费。
"""
import argparse
import csv
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from agent_system_pro import AgentState, MultiAgentOrchestrator, TaskDecompositionAgent
from experiment_support import atomic_json, digest, load_config, load_samples, MeasuredClient, usage_totals
from graph_retrieval import RetrievalConfig, local_embedder

EXPERIMENTS={'full_method','baseline1','baseline2','baseline3','no_agent2','no_agent4',
             'full_kg','topk','static_graph','no_relations','agent1_only'}
MODE={'full_kg':'full','topk':'topk','static_graph':'static_graph'}
BASELINE2='''你是{domain}的助教。自行判断问题类型并遵守教学规约：
理论概念可以完整说明；作业题只提供原理、必要的中间步骤和引导性问题，
保留学生需要完成的最终结果，不直接给出最终数值、判断、选项或完整代码。
用问题的语言回答。'''


def file_hash(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(config, mock):
    """绑定配置、输入文件和实验代码，防止改模型/方法后混进旧结果。"""
    paths=set()
    for ds in config['datasets']:
        paths.update(ds[k] for k in ['path','graph','graphs','labels'] if ds.get(k))
    sources=['agent_system_pro.py','graph_retrieval.py','run_experiments.py','experiment_support.py']
    hashes={str(p):file_hash(p) for p in sorted(paths)}
    hashes.update({name:file_hash(Path(__file__).with_name(name)) for name in sources})
    return digest(dict(config=config,mock=mock,files=hashes)),hashes


def experiment_output(client, ds, sample, exp, config, decomposition, embedder):
    """生成过程不读取sample.labels；所有方法共用pro中的教学智能体。"""
    question=sample['question']; domain=ds.get('domain','通用教学')
    # EntailmentBank统一作为提示任务时，该策略必须对所有受规约基线一致。
    policy='\n本实验将本题作为作业提示任务，请不要直接给最终答案。' if ds.get('intent_override')=='assignment_problem' else ''
    if exp in {'baseline1','baseline2'}:
        prompt='You are a helpful assistant.' if exp=='baseline1' else BASELINE2.format(domain=domain)+policy
        text=client.chat(prompt,question)
        return dict(final_response=text,intent_type=None,metrics={})
    if exp=='agent1_only':
        return dict(final_response='',intent_type=decomposition['intent_type'],keywords=decomposition['keywords'],metrics={})
    retrieval=RetrievalConfig(mode=MODE.get(exp,'graph'),budget=config['budget'],include_relations=exp!='no_relations')
    system=MultiAgentOrchestrator(client,graph_data=sample['graph'],retrieval_config=retrieval,embedder=embedder,domain=domain)
    # baseline3 = Agent1+Agent3；no_agent2保留审核；no_agent4保留检索。
    state=system.run_pipeline(question,max_chapter=ds.get('max_chapter',99),
        enable_pruning=exp not in {'baseline3','no_agent2'},
        enable_eval=exp not in {'baseline3','no_agent4'},max_retries=config['max_rounds'],
        intent_override=ds.get('intent_override'),decomposition_override=decomposition)
    result=asdict(state)
    # 没运行Agent4时，不把程序设置的“放行”当成审核通过结果。
    result['agent4_executed']=exp not in {'baseline3','no_agent4'}
    return result


def offline_metrics(output, labels, decomposition):
    """生成完成后才读取金标准；EntailmentBank无意图标签，不报告分类准确率。"""
    metrics={}
    if labels.get('intent') and decomposition:
        metrics['intent_correct']=int(decomposition['intent_type']==labels['intent'])
    retrieval=output.get('metrics',{}).get('retrieval')
    if retrieval and 'gold_supporting_facts' in labels:
        gold=set(labels['gold_supporting_facts']); pred=set(retrieval['selected_nodes'])
        hits=len(gold&pred); p=hits/len(pred) if pred else 0; r=hits/len(gold)
        metrics.update(evidence_precision=p,evidence_recall=r,evidence_f1=2*p*r/(p+r) if p+r else 0,
                       complete_recall=int(gold<=pred),selected_count=len(pred))
    return metrics


def summarize(out):
    """只汇总已保存的结果；错误不算作成功，也不进入质量指标分母。"""
    records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((out/'records').glob('*.json'))]
    rows=[]
    for key in sorted({(r['model_name'],r['dataset'],r['experiment']) for r in records}):
        group=[r for r in records if (r['model_name'],r['dataset'],r['experiment'])==key]
        ok=[r for r in group if r['status']=='success']
        row=dict(model=key[0],dataset=key[1],experiment=key[2],completed=len(group),success=len(ok),errors=len(group)-len(ok))
        row['mock']=any(r['mock'] for r in group)
        row['mean_job_seconds']=mean(r['elapsed_seconds'] for r in ok) if ok else None
        row['api_attempts_excluding_shared_a1']=sum(r['usage']['api_attempts'] for r in group)
        for name in ['intent_correct','evidence_precision','evidence_recall','evidence_f1','complete_recall','selected_count']:
            values=[r['offline_metrics'][name] for r in ok if name in r.get('offline_metrics',{})]
            row[name]=mean(values) if values else None
            row[name+'_n']=len(values)
        reviewed=[r for r in ok if r['output'].get('agent4_executed')]
        row['agent4_internal_pass_rate']=mean(bool(r['output']['eval_pass']) for r in reviewed) if reviewed else None
        rows.append(row)
    atomic_json(out/'summary.json',rows)
    if rows:
        with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # Agent1缓存的调用费用单独核算，不在每个消融组重复计费。
    caches=[json.loads(p.read_text(encoding='utf-8')) for p in (out/'agent1_cache').glob('*.json')]
    events=[e for r in records for e in r.get('api_events',[])]+[e for r in caches for e in r.get('api_events',[])]
    atomic_json(out/'usage.json',usage_totals(events))
    return rows


def run(args):
    config=load_config(args.config)
    experiments=args.experiments or config['experiments']
    if not set(experiments)<=EXPERIMENTS: raise ValueError('未知实验名称')
    models=[m for m in config['models'] if m.get('enabled',True) and (not args.models or m['name'] in args.models)]
    datasets=[d for d in config['datasets'] if not args.datasets or d['name'] in args.datasets]
    if not models or not datasets: raise ValueError('没有选中模型或数据集')
    if args.models and set(args.models)!={m['name'] for m in models}: raise ValueError('模型名称未知或尚未enabled')
    if args.datasets and set(args.datasets)!={d['name'] for d in datasets}: raise ValueError('未知数据集名称')
    if args.max_samples is not None and args.max_samples<1: raise ValueError('max-samples必须为正')
    data={d['name']:load_samples(d) for d in datasets}
    if args.max_samples:
        data={name:items[:args.max_samples] for name,items in data.items()}
    jobs=sum(len(x) for x in data.values())*len(models)*len(experiments)*config['repeats']
    print(json.dumps(dict(samples={k:len(v) for k,v in data.items()},models=[m['name'] for m in models],
        experiments=experiments,jobs=jobs,mock=args.mock),ensure_ascii=False,indent=2))
    if args.dry_run: return
    mock=args.mock
    # 模拟结果与正式结果始终分目录，不能混为论文结果。
    out=Path(config['output_dir']+('_mock' if mock else ''))
    out.mkdir(parents=True,exist_ok=True)
    marker,hashes=fingerprint(config,mock)
    manifest_path=out/'run_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8'))['fingerprint']!=marker:
            raise ValueError('配置、数据或代码已变更，请改output_dir创建新实验目录')
    # 先确认全部模型配置和向量模型可用，避免跑一半才发现缺密钥。
    clients={m['name']:MeasuredClient(m,config.get('request_interval_seconds',0),mock) for m in models}
    embedder=local_embedder(config.get('embedding_model'))
    from graph_retrieval import CommonGraphRetriever
    for samples in data.values(): CommonGraphRetriever(samples[0]['graph'],embedder=embedder)
    lock=out/'RUNNING.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('输出目录已有RUNNING.lock；确认旧进程已退出后再删除该锁文件')
    os.close(fd)
    try:
        if not manifest_path.exists(): atomic_json(manifest_path,dict(fingerprint=marker,config=config,mock=mock,input_hashes=hashes))
        done=0
        for model in models:
            client=clients[model['name']]
            for ds in datasets:
                for repeat in range(config['repeats']):
                    for sample in data[ds['name']]:
                        base=dict(model_name=model['name'],dataset=ds['name'],repeat=repeat,sample_id=sample['id'])
                        cache_path=out/'agent1_cache'/(digest(base)+'.json')
                        for exp in experiments:
                            job=dict(base,experiment=exp); job_id=digest(job)
                            target=out/'records'/(job_id+'.json')
                            old=json.loads(target.read_text(encoding='utf-8')) if target.exists() else None
                            if old and (old['status']=='success' or not args.retry_errors): continue
                            start=time.perf_counter(); event_start=len(client.events); decomposition=None
                            record=dict(job,job_id=job_id,mock=mock,model_id=model['model'],
                                question=sample['question'],labels=sample['labels'],status='error',
                                evaluation_intent=ds.get('intent_override') or sample['labels'].get('intent'))
                            try:
                                if exp not in {'baseline1','baseline2'}:
                                    if not cache_path.exists():
                                        state=TaskDecompositionAgent(client).run(AgentState(sample['question']))
                                        cached=dict(intent_type=state.intent_type,keywords=state.keywords)
                                        atomic_json(cache_path,dict(base,result=cached,api_events=client.events[event_start:]))
                                        # 缓存调用独立记录，避免计入每个实验组。
                                        event_start=len(client.events)
                                    decomposition=json.loads(cache_path.read_text(encoding='utf-8'))['result']
                                    record['shared_agent1_key']=cache_path.stem
                                output=experiment_output(client,ds,sample,exp,config,decomposition,embedder)
                                record.update(status='success',output=output,
                                    offline_metrics=offline_metrics(output,sample['labels'],decomposition))
                            except Exception as exc:
                                # MeasuredClient已去除服务端错误正文；本地异常也避免保存密钥。
                                error=str(exc)
                                if client.key: error=error.replace(client.key,'[REDACTED]')
                                record.update(error=error,error_type=type(exc).__name__)
                            record['elapsed_seconds']=time.perf_counter()-start
                            record['api_events']=(old.get('api_events',[]) if old else [])+client.events[event_start:]
                            record['usage']=usage_totals(record['api_events'])
                            record['attempt_number']=(old.get('attempt_number',0) if old else 0)+1
                            atomic_json(target,record)
                            done+=1
                            print(f"[{done}] {model['name']} / {ds['name']} / {exp} / {sample['id']}: {record['status']}",flush=True)
                            # 鉴权/配置错误不应连续扫完整个数据集。
                            if record['status']=='error' and any(code in record.get('error','') for code in ['HTTP 400','HTTP 401','HTTP 403','HTTP 404']):
                                raise RuntimeError('API配置或权限错误，已保存失败记录并停止；修正配置后使用新输出目录')
    finally:
        try:
            summarize(out)
        finally:
            lock.unlink(missing_ok=True)
    print(f'完成。结果与断点目录：{out}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(Path(__file__).with_name('experiment_config.json')))
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run',action='store_true'); mode.add_argument('--mock',action='store_true'); mode.add_argument('--live',action='store_true')
    p.add_argument('--datasets',nargs='+'); p.add_argument('--models',nargs='+')
    p.add_argument('--experiments',nargs='+',choices=sorted(EXPERIMENTS))
    p.add_argument('--max-samples',type=int); p.add_argument('--retry-errors',action='store_true')
    run(p.parse_args())


if __name__=='__main__': main()
