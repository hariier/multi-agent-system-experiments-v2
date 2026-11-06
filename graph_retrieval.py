"""两个图适配器与预算约束的贪心路径检索。

本模块不读取答案标签、不调用生成API。默认特征为词汇TF-IDF；可选本地
sentence-transformers模型启用语义特征。算法是贪心启发式，并非精确PCST。
"""
import heapq
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

STOP = set('a an the is are was were be been being of to in on at by for from and or if then that this these those as with it its into will can may what which how why does do did have has had most some than when where about one something'.split())


def tokens(text):
    text = re.sub(r'\[cite:\s*\d+\]', '', str(text).lower())
    words = [x for x in re.findall(r'[a-z0-9]+', text) if x not in STOP]
    for span in re.findall(r'[\u4e00-\u9fff]+', text):
        words.extend(span[i:i+2] for i in range(max(1, len(span)-1)))
    return words


class TextIndex:
    def __init__(self, texts, embedder=None):
        self.embedder = embedder
        counts = [Counter(tokens(t)) for t in texts]
        df = Counter(t for c in counts for t in c)
        self.idf = {t: math.log((1+len(texts))/(1+n))+1 for t, n in df.items()}
        self.vectors = [self.vector(t) for t in texts]
        self.dense = self._dense(texts) if embedder else None

    def vector(self, text):
        c = Counter(tokens(text))
        v = {t: (1+math.log(n))*self.idf[t] for t, n in c.items() if t in self.idf}
        norm = math.sqrt(sum(x*x for x in v.values())) or 1
        return {t: x/norm for t, x in v.items()}

    def _dense(self, texts):
        rows = self.embedder.encode(texts)
        result = []
        for row in rows:
            row = [float(x) for x in row]
            if not row or not all(math.isfinite(x) for x in row):
                raise ValueError('Invalid embedding')
            norm = math.sqrt(sum(x*x for x in row)) or 1
            result.append([x/norm for x in row])
        if len(result) != len(texts) or len({len(r) for r in result}) > 1:
            raise ValueError('Embedding shape mismatch')
        return result

    @staticmethod
    def dot(a, b):
        return sum(x*b.get(t, 0) for t, x in a.items())

    def pair(self, i, j, semantic=False):
        if semantic and self.dense is not None:
            return max(0.0, min(1.0, sum(a*b for a, b in zip(self.dense[i], self.dense[j]))))
        return self.dot(self.vectors[i], self.vectors[j])

    def query(self, text):
        lexical = [self.dot(self.vector(text), v) for v in self.vectors]
        if self.dense is None:
            return lexical
        q = self._dense([text])[0]
        return [0.3*l + 0.7*max(0.0, sum(a*b for a, b in zip(q, v)))
                for l, v in zip(lexical, self.dense)]


def local_embedder(model_path=None):
    if not model_path:
        return None
    # Explicit local directory only: never download a model implicitly.
    if not Path(model_path).is_dir():
        raise ValueError('Embedding model must be an existing local directory')
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(model_path), local_files_only=True)


def validate_graph(g):
    if g.get('schema_version') != 1:
        raise ValueError('Unsupported graph schema')
    ids = [n['id'] for n in g['nodes']]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('Graph requires nonempty, unique node IDs')
    for n in g['nodes']:
        if not isinstance(n['text'], str) or not n['text'].strip():
            raise ValueError('Empty node text')
    for e in g['edges']:
        if e['source'] not in ids or e['target'] not in ids or e['source'] == e['target']:
            raise ValueError('Invalid edge endpoint')
        if not 0 <= e['weight'] <= 1:
            raise ValueError('Invalid edge weight')
    return g


def add_similarity_edges(g, embedder=None, neighbors=3):
    nodes = [n for n in g['nodes'] if n['kind'] == 'content']
    index = TextIndex([n['text'] for n in nodes], embedder)
    # Mutual nearest neighbors, positive similarities only. No forced zero-score bridges.
    for semantic in ([False, True] if embedder else [False]):
        scores = {(i,j): index.pair(i,j,semantic) for i in range(len(nodes)) for j in range(i+1,len(nodes))}
        def score(i,j):
            return scores[tuple(sorted((i,j)))]
        near = {i: set(sorted((j for j in range(len(nodes)) if j != i and score(i,j)>0),
                              key=lambda j: (-score(i,j),nodes[j]['id']))[:neighbors]) for i in range(len(nodes))}
        for (i,j), w in scores.items():
            if j in near[i] and i in near[j]:
                g['edges'].append(dict(source=nodes[i]['id'], target=nodes[j]['id'],
                    relation='semantic_neighbor' if semantic else 'lexical_neighbor',
                    weight=round(w,8), directed=False))
    g['build_config'] = dict(neighbors=neighbors, feature_backend='local_embeddings+tfidf' if embedder else 'tfidf')
    return validate_graph(g)


class CourseGraphAdapter:
    """课程树→统一图。保留章节结构；列表先后顺序仅标为sequence。"""
    def build(self, source, embedder=None, neighbors=3):
        data = json.loads(Path(source).read_text(encoding='utf-8-sig')) if isinstance(source,(str,Path)) else source
        g = dict(schema_version=1, id='computer_network', dataset='course', query='', nodes=[], edges=[])
        def node(nid, text, kind, **meta):
            g['nodes'].append(dict(id=nid,text=str(text),kind=kind,**meta))
        def edge(a,b,rel,w):
            g['edges'].append(dict(source=a,target=b,relation=rel,weight=w,directed=True))
        for ch in data['knowledge_tree']:
            cid=ch['chapter_id']; cn=f'ch_{cid}'
            node(cn,ch['chapter_title'],'structure',chapter=cid)
            for ti,tp in enumerate(ch.get('topics',[])):
                tn=f'tp_{cid}_{ti}'
                node(tn,tp['topic'],'structure',chapter=cid)
                edge(cn,tn,'contains',0.2)
                previous=None
                for ki,text in enumerate(tp.get('knowledge_points',[])):
                    nid=f'kp_{cid}_{ti}_{ki}'
                    node(nid,text,'content',chapter=cid,topic_id=tn,topic_name=tp['topic'],aliases=tp.get('aliases',[]))
                    edge(tn,nid,'contains',0.4)
                    if previous:
                        edge(previous,nid,'sequence',0.25)
                    previous=nid
        return add_similarity_edges(g,embedder,neighbors)


class EntailmentBankAdapter:
    """一题一图，严格按白名单读取候选事实；标签在labels函数单独处理。"""
    def build(self, sample, embedder=None, neighbors=3):
        # Whitelist input fields. In particular, do not copy meta wholesale.
        facts=sample['meta']['triples']
        g=dict(schema_version=1,id=sample['id'],dataset='entailmentbank',query=sample['question'],
               nodes=[dict(id=k,text=v,kind='content') for k,v in facts.items()],edges=[])
        return add_similarity_edges(g,embedder,neighbors)

    @staticmethod
    def labels(sample):
        gold=sorted(set(re.findall(r'\bsent\d+\b',sample['proof'])))
        if not gold or not set(gold).issubset(sample['meta']['triples']):
            raise ValueError(f"Invalid proof leaves: {sample['id']}")
        return dict(id=sample['id'],gold_supporting_facts=gold,answer=sample['answer'],
                    proof_depth=sample['depth_of_proof'])


@dataclass
class RetrievalConfig:
    mode: str = 'graph'  # graph / static_graph / topk / full / none
    budget: int = 8      # hard limit on content nodes; structure nodes reported separately
    edge_penalty: float = 0.15
    include_relations: bool = True

    def __post_init__(self):
        if self.mode not in {'graph','static_graph','topk','full','none'}:
            raise ValueError('Unknown retrieval mode')
        if self.budget < 1 or self.edge_penalty < 0:
            raise ValueError('Invalid retrieval budget or penalty')


class CommonGraphRetriever:
    """两个数据集共用的检索器；更改config.mode即可做受控图消融。"""
    def __init__(self, graph, config=None, embedder=None):
        self.graph=validate_graph(graph)
        backend=graph.get('build_config',{}).get('feature_backend')
        if backend and ('embeddings' in backend) != bool(embedder):
            raise ValueError('Graph construction and retrieval must use the same feature backend')
        self.config=config or RetrievalConfig()
        self.nodes={n['id']:n for n in graph['nodes']}
        self.ids=list(self.nodes)
        texts=[n['text']+' '+n.get('topic_name','')+' '+' '.join(n.get('aliases',[])) for n in graph['nodes']]
        self.index=TextIndex(texts,embedder)

    def retrieve(self, query, keywords=None, max_chapter=None):
        cfg=self.config
        # 先过滤课程章节，再做评分和连通路径搜索；科学事实图没有章节限制。
        active={nid for nid,n in self.nodes.items() if max_chapter is None or n.get('chapter') is None or n['chapter']<=max_chapter}
        content={nid for nid in active if self.nodes[nid]['kind']=='content'}
        q=query+' '+' '.join(str(x) for x in (keywords or []))
        scores=dict(zip(self.ids,self.index.query(q)))
        ranked=sorted(content,key=lambda n:(-scores[n],n))
        adj={n:[] for n in active}
        for e in self.graph['edges']:
            a,b=e['source'],e['target']
            if a not in active or b not in active:
                continue
            confidence=e['weight']
            if cfg.mode != 'static_graph':
                confidence=0.5*confidence+0.25*(scores[a]+scores[b])
            cost=max(0.01,1-confidence)
            # Connectivity uses an undirected projection. Original direction is
            # preserved in returned edges and never relabeled as logical entailment.
            adj[a].append((b,cost)); adj[b].append((a,cost))

        # The following diagnostics are observational only and do not affect selection.
        selected=set(); seeds=[]; stop_reason=None; iterations=0
        last_reachable_candidates=0
        last_budget_feasible_candidates=0
        last_positive_utility_candidates=0
        last_best_utility=None

        if cfg.mode=='full':
            selected=set(active)
            stop_reason='full_mode'
        elif cfg.mode=='topk':
            selected=set(ranked[:cfg.budget])
            stop_reason='topk_mode'
        elif cfg.mode in {'graph','static_graph'}:
            if not ranked:
                stop_reason='no_content'
            elif scores[ranked[0]]<=0:
                stop_reason='no_positive_seed'
            else:
                seeds=[ranked[0]]; selected=set(seeds)
                while len(selected & content)<cfg.budget:
                    iterations+=1
                    # 从已选连通子图出发寻找最短扩展路径；结构节点用于连接，
                    # 预算只计算内容节点。边成本与内容奖励共同决定是否继续扩展。
                    dist={n:0.0 for n in selected}; paths={n:[n] for n in selected}
                    queue=[(0.0,n) for n in sorted(selected)]; heapq.heapify(queue)
                    while queue:
                        d,u=heapq.heappop(queue)
                        if d!=dist[u]: continue
                        for v,c in sorted(adj[u]):
                            nd=d+c
                            if nd < dist.get(v,float('inf')):
                                dist[v]=nd; paths[v]=paths[u]+[v]; heapq.heappush(queue,(nd,v))

                    best=None
                    reachable_candidates=0
                    budget_feasible_candidates=0
                    positive_utility_candidates=0
                    for target in ranked:
                        if target in selected or target not in paths: continue
                        reachable_candidates+=1
                        addition=set(paths[target])-selected
                        added_content=addition & content
                        if len((selected|addition)&content)>cfg.budget: continue
                        budget_feasible_candidates+=1
                        gain=sum(scores[n] for n in added_content)-cfg.edge_penalty*dist[target]
                        utility=gain/max(1,len(added_content))
                        candidate=(utility,target,addition)
                        if utility>0:
                            positive_utility_candidates+=1
                            if best is None or utility>best[0]:
                                best=candidate

                    last_reachable_candidates=reachable_candidates
                    last_budget_feasible_candidates=budget_feasible_candidates
                    last_positive_utility_candidates=positive_utility_candidates

                    if best is None:
                        if reachable_candidates==0:
                            stop_reason='no_reachable_candidate'
                        elif budget_feasible_candidates==0:
                            stop_reason='budget_blocked'
                        else:
                            stop_reason='no_positive_gain'
                        break

                    last_best_utility=float(best[0])
                    selected.update(best[2])

                if len(selected & content)>=cfg.budget:
                    stop_reason='budget_reached'
                elif stop_reason is None:
                    stop_reason='stopped'
        else:
            stop_reason='none_mode'

        # Connectivity upper bound from the single seed. This is label-free and
        # later intersected with gold support only in the evaluation script.
        seed_component=set()
        if seeds:
            stack=[seeds[0]]
            seed_component={seeds[0]}
            while stack:
                u=stack.pop()
                for v,_ in adj.get(u,[]):
                    if v not in seed_component:
                        seed_component.add(v)
                        stack.append(v)
        seed_component_content=sorted(seed_component & content)

        selected_content=sorted(selected & content,key=lambda n:(-scores[n],n))
        edges=[dict(e) for e in self.graph['edges'] if e['source'] in selected and e['target'] in selected]
        lines=['【检索参考资料 / Retrieved reference facts】']
        lines.extend(f"[{nid}] {self.nodes[nid]['text']}" for nid in selected_content)
        structural=sorted(selected-content)
        if cfg.include_relations and edges:
            lines.append('【结构节点 / Structural labels】')
            lines.extend(f"[{nid}] {self.nodes[nid]['text']}" for nid in structural)
            lines.append('【关联关系；相似/顺序不代表逻辑蕴含 / Relations, not entailment claims】')
            lines.extend(f"{e['source']} {'->' if e['directed'] else '--'} {e['target']} ({e['relation']})" for e in edges)
        if not selected_content: lines.append('未检索到相关事实 / No relevant facts retrieved.')
        context='\n'.join(lines)
        return dict(id=self.graph['id'],seed_nodes=seeds,selected_nodes=selected_content,
                    structural_nodes=structural,selected_edges=edges,context=context,
                    node_scores={n:scores[n] for n in selected_content},
                    selected_topics=sorted({self.nodes[n]['topic_id'] for n in selected_content if 'topic_id' in self.nodes[n]}),
                    config=asdict(cfg),feature_backend=self.graph.get('build_config',{}).get('feature_backend','unknown'),
                    diagnostics=dict(
                        content_count=len(selected_content),
                        structure_count=len(structural),
                        edge_count=len(edges),
                        context_characters=len(context),
                        budget_applied=cfg.mode not in {'full','none'},
                        stop_reason=stop_reason,
                        iterations=iterations,
                        seed_id=seeds[0] if seeds else None,
                        seed_score=float(scores[seeds[0]]) if seeds else None,
                        seed_component_content_count=len(seed_component_content),
                        seed_component_content_nodes=seed_component_content,
                        positive_score_content_count=sum(1 for n in content if scores[n]>0),
                        last_reachable_candidates=last_reachable_candidates,
                        last_budget_feasible_candidates=last_budget_feasible_candidates,
                        last_positive_utility_candidates=last_positive_utility_candidates,
                        last_best_utility=last_best_utility))


def apply_retrieval_to_state(state, result):
    """统一写回AgentState，保证批量实验保存的节点和LLM上下文来自同一结果。"""
    state.matched_nodes=result['seed_nodes']
    state.selected_nodes=result['selected_nodes']
    state.selected_edges=result['selected_edges']
    state.selected_topics=result['selected_topics']
    state.pruned_subgraph_context=result['context']
    state.retrieval_score=max(result['node_scores'].values(),default=0.0)
    state.retrieval_stage=result['config']['mode']
    state.retrieval_fallback_used=False
    state.metrics['retrieval']=result
    return state
