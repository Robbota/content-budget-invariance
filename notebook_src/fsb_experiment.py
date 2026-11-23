# %% [markdown]
# # Content-Budget Invariance in Multilingual Dense Retrieval: experiment notebook
#
# Paper 2 (MSc CS, MCS 502), Robert Bota, R179595Q.
#
# This notebook runs the whole experiment described in the proposal (Section 3):
#
# 1. **Phase 0**: token premiums (pi_L) on FLORES-200 for every encoder tokenizer, the language swap rule, and Shona's predicted budgets.
# 2. **Data**: TyDi QA primary-task dev questions with at least 2 non-null passage annotations; one corpus of N articles per language (gold articles plus seeded distractors from the training split).
# 3. **Runs**: every chunking policy (B1 fixed-token sweep, FSB content sweep, B2 sliding window, B3 native passages, B4 fixed-character, FSB-W, FSB-S) for each encoder, with exact search (FAISS IndexFlatIP). Each finished run is saved to Google Drive, so a disconnect loses at most one run: just re-run the notebook and it skips finished work.
# 4. **Analysis**: H1 (mixed-effects model), H2 (bootstrap ranking of K*_L against pi_L), H3 (bootstrap TOST on C*_L / C*_en), RQ3 (paired randomization tests, Holm-Bonferroni), robustness checks, tables and figures.
#
# **How to run on Colab**
#
# 1. Runtime > Change runtime type > **T4 GPU**.
# 2. Run the first time with `SMOKE = True` (about 10 minutes). Check that every cell finishes.
# 3. Set `SMOKE = False` and run all cells. If the session ends, reconnect and run all cells again; finished runs are skipped.
# 4. When the analysis cells finish, download the folder `MyDrive/fsb_experiment/<RUN_NAME>/analysis` (and `runs_summary.csv`) and bring it back for writing the Results section.

# %% [markdown]
# ## 0. Configuration
# Everything that can change is set in this cell. The values below are the pre-registered design unless a comment says otherwise.

# %%
import os, sys, json, time, math, gzip, re, glob, pickle, random, platform, tarfile, urllib.request
from pathlib import Path

SMOKE = False            # True: tiny run to check the notebook end to end (minutes, not hours)
RUN_NAME = 'main'        # results go to <WORK_ROOT>/<RUN_NAME>; smoke runs go to '<RUN_NAME>_smoke'

# Languages (TyDi QA names). English is the reference. Swahili and Telugu are fixed by design decision 11.
LANGS = ['english', 'indonesian', 'finnish', 'swahili', 'telugu']
FIXED_LANGS = ['english', 'swahili', 'telugu']
SWAP_CANDIDATES = ['arabic', 'bengali', 'russian']
SWAP_GAP = 0.15           # Phase 0 rule: two chosen languages with premiums closer than this trigger a swap
APPLY_SWAP_RULE = True    # if no candidate satisfies the rule, the original language is kept and a warning is printed
MIN_QUESTIONS = 300       # a swap candidate needs at least this many usable dev questions
PREDICT_ONLY = ['shona']  # premium measured, budgets predicted, no retrieval data

# Encoders: the first is primary. 'me5-base' replicates only on budgets that fit its 512-token window.
# 'qwen3-0.6b' is OPTIONAL (not in the proposal): a different tokenizer family with a much wider premium range.
ENCODERS = ['bge-m3', 'me5-base']
MAX_PRIORITY = {}        # e.g. {'qwen3-0.6b': 1} runs only the H1-H3 sweeps for that encoder
RUN_EXPERIMENTS = True   # False: skip Section 9 and only analyse the runs already saved (no GPU needed)

CONTENT_GRID = [32, 48, 64, 96, 128, 192, 256, 384]   # English-equivalent words (design decision 4)
RQ3_K = [128, 256, 512]                                # reference token budgets for the baseline comparison
K_MIN = 32                                             # FSB floor (tokens)
TOKEN_GRID = None       # B1 sweep; None = geometric 32, 48, 64, 96, ... up to the largest FSB budget
N_ARTICLES = 'auto'     # same N for every language; 'auto' = largest gold-article count rounded up to the next 100
Q_MAX = None            # cap on questions per language (None = all usable questions)
AR_W = [128, 256]       # answer-recall control: reading budgets in English-equivalent words
TOPK = 100              # chunks retrieved per query
N_BOOT = 10000
N_PERM = 10000
EQ_BOUNDS = (0.67, 1.5) # H3 equivalence bounds on C*_L / C*_en
ALPHA = 0.05
SEED = 2026

if SMOKE:
    RUN_NAME = RUN_NAME + '_smoke'
    Q_MAX, N_ARTICLES = 25, 40
    CONTENT_GRID = [32, 96, 256]
    TOKEN_GRID = [32, 128, 512]
    RQ3_K = [128]
    N_BOOT, N_PERM = 300, 300

IN_COLAB = 'google.colab' in sys.modules
IN_KAGGLE = os.path.exists('/kaggle/working')
LOCAL_TEST = os.environ.get('FSB_LOCAL_TEST') == '1'   # used only to test this notebook without a GPU
if LOCAL_TEST:
    exec(open(os.environ['FSB_LOCAL_CONFIG']).read())

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    WORK_ROOT = Path('/content/drive/MyDrive/fsb_experiment')
    CACHE = Path('/content/cache')          # big downloads stay on the VM disk, not on Drive
elif IN_KAGGLE:
    WORK_ROOT = Path('/kaggle/working/fsb_experiment'); CACHE = Path('/kaggle/tmp/cache')
elif not LOCAL_TEST:
    WORK_ROOT = Path('fsb_experiment'); CACHE = Path('fsb_cache')
WORK = WORK_ROOT / RUN_NAME
for p in [WORK, CACHE, WORK / 'data', WORK / 'runs', WORK / 'analysis']:
    p.mkdir(parents=True, exist_ok=True)
print('Results folder:', WORK)

# %% [markdown]
# ## 1. Install and import

# %%
if IN_COLAB:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'faiss-cpu'], check=False)
import numpy as np, pandas as pd
import pyarrow.parquet as pq
try:
    import faiss
except ImportError:
    faiss = None
    print('faiss not installed: using exact numpy search (same results, slower)')
from transformers import AutoTokenizer
import transformers
try:
    import torch
except ImportError:
    torch = None
transformers.logging.set_verbosity_error()

ENV = dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
           transformers=transformers.__version__, torch=getattr(torch, '__version__', None),
           faiss=getattr(faiss, '__version__', None), platform=platform.platform(),
           gpu=(torch.cuda.get_device_name(0) if torch is not None and torch.cuda.is_available() else None),
           started=time.strftime('%Y-%m-%d %H:%M:%S'))
print(ENV)
DEVICE = 'cuda' if torch is not None and torch.cuda.is_available() else 'cpu'
if not LOCAL_TEST and DEVICE != 'cuda':
    print('WARNING: no GPU found. Runtime > Change runtime type > T4 GPU.')

# %% [markdown]
# ## 2. Encoders
# Pooling and prefixes follow each model card. Budgets count content tokens only (no special tokens, no prefix).

# %%
ENCODER_SPECS = {
    'bge-m3':     dict(hf='BAAI/bge-m3', max_len=8192, pooling='cls', q_prefix='', d_prefix='',
                       padding_side='right', batch_tokens=32768, dtype='fp16'),
    'me5-base':   dict(hf='intfloat/multilingual-e5-base', max_len=512, pooling='mean',
                       q_prefix='query: ', d_prefix='passage: ', padding_side='right', batch_tokens=65536, dtype='fp16'),
    'qwen3-0.6b': dict(hf='Qwen/Qwen3-Embedding-0.6B', max_len=8192, pooling='last', append_eos=True, eos_token='<|endoftext|>',
                       q_prefix='Instruct: Given a question, retrieve Wikipedia passages that answer the question\nQuery:',
                       d_prefix='', padding_side='left', batch_tokens=16384, dtype='fp16', check_fp16=True),
}
if LOCAL_TEST:
    ENCODER_SPECS.update(LOCAL_ENCODER_SPECS)

_TOK = {}
def get_tokenizer(enc):
    hf = ENCODER_SPECS[enc]['hf']
    if hf not in _TOK:
        t = AutoTokenizer.from_pretrained(hf)
        t.model_max_length = 10**9          # we cut by ourselves; silences length warnings
        _TOK[hf] = t
    return _TOK[hf]

def content_kmax(enc):
    """Largest number of content tokens a chunk may have for this encoder."""
    s = ENCODER_SPECS[enc]; tok = get_tokenizer(enc)
    n_special = len(tok('a', add_special_tokens=True)['input_ids']) - len(tok('a', add_special_tokens=False)['input_ids'])
    n_prefix = len(tok(s['d_prefix'], add_special_tokens=False)['input_ids']) if s['d_prefix'] else 0
    return s['max_len'] - n_special - n_prefix - (1 if s.get('append_eos') else 0)

# %% [markdown]
# ## 3. Phase 0: token premiums on FLORES-200
# pi_L = tokens of the FLORES-200 dev sentences in language L / tokens of the same sentences in English (Petrov et al., 2023).
# t_en = English tokens per whitespace word, used to convert English-equivalent words to tokens.

# %%
FLORES = dict(english='eng_Latn', indonesian='ind_Latn', finnish='fin_Latn', swahili='swh_Latn', telugu='tel_Telu',
              arabic='arb_Arab', bengali='ben_Beng', russian='rus_Cyrl', japanese='jpn_Jpan', korean='kor_Hang',
              thai='tha_Thai', shona='sna_Latn')
FLORES_URL = 'https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz'

def flores_dir():
    d = CACHE / 'flores200_dataset'
    if not (d / 'dev').exists():
        tgz = CACHE / 'flores200_dataset.tar.gz'
        if not tgz.exists():
            urllib.request.urlretrieve(FLORES_URL, tgz)
        with tarfile.open(tgz) as tf:
            tf.extractall(CACHE)
    return d / 'dev'

def flores_sentences(lang):
    return (flores_dir() / f'{FLORES[lang]}.dev').read_text(encoding='utf-8').splitlines()

def premiums(enc, langs):
    tok = get_tokenizer(enc)
    count = lambda ss: sum(len(x) for x in tok(ss, add_special_tokens=False)['input_ids'])
    en = flores_sentences('english'); n_en = count(en)
    out = {}
    for l in langs:
        s = flores_sentences(l); n = count(s)
        out[l] = dict(pi=n / n_en, tokens=n, words=sum(len(x.split()) for x in s), chars=sum(len(x) for x in s))
    t_en = n_en / sum(len(x.split()) for x in en)
    cpt_en = sum(len(x) for x in en) / n_en
    return out, t_en, cpt_en

ALL_PHASE0 = sorted(set(LANGS) | set(SWAP_CANDIDATES) | set(PREDICT_ONLY) | {'english'})
PREM, T_EN, CPT_EN = {}, {}, {}
for enc in ENCODERS:
    PREM[enc], T_EN[enc], CPT_EN[enc] = premiums(enc, ALL_PHASE0)
prem_table = pd.DataFrame({enc: {l: round(v['pi'], 3) for l, v in PREM[enc].items()} for enc in ENCODERS})
print('Token premium pi_L (FLORES-200 dev, 997 sentences):'); print(prem_table)
print('English tokens per word:', {e: round(v, 3) for e, v in T_EN.items()})

# %% [markdown]
# ### Phase 0 swap rule (design decision 11)
# If two chosen languages have premiums within 0.15 (primary encoder), a non-fixed language is replaced by Arabic, Bengali or Russian, provided the candidate is at least 0.15 from every other chosen language and has enough questions. If no candidate qualifies, the original language is kept and the problem is reported.

# %%
def usable_question_counts(langs):
    df = load_dev()
    return df[df.language.isin(langs)].groupby('language').size().to_dict()

def apply_swap_rule(langs, pi, counts):
    langs = list(langs); log = []
    def clashes(ls):
        return [(a, b) for i, a in enumerate(ls) for b in ls[i+1:] if abs(pi[a] - pi[b]) < SWAP_GAP]
    for a, b in clashes(langs):
        if a not in langs or b not in langs or abs(pi[a] - pi[b]) >= SWAP_GAP:
            continue
        movable = [x for x in (a, b) if x not in FIXED_LANGS]
        log.append(f'clash: {a} ({pi[a]:.3f}) and {b} ({pi[b]:.3f}), gap {abs(pi[a]-pi[b]):.3f} < {SWAP_GAP}')
        done = False
        for m in movable:
            others = [x for x in langs if x != m]
            for c in SWAP_CANDIDATES:
                if c in langs: continue
                if counts.get(c, 0) < MIN_QUESTIONS:
                    log.append(f'  candidate {c}: only {counts.get(c, 0)} usable questions'); continue
                gaps = {o: abs(pi[c] - pi[o]) for o in others}
                if min(gaps.values()) >= SWAP_GAP:
                    langs[langs.index(m)] = c; log.append(f'  -> replaced {m} with {c} (pi {pi[c]:.3f})'); done = True; break
                worst = min(gaps, key=gaps.get)
                log.append(f'  candidate {c} (pi {pi[c]:.3f}) rejected: within {gaps[worst]:.3f} of {worst}')
            if done: break
        if not done:
            log.append(f'  -> NO candidate satisfies the rule; kept {a} and {b}')
    return langs, log

# %% [markdown]
# ## 4. Data: TyDi QA questions and equal-N corpora

# %%
TYDI_REPO = 'google-research-datasets/tydiqa'
DEV_FILE = 'primary_task/validation-00000-of-00001.parquet'
TRAIN_FILES = [f'primary_task/train-{i:05d}-of-00012.parquet' for i in range(12)]
COLS = ['language', 'question_text', 'document_title', 'document_url', 'document_plaintext',
        'passage_answer_candidates', 'annotations']

def tydi_file(fname):
    if LOCAL_TEST and (Path(LOCAL_TYDI) / fname).exists():
        return str(Path(LOCAL_TYDI) / fname)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(TYDI_REPO, fname, repo_type='dataset', cache_dir=str(CACHE / 'hf'))

def byte_to_char(text):
    """Array mapping each UTF-8 byte offset (0..len(bytes)) to a character offset."""
    lens = np.fromiter((len(ch.encode('utf-8')) for ch in text), dtype=np.int64, count=len(text))
    starts = np.concatenate([[0], np.cumsum(lens)])          # byte offset where each char starts (+ end)
    nbytes = starts[-1]
    return np.searchsorted(starts, np.arange(nbytes + 1), side='right') - 1, starts

def spans_to_chars(text, sb, eb):
    b2c, starts = byte_to_char(text)
    sb = np.asarray(sb, dtype=np.int64); eb = np.asarray(eb, dtype=np.int64)
    s = b2c[sb]; e = b2c[eb]
    # an end byte that falls inside a character rounds up to the end of that character
    e = np.where(starts[e] < eb, e + 1, e)
    return np.stack([s, e], axis=1).astype(np.int64)

_DEV = None
def load_dev():
    """All dev questions with >= 2 non-null passage annotations (design decision 9)."""
    global _DEV
    if _DEV is not None: return _DEV
    t = pq.read_table(tydi_file(DEV_FILE), columns=['language', 'question_text', 'document_url', 'annotations'])
    rows = []
    for i, r in enumerate(t.to_pylist()):
        a = r['annotations']; idx = [x for x in a['passage_answer_candidate_index'] if x >= 0]
        if len(idx) >= 2:
            mins = sorted({(s, e) for s, e in zip(a['minimal_answers_start_byte'], a['minimal_answers_end_byte']) if s >= 0 and e > s})
            rows.append(dict(row=i, language=r['language'], question=r['question_text'], url=r['document_url'],
                             gold=sorted(set(idx)), min_bytes=mins))
    _DEV = pd.DataFrame(rows)
    return _DEV

def read_docs(path, langs, urls=None):
    """Stream a TyDi parquet file and keep one copy of each article (by URL) in the given languages."""
    out = {}
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=['language', 'document_title', 'document_url', 'document_plaintext', 'passage_answer_candidates'])
        lang = t.column('language').to_pylist(); url = t.column('document_url').to_pylist()
        keep = [i for i in range(len(lang)) if lang[i] in langs and url[i] not in out and (urls is None or url[i] in urls)]
        if not keep: continue
        sub = t.take(keep).to_pylist()
        for r in sub:
            if r['document_url'] in out: continue
            pc = r['passage_answer_candidates']
            out[r['document_url']] = dict(lang=r['language'], url=r['document_url'], title=r['document_title'],
                                          text=r['document_plaintext'], sb=pc['plaintext_start_byte'], eb=pc['plaintext_end_byte'])
        del t
    return out

def build_corpora(langs):
    path = WORK / 'data' / 'corpora.pkl'
    if path.exists():
        # safe: this pickle is the notebook's own cache file in the student's Drive folder, written below
        C = pickle.load(open(path, 'rb'))
        if C['langs'] == langs: return C
    rng = np.random.default_rng(SEED)
    dev = load_dev(); dev = dev[dev.language.isin(langs)].copy()
    if Q_MAX:
        dev = pd.concat([g.sample(n=min(Q_MAX, len(g)), random_state=SEED) for _, g in dev.groupby('language')])
    gold_docs = read_docs(tydi_file(DEV_FILE), set(langs), urls=set(dev.url))
    all_dev_urls = set(pq.read_table(tydi_file(DEV_FILE), columns=['document_url']).column(0).to_pylist())
    n_gold = dev.groupby('language').url.nunique().to_dict()
    N = N_ARTICLES if N_ARTICLES != 'auto' else int(math.ceil(max(n_gold.values()) / 100.0) * 100)
    print('gold articles per language:', n_gold, '-> N =', N)
    assert N >= max(n_gold.values())
    need = {l: N - n_gold[l] for l in langs}
    pools = {l: {} for l in langs}
    for f in TRAIN_FILES:
        if all(len(pools[l]) >= 3 * need[l] + 50 for l in langs): break
        docs = read_docs(tydi_file(f), {l for l in langs if len(pools[l]) < 3 * need[l] + 50})
        for u, d in docs.items():
            if u not in all_dev_urls and len(d['sb']) >= 1 and len(d['text'].split()) >= 50:
                pools[d['lang']].setdefault(u, d)
        print('distractor pool sizes after', f.split('/')[-1], {l: len(p) for l, p in pools.items()})
    corpora, questions = {}, {}
    for l in langs:
        pool_urls = sorted(pools[l]); assert len(pool_urls) >= need[l], f'not enough distractors for {l}'
        pick = rng.choice(len(pool_urls), size=need[l], replace=False)
        docs = [gold_docs[u] for u in sorted(set(dev[dev.language == l].url))] + [pools[l][pool_urls[i]] for i in sorted(pick)]
        out = []
        for j, d in enumerate(docs):
            P = spans_to_chars(d['text'], d['sb'], d['eb'])
            out.append(dict(doc=j, url=d['url'], title=d['title'], text=d['text'], passages=P, gold_article=j < n_gold[l]))
        url2doc = {d['url']: d['doc'] for d in out}
        qs = []
        for k, r in enumerate(dev[dev.language == l].itertuples()):
            d = out[url2doc[r.url]]
            mins = spans_to_chars(d['text'], [s for s, _ in r.min_bytes], [e for _, e in r.min_bytes]) if r.min_bytes else np.zeros((0, 2), np.int64)
            qs.append(dict(qid=f'{l}-{k:04d}', lang=l, text=r.question, doc=d['doc'], gold=set(r.gold), minimal=mins))
        corpora[l], questions[l] = out, qs
    C = dict(langs=langs, N=N, corpora=corpora, questions=questions)
    pickle.dump(C, open(path, 'wb'))
    return C

# Phase 0 decision, then the corpora
counts = usable_question_counts(ALL_PHASE0)
primary = ENCODERS[0]
PI0 = {l: v['pi'] for l, v in PREM[primary].items()}
if APPLY_SWAP_RULE:
    LANGS_FINAL, swap_log = apply_swap_rule(LANGS, PI0, counts)
else:
    LANGS_FINAL, swap_log = list(LANGS), ['swap rule switched off']
print('\n'.join(swap_log) or 'no clashes'); print('Languages used:', LANGS_FINAL)
for enc in ENCODERS[1:]:
    pi = {l: v['pi'] for l, v in PREM[enc].items()}
    _, lg = apply_swap_rule(LANGS_FINAL, pi, counts)
    print(f'[{enc}] clash check on the final set:', '; '.join(x for x in lg if x.startswith('clash')) or 'none')
json.dump(dict(premiums=PREM, t_en=T_EN, cpt_en=CPT_EN, usable_questions=counts, swap_log=swap_log,
               langs_final=LANGS_FINAL, env=ENV), open(WORK / 'analysis' / 'phase0.json', 'w'), indent=1, default=str)

DATA = build_corpora(LANGS_FINAL)
for l in LANGS_FINAL:
    docs = DATA['corpora'][l]
    print(f"{l:11s} questions={len(DATA['questions'][l]):5d} articles={len(docs)} (gold {sum(d['gold_article'] for d in docs)}) "
          f"words={sum(len(d['text'].split()) for d in docs):,}")

# %% [markdown]
# ## 5. Chunking policies
# All chunks are character spans of the original article, so every chunk maps back to TyDi passages and answers.

# %%
_DOCTOK = {}
def doc_tokens(enc, lang):
    """Token start/end character offsets for every article (content tokens only)."""
    hf = ENCODER_SPECS[enc]['hf']
    key = (hf, lang)
    if key not in _DOCTOK:
        tok = get_tokenizer(enc); docs = DATA['corpora'][lang]
        res = []
        for i in range(0, len(docs), 64):
            batch = tok([d['text'] for d in docs[i:i+64]], add_special_tokens=False, return_offsets_mapping=True)
            for off in batch['offset_mapping']:
                a = np.asarray(off, dtype=np.int64).reshape(-1, 2).copy()
                # XLM-R gives a standalone '▁' token the span of the next word's first character;
                # clip every token end to the next token's start so chunk boundaries never cut a word
                if len(a) > 1:
                    a[:-1, 1] = np.maximum(a[:-1, 0], np.minimum(a[:-1, 1], a[1:, 0]))
                res.append(a)
        _DOCTOK[key] = res
    return _DOCTOK[key]

def chunks_fixed_token(offs, K, stride=None):
    stride = stride or K; out = []
    for d, a in enumerate(offs):
        n = len(a)
        if n == 0: continue
        starts, s = [], 0                                      # windows of K tokens every `stride` tokens;
        while True:                                            # the last window is shorter if the article ends
            starts.append(s)
            if s + K >= n: break
            s += stride
        for s in starts:
            e = min(s + K, n); out.append((d, a[s, 0], a[e - 1, 1], e - s))
    return out

def chunks_fixed_char(docs, offs, nchar):
    out = []
    for d, doc in enumerate(docs):
        L = len(doc['text']); a = offs[d]; st = a[:, 0] if len(a) else np.zeros(0, np.int64)
        for s in range(0, L, nchar):
            e = min(s + nchar, L)
            ntok = int(np.searchsorted(st, e, 'left') - np.searchsorted(st, s, 'left'))
            if ntok > 0: out.append((d, s, e, ntok))
    return out

def chunks_native(docs, offs):
    out = []
    for d, doc in enumerate(docs):
        st = offs[d][:, 0] if len(offs[d]) else np.zeros(0, np.int64)
        for s, e in doc['passages']:
            ntok = int(np.searchsorted(st, e, 'left') - np.searchsorted(st, s, 'left'))
            if ntok > 0: out.append((d, int(s), int(e), ntok))
    return out

SENT_END = re.compile(r'(?<=[.!?।॥。！？])\s+|\n+')
def sentence_spans(text):
    spans, s = [], 0
    for m in SENT_END.finditer(text):
        if m.start() > s: spans.append((s, m.start()))
        s = m.end()
    if s < len(text): spans.append((s, len(text)))
    return spans

def chunks_sentence(docs, offs, K):
    """FSB-S: pack whole sentences up to K tokens; a sentence longer than K is cut by tokens."""
    out = []
    for d, doc in enumerate(docs):
        a = offs[d]
        if len(a) == 0: continue
        st = a[:, 0]; cur = None                               # cur = [start_tok, end_tok)
        for s, e in sentence_spans(doc['text']):
            t0, t1 = int(np.searchsorted(st, s, 'left')), int(np.searchsorted(st, e, 'left'))
            if t1 <= t0: continue
            if cur and t1 - cur[0] <= K:
                cur[1] = t1; continue
            if cur: out.append((d, a[cur[0], 0], a[cur[1]-1, 1], cur[1] - cur[0]))
            while t1 - t0 > K:
                out.append((d, a[t0, 0], a[t0+K-1, 1], K)); t0 += K
            cur = [t0, t1]
        if cur: out.append((d, a[cur[0], 0], a[cur[1]-1, 1], cur[1] - cur[0]))
    return out

def word_fertility(enc, lang):
    offs = doc_tokens(enc, lang); docs = DATA['corpora'][lang]
    return sum(len(a) for a in offs) / sum(len(d['text'].split()) for d in docs)

def budget_fsb(enc, lang, k_ref):
    return int(min(max(round(k_ref * PREM[enc][lang]['pi']), K_MIN), content_kmax(enc)))

def make_chunks(enc, lang, policy, param):
    docs = DATA['corpora'][lang]; offs = doc_tokens(enc, lang)
    if policy == 'B1':   K = int(param); ch = chunks_fixed_token(offs, K)
    elif policy == 'B2': K = int(param); ch = chunks_fixed_token(offs, K, stride=K // 2)
    elif policy == 'B3': K = None; ch = chunks_native(docs, offs)
    elif policy == 'B4': K = int(round(param * CPT_EN[enc])); ch = chunks_fixed_char(docs, offs, K)
    elif policy in ('FSB', 'FSBC'):   # FSB at a reference token budget (FSB) or at a content budget in words (FSBC)
        k_ref = param if policy == 'FSB' else param * T_EN[enc]
        K = budget_fsb(enc, lang, k_ref); ch = chunks_fixed_token(offs, K)
    elif policy == 'FSBW':
        phi = word_fertility(enc, lang) / word_fertility(enc, 'english')
        K = int(min(max(round(param * phi), K_MIN), content_kmax(enc))); ch = chunks_fixed_token(offs, K)
    elif policy == 'FSBS':
        K = budget_fsb(enc, lang, param); ch = chunks_sentence(docs, offs, K)
    else:
        raise ValueError(policy)
    arr = np.array(ch, dtype=np.int64).reshape(-1, 4)
    return arr, K

# %% [markdown]
# ## 6. Encoding and exact search

# %%
_MODEL = {}
def get_model(enc):
    s = ENCODER_SPECS[enc]
    if s.get('kind') == 'hash':
        return None
    if enc not in _MODEL:
        for k in list(_MODEL): del _MODEL[k]
        if torch is not None and torch.cuda.is_available(): torch.cuda.empty_cache()
        from transformers import AutoModel
        dtype = torch.float16 if (s['dtype'] == 'fp16' and DEVICE == 'cuda') else torch.float32
        try:
            m = AutoModel.from_pretrained(s['hf'], torch_dtype=dtype, attn_implementation='sdpa')
        except Exception:
            m = AutoModel.from_pretrained(s['hf'], torch_dtype=dtype)
        _MODEL[enc] = m.to(DEVICE).eval()
        if dtype == torch.float16 and s.get('check_fp16'):
            fp16_check(enc)
    return _MODEL[enc]

def fp16_check(enc):
    """Compare fp16 GPU embeddings with fp32 CPU embeddings on a few texts; fall back to fp32 if they differ."""
    from transformers import AutoModel
    s = ENCODER_SPECS[enc]
    texts = [d['text'][:2000] for d in DATA['corpora'][LANGS_FINAL[-1]][:4]] + [q['text'] for q in DATA['questions']['english'][:4]]
    a = encode(enc, texts, 'doc').astype(np.float32)
    ref = AutoModel.from_pretrained(s['hf'], torch_dtype=torch.float32).eval()
    _MODEL[enc], keep = ref, _MODEL[enc]
    global DEVICE
    dev0 = DEVICE; DEVICE = 'cpu'
    b = encode(enc, texts, 'doc').astype(np.float32)
    DEVICE = dev0; _MODEL[enc] = keep
    cos = (a * b).sum(1)
    print(f'[{enc}] fp16 vs fp32 cosine: min {cos.min():.4f}')
    if cos.min() < 0.99:
        print(f'[{enc}] fp16 is not safe: reloading in fp32 (slower)')
        _MODEL[enc] = ref.to(DEVICE)
    else:
        del ref

def _ids(enc, texts, kind):
    s = ENCODER_SPECS[enc]; tok = get_tokenizer(enc)
    pre = s['q_prefix'] if kind == 'query' else s['d_prefix']
    ids = tok([pre + t for t in texts], add_special_tokens=True, truncation=True, max_length=s['max_len'])['input_ids']
    if s.get('append_eos'):
        eos = tok.convert_tokens_to_ids(s['eos_token']) if s.get('eos_token') else (tok.eos_token_id or tok.pad_token_id)
        ids = [x if x and x[-1] == eos else (x[:s['max_len'] - 1] + [eos]) for x in ids]
    return ids

def _hash_embed(ids_list, dim=256):
    out = np.zeros((len(ids_list), dim), np.float32)
    for i, ids in enumerate(ids_list):
        for t in ids:
            out[i, (t * 2654435761) % dim] += 1.0 + (t % 7) / 7
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    return out.astype(np.float16)

STATS = {'tokens': 0, 'seconds': 0.0, 'truncated': 0}
def encode(enc, texts, kind):
    s = ENCODER_SPECS[enc]
    ids = _ids(enc, texts, kind)
    if s.get('kind') == 'hash':
        return _hash_embed(ids)
    model = get_model(enc); tok = get_tokenizer(enc)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    lens = np.array([len(x) for x in ids]); order = np.argsort(-lens, kind='stable')
    out = np.zeros((len(ids), model.config.hidden_size), np.float16)
    t0 = time.time(); i = 0; budget = s['batch_tokens']
    while i < len(order):
        L = lens[order[i]]; bs = max(1, min(512, budget // max(L, 1)))
        idx = order[i:i+bs]
        batch = np.full((len(idx), L), pad, np.int64); mask = np.zeros((len(idx), L), np.int64)
        for j, k in enumerate(idx):
            x = ids[k]
            if s['padding_side'] == 'left': batch[j, L-len(x):] = x; mask[j, L-len(x):] = 1
            else: batch[j, :len(x)] = x; mask[j, :len(x)] = 1
        try:
            with torch.inference_mode():
                bt = torch.from_numpy(batch).to(DEVICE); mt = torch.from_numpy(mask).to(DEVICE)
                h = model(input_ids=bt, attention_mask=mt).last_hidden_state
                if s['pooling'] == 'cls': e = h[:, 0]
                elif s['pooling'] == 'mean':
                    m = mt.unsqueeze(-1).to(h.dtype); e = (h * m).sum(1) / m.sum(1)
                else: e = h[:, -1]                              # last token (left padding)
                e = torch.nn.functional.normalize(e.float(), dim=-1)
                if not torch.isfinite(e).all(): raise FloatingPointError('non-finite embedding (fp16 overflow?)')
                out[idx] = e.cpu().numpy().astype(np.float16)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); budget = max(budget // 2, L); print(f'  OOM: batch token budget -> {budget}'); continue
        i += len(idx)
    STATS['tokens'] += int(lens.sum()); STATS['seconds'] += time.time() - t0
    STATS['truncated'] += int((lens >= s['max_len']).sum())
    return out

def search(chunk_emb, q_emb, k):
    k = min(k, len(chunk_emb))
    if faiss is not None:
        index = faiss.IndexFlatIP(chunk_emb.shape[1])
        for i in range(0, len(chunk_emb), 200000):
            index.add(chunk_emb[i:i+200000].astype(np.float32))
        D, I = index.search(q_emb.astype(np.float32), k)
        return D, I
    sims = q_emb.astype(np.float32) @ chunk_emb.astype(np.float32).T
    I = np.argsort(-sims, axis=1, kind='stable')[:, :k]
    return np.take_along_axis(sims, I, 1), I

def query_embeddings(enc, lang):
    p = WORK / 'data' / f'qemb_{enc}_{lang}.npy'
    if p.exists(): return np.load(p)
    e = encode(enc, [q['text'] for q in DATA['questions'][lang]], 'query'); np.save(p, e); return e

# %% [markdown]
# ## 7. Metrics
# Retrieved chunks are mapped to the TyDi passages they overlap, in rank order and then document order, so a long chunk takes as many ranks as the passages it covers. nDCG@10 (primary), MRR@10 and Recall@5 are computed on that passage ranking. AR@W is the share of questions whose minimal answer lies inside the first W English-equivalent words of retrieved text. Hit@5 (chunk level) is reported only as a secondary metric.

# %%
def passages_in(doc, cs, ce):
    P = doc['passages']
    if len(P) == 0: return range(0)
    i0 = int(np.searchsorted(P[:, 1], cs, 'right')); i1 = int(np.searchsorted(P[:, 0], ce, 'left'))
    return range(i0, max(i0, i1))

def merge(iv):
    iv = sorted(iv); out = []
    for s, e in iv:
        if out and s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return out

def score_queries(enc, lang, chunks, I):
    docs = DATA['corpora'][lang]; qs = DATA['questions'][lang]; offs = doc_tokens(enc, lang)
    pi = PREM[enc][lang]['pi']; rows = []
    for qi, q in enumerate(qs):
        G = q['gold']; ranked, seen, hit5 = [], set(), 0
        for r, c in enumerate(I[qi]):
            d, cs, ce, nt = chunks[c]
            if r < 5 and d == q['doc'] and any(p in G for p in passages_in(docs[d], cs, ce)): hit5 = 1
            for p in passages_in(docs[d], cs, ce):
                if (d, p) not in seen:
                    seen.add((d, p)); ranked.append(1 if (d == q['doc'] and p in G) else 0)
            if len(ranked) >= 10 and r >= 4: break
        rel = np.array(ranked[:10] + [0] * (10 - len(ranked[:10])))
        dcg = (rel / np.log2(np.arange(2, 12))).sum()
        idcg = (1 / np.log2(np.arange(2, 2 + min(len(G), 10)))).sum()
        first = np.flatnonzero(rel)
        row = dict(qid=q['qid'], ndcg10=dcg / idcg, mrr10=(1 / (first[0] + 1) if len(first) else 0.0),
                   recall5=rel[:5].sum() / len(G), hit5_chunk=hit5)
        for W in AR_W:
            if len(q['minimal']) == 0: row[f'ar{W}'] = np.nan; continue
            budget = int(round(W * T_EN[enc] * pi)); used = 0; spans = []
            for c in I[qi]:
                d, cs, ce, nt = chunks[c]
                take = min(nt, budget - used)
                if take <= 0: break
                if take < nt:
                    st = offs[d][:, 0]; t0 = int(np.searchsorted(st, cs, 'left'))
                    ce = int(offs[d][min(t0 + take, len(st)) - 1, 1])
                if d == q['doc']: spans.append((int(cs), int(ce)))
                used += take
            m = merge(spans)
            row[f'ar{W}'] = float(any(any(s <= a and b <= e for s, e in m) for a, b in q['minimal']))
        rows.append(row)
    return pd.DataFrame(rows)

# %% [markdown]
# ## 8. Run plan
# Order = priority: H1/H2 (B1 sweep) and H3 (FSB sweep) for the primary encoder first, then RQ3 baselines, then robustness (FSB-W, FSB-S), then the replication encoder(s).

# %%
def token_grid(enc):
    if TOKEN_GRID: return [k for k in TOKEN_GRID if k <= content_kmax(enc)]
    top = max(budget_fsb(enc, l, max(CONTENT_GRID) * T_EN[enc]) for l in LANGS_FINAL)
    g, k = [], 32
    while True:
        g.append(k)
        if k >= top or k >= content_kmax(enc): break
        k = int(k * 1.5) if len(g) % 2 else int(k * 4 / 3)
    for k in RQ3_K:
        if k not in g and k <= content_kmax(enc): g.append(k)
    return sorted(set(min(x, content_kmax(enc)) for x in g))

def p95_tokens(enc, lang, policy, param):
    arr, _ = make_chunks(enc, lang, policy, param)
    return float(np.percentile(arr[:, 3], 95)) if len(arr) else 0.0

def plan(enc):
    """List of (priority, policy, param). For a replication encoder, keep jobs that fit its window in every language."""
    jobs = []
    jobs += [(1, 'B1', k) for k in token_grid(enc)]
    jobs += [(1, 'FSBC', c) for c in CONTENT_GRID]
    jobs += [(2, p, k) for k in RQ3_K for p in ('FSB', 'B2', 'B4')] + [(2, 'B3', 0)]
    jobs += [(3, p, k) for k in RQ3_K for p in ('FSBW', 'FSBS')]
    if enc != ENCODERS[0]:
        km = content_kmax(enc); keep = []
        for j in jobs:
            _, pol, prm = j
            if pol in ('B1', 'B2'): ok = prm <= km
            elif pol in ('FSB', 'FSBS'): ok = all(budget_fsb(enc, l, prm) < km for l in LANGS_FINAL) and prm <= km
            elif pol == 'FSBC': ok = all(round(prm * T_EN[enc] * PREM[enc][l]['pi']) <= km for l in LANGS_FINAL)
            elif pol == 'FSBW':
                ok = all(round(prm * word_fertility(enc, l) / word_fertility(enc, 'english')) <= km for l in LANGS_FINAL)
            else: ok = all(p95_tokens(enc, l, pol, prm) <= km for l in LANGS_FINAL)
            if ok: keep.append(j)
        jobs = keep
    if enc in MAX_PRIORITY: jobs = [j for j in jobs if j[0] <= MAX_PRIORITY[enc]]
    return sorted(jobs, key=lambda j: j[0])

def run_path(enc, lang, policy, param):
    return WORK / 'runs' / enc / lang / f'{policy}_{param:g}'

THROUGHPUT_GUESS = {'bge-m3': 20000, 'me5-base': 60000, 'qwen3-0.6b': 12000}
print('Estimated cost (tokens to embed):')
TOTAL_H = 0
for enc in ENCODERS:
    corpus_tok = {l: sum(len(a) for a in doc_tokens(enc, l)) for l in LANGS_FINAL}
    js = plan(enc); tot = 0
    for _, pol, prm in js:
        tot += sum(corpus_tok.values()) * (2 if pol == 'B2' else 1)
    done = sum(all((run_path(enc, l, p, k) / 'scores.csv.gz').exists() for l in LANGS_FINAL) for _, p, k in js)
    h = tot / THROUGHPUT_GUESS.get(enc, 20000) / 3600; TOTAL_H += h
    print(f'  {enc}: {len(js)} configurations x {len(LANGS_FINAL)} languages, {tot/1e6:,.0f}M tokens, about {h:.1f} GPU-hours (guess); {done} configurations already done')
    print('    corpus tokens per language:', {l: f'{v/1e6:.2f}M' for l, v in corpus_tok.items()})

# %% [markdown]
# ## 9. Run everything (resumable)
# Each finished (encoder, language, policy, budget) writes `scores.csv.gz` (per-query metrics) and `run.npz` (top-100 chunk spans), so metrics can be recomputed without re-embedding.

# %%
SUMMARY = WORK / 'runs_summary.csv'
def run_all():
    rows = pd.read_csv(SUMMARY).to_dict('records') if SUMMARY.exists() else []
    for enc in ENCODERS:
        for prio, pol, prm in plan(enc):
            for lang in LANGS_FINAL:
                out = run_path(enc, lang, pol, prm)
                if (out / 'scores.csv.gz').exists(): continue
                out.mkdir(parents=True, exist_ok=True); t0 = time.time(); tok0 = STATS['tokens']
                chunks, K = make_chunks(enc, lang, pol, prm)
                texts = [DATA['corpora'][lang][d]['text'][cs:ce] for d, cs, ce, _ in chunks]
                cemb = encode(enc, texts, 'doc'); qemb = query_embeddings(enc, lang)
                D, I = search(cemb, qemb, TOPK)
                sc = score_queries(enc, lang, chunks, I)
                sc.insert(0, 'lang', lang); sc.insert(0, 'policy', pol); sc.insert(0, 'param', prm); sc.insert(0, 'encoder', enc)
                np.savez_compressed(out / 'run.npz', I=I.astype(np.int32), D=D.astype(np.float16), chunks=chunks.astype(np.int32))
                sc.to_csv(out / 'scores.csv.gz', index=False)
                dt = time.time() - t0; ntok = STATS['tokens'] - tok0
                row = dict(encoder=enc, lang=lang, policy=pol, param=prm, K_tokens=K, n_chunks=len(chunks),
                           mean_chunk_tokens=float(chunks[:, 3].mean()), ndcg10=sc.ndcg10.mean(), seconds=round(dt, 1),
                           tok_per_s=round(ntok / max(dt, 1e-9)), priority=prio, finished=time.strftime('%Y-%m-%d %H:%M:%S'))
                rows.append(row); pd.DataFrame(rows).to_csv(SUMMARY, index=False)
                print(f"[{row['finished']}] {enc} {lang:10s} {pol}({prm:g}) K={K} chunks={len(chunks):,} nDCG@10={row['ndcg10']:.3f} {dt:.0f}s {row['tok_per_s']:,} tok/s")
                del cemb, texts
if not RUN_EXPERIMENTS:
    print('RUN_EXPERIMENTS = False: skipping runs, analysing saved results only.')
elif not LOCAL_TEST and DEVICE != 'cuda':
    print('No GPU: runs skipped (they would take days on CPU). Saved results are analysed below.')
else:
    run_all()
    print('All runs finished.')

# %% [markdown]
# ## 10. Analysis
# Run after Section 9 finishes. Everything below reads only the saved per-query scores.

# %%
from scipy import stats as sstats
import statsmodels.formula.api as smf
import matplotlib, matplotlib.ticker
matplotlib.use('Agg') if not IN_COLAB else None
import matplotlib.pyplot as plt

A = WORK / 'analysis'
def load_scores(enc, policy, params=None):
    fs = []
    for lang in LANGS_FINAL:
        for p in glob.glob(str(WORK / 'runs' / enc / lang / f'{policy}_*' / 'scores.csv.gz')):
            fs.append(pd.read_csv(p))
    if not fs: return pd.DataFrame()
    df = pd.concat(fs, ignore_index=True)
    return df[df.param.isin(params)] if params is not None else df

def matrix(df, lang, grid, metric='ndcg10'):
    sub = df[df.lang == lang].pivot(index='qid', columns='param', values=metric)
    return sub[[g for g in grid if g in sub.columns]].to_numpy(), [g for g in grid if g in sub.columns]

def boot_means(M, rng, B):
    n = M.shape[0]; W = rng.multinomial(n, np.full(n, 1 / n), size=B) / n
    return W @ M                                                   # B x grid

def vertex(x_log, Y, local=False):
    """Vertex of a quadratic in log budget for each row of Y; clipped to the grid; argmax if not concave."""
    out = np.empty(len(Y))
    for i, y in enumerate(Y):
        xs, ys = x_log, y
        if local:
            j = int(np.argmax(y)); lo, hi = max(0, j - 2), min(len(y), j + 3); xs, ys = x_log[lo:hi], y[lo:hi]
        a, b, c = np.polyfit(xs, ys, 2) if len(xs) >= 3 else (0, 0, 0)
        v = -b / (2 * a) if a < 0 else x_log[int(np.argmax(y))]
        out[i] = np.exp(np.clip(v, x_log[0], x_log[-1]))
    return out

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (m - r) * p[i])); adj[i] = run
    return adj

def sign_flip(d, n_perm, rng):
    d = d[~np.isnan(d)]; obs = abs(d.mean())
    S = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    return (1 + (np.abs((S * d).mean(1)) >= obs - 1e-12).sum()) / (n_perm + 1)

def analyse(enc):
    rng = np.random.default_rng(SEED); res = {}
    pi = {l: PREM[enc][l]['pi'] for l in LANGS_FINAL}; ten = T_EN[enc]
    b1 = load_scores(enc, 'B1'); fsbc = load_scores(enc, 'FSBC')
    if b1.empty or fsbc.empty:
        print(f'[{enc}] no sweep results yet'); return None
    # ---- H1: mixed-effects model, score ~ language x budget + (1 | query), likelihood-ratio test of the interaction
    d = b1.copy(); d['budget'] = d.param.astype(int).astype(str)
    full = smf.mixedlm('ndcg10 ~ C(lang) * C(budget)', d, groups=d['qid']).fit(reml=False)
    red = smf.mixedlm('ndcg10 ~ C(lang) + C(budget)', d, groups=d['qid']).fit(reml=False)
    lr = 2 * (full.llf - red.llf); df_ = len(full.fe_params) - len(red.fe_params)
    res['H1'] = dict(LR=lr, df=df_, p=float(sstats.chi2.sf(lr, df_)), n_rows=len(d), n_queries=d.qid.nunique())
    # ---- H2: bootstrap K*_L and its rank correlation with pi_L
    grid = sorted(b1.param.unique()); xk = np.log(grid); Kstar = {}; Kloc = {}
    for l in LANGS_FINAL:
        M, g = matrix(b1, l, grid); Bm = boot_means(M, rng, N_BOOT)
        Kstar[l] = vertex(np.log(g), Bm); Kloc[l] = vertex(np.log(g), Bm, local=True)
        Kstar[l + '_point'] = vertex(np.log(g), M.mean(0, keepdims=True))[0]
    order = list(LANGS_FINAL); piv = np.array([pi[l] for l in order])
    rho_b = np.array([sstats.spearmanr(piv, [Kstar[l][b] for l in order])[0] for b in range(N_BOOT)])
    rho_pt = sstats.spearmanr(piv, [Kstar[l + '_point'] for l in order])[0]
    res['H2'] = dict(rho_point=rho_pt, rho_ci95=np.nanpercentile(rho_b, [2.5, 97.5]).tolist(), share_rho_gt0=float(np.nanmean(rho_b > 0)),
                     Kstar={l: dict(point=Kstar[l + '_point'], ci95=np.percentile(Kstar[l], [2.5, 97.5]).tolist(),
                                    local_point=float(np.median(Kloc[l]))) for l in order})
    # ---- H3: bootstrap C*_L under FSB; TOST via the 90% CI of log(C*_L / C*_en) inside the bounds
    cgrid = sorted(fsbc.param.unique()); Cstar = {}
    for l in LANGS_FINAL:
        M, g = matrix(fsbc, l, cgrid); Cstar[l] = vertex(np.log(g), boot_means(M, rng, N_BOOT))
        Cstar[l + '_point'] = vertex(np.log(g), M.mean(0, keepdims=True))[0]
    # same quantity under fixed-token chunking (content of a K-token chunk in language L = K / (t_en * pi_L))
    Cfix = {l: Kstar[l] / (ten * pi[l]) for l in LANGS_FINAL}
    lo, hi = np.log(EQ_BOUNDS[0]), np.log(EQ_BOUNDS[1]); h3 = {}
    for l in LANGS_FINAL:
        if l == 'english': continue
        r = np.log(Cstar[l] / Cstar['english']); rf = np.log(Cfix[l] / Cfix['english'])
        ci90 = np.percentile(r, [5, 95]); ci90f = np.percentile(rf, [5, 95])
        h3[l] = dict(ratio_point=Cstar[l + '_point'] / Cstar['english_point'], ratio_ci90=np.exp(ci90).tolist(),
                     equivalent=bool(ci90[0] > lo and ci90[1] < hi),
                     fixed_token_ratio_ci90=np.exp(ci90f).tolist(), fixed_token_equivalent=bool(ci90f[0] > lo and ci90f[1] < hi))
    # exploratory: does FSB bring the optima closer than fixed-token chunking? (largest |log ratio| across languages)
    spread_fsb = np.max([np.abs(np.log(Cstar[l] / Cstar['english'])) for l in LANGS_FINAL if l != 'english'], axis=0)
    spread_fix = np.max([np.abs(np.log(Cfix[l] / Cfix['english'])) for l in LANGS_FINAL if l != 'english'], axis=0)
    res['H3'] = dict(per_language=h3, all_equivalent=all(v['equivalent'] for v in h3.values()),
                     Cstar={l: dict(point=Cstar[l + '_point'], ci95=np.percentile(Cstar[l], [2.5, 97.5]).tolist()) for l in LANGS_FINAL},
                     exploratory_spread=dict(fsb_median=float(np.median(spread_fsb)), fixed_median=float(np.median(spread_fix)),
                                             diff_ci95=np.percentile(spread_fsb - spread_fix, [2.5, 97.5]).tolist()))
    # ---- RQ3: FSB vs each baseline, paired sign-flip randomization test, Holm across all comparisons
    rows = []
    b3 = load_scores(enc, 'B3')
    for K in RQ3_K:
        f = load_scores(enc, 'FSB', [K])
        if f.empty: continue
        for base, bdf in [('B1', load_scores(enc, 'B1', [K])), ('B2', load_scores(enc, 'B2', [K])),
                          ('B3', b3), ('B4', load_scores(enc, 'B4', [K]))]:
            if bdf.empty: continue
            for l in LANGS_FINAL:
                if base == 'B1' and l == 'english': continue    # FSB equals B1 in English
                a = f[f.lang == l].set_index('qid').ndcg10; b = bdf[bdf.lang == l].set_index('qid').ndcg10
                dlt = (a - b.reindex(a.index)).to_numpy()
                bm = boot_means(dlt[:, None], rng, N_BOOT)[:, 0]
                rows.append(dict(K=K, baseline=base, lang=l, fsb=a.mean(), base=b.mean(), diff=np.nanmean(dlt),
                                 diff_ci95_lo=np.percentile(bm, 2.5), diff_ci95_hi=np.percentile(bm, 97.5),
                                 p=sign_flip(dlt, N_PERM, rng)))
    rq3 = pd.DataFrame(rows)
    if len(rq3): rq3['p_holm'] = holm(rq3.p); rq3['significant'] = rq3.p_holm < ALPHA
    rq3.to_csv(A / f'rq3_{enc}.csv', index=False); res['RQ3_n_tests'] = len(rq3)
    # ---- robustness: FSB-W and FSB-S against FSB, AR@W control
    rob = []
    for pol in ('FSBW', 'FSBS'):
        for K in RQ3_K:
            x = load_scores(enc, pol, [K]); f = load_scores(enc, 'FSB', [K])
            if x.empty or f.empty: continue
            for l in LANGS_FINAL:
                a = x[x.lang == l].set_index('qid').ndcg10; b = f[f.lang == l].set_index('qid').ndcg10
                dlt = (a - b.reindex(a.index)).to_numpy()
                rob.append(dict(policy=pol, K=K, lang=l, ndcg=a.mean(), fsb=b.mean(), diff=np.nanmean(dlt), p=sign_flip(dlt, N_PERM, rng)))
    pd.DataFrame(rob).to_csv(A / f'robustness_{enc}.csv', index=False)
    allsc = pd.concat([load_scores(enc, p) for p in ('B1', 'FSBC', 'FSB', 'B2', 'B3', 'B4', 'FSBW', 'FSBS')], ignore_index=True)
    allsc.groupby(['policy', 'param', 'lang'])[['ndcg10', 'mrr10', 'recall5', 'hit5_chunk'] + [f'ar{w}' for w in AR_W]].mean() \
         .round(4).to_csv(A / f'means_{enc}.csv')
    # ---- Shona: predicted budgets (no retrieval data)
    res['prediction'] = {l: dict(pi=PREM[enc][l]['pi'], fsb_budget_tokens={c: budget_fsb(enc, l, c * ten) for c in CONTENT_GRID},
                                 predicted_Kstar_if_H3=float(Cstar['english_point'] * ten * PREM[enc][l]['pi']))
                         for l in PREDICT_ONLY}
    # ---- figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for l in LANGS_FINAL:
        M, g = matrix(b1, l, grid); bm = boot_means(M, rng, 1000)
        ax[0].plot(g, M.mean(0), marker='o', label=f'{l} (π={pi[l]:.2f})')
        ax[0].fill_between(g, *np.percentile(bm, [2.5, 97.5], axis=0), alpha=0.15)
        M, g = matrix(fsbc, l, cgrid); bm = boot_means(M, rng, 1000)
        ax[1].plot(g, M.mean(0), marker='o', label=l); ax[1].fill_between(g, *np.percentile(bm, [2.5, 97.5], axis=0), alpha=0.15)
    for a_, xl in zip(ax, ['Chunk budget (tokens), fixed-token chunking', 'Content budget (English-equivalent words), FSB']):
        a_.set_xscale('log'); a_.set_xlabel(xl); a_.set_ylabel('nDCG@10'); a_.grid(alpha=0.3)
        a_.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([32, 64, 128, 256, 512]))
        a_.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); a_.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(A / f'fig_budget_curves_{enc}.png', dpi=200); plt.close(fig)
    json.dump(res, open(A / f'results_{enc}.json', 'w'), indent=1, default=float)
    return res

RESULTS = {}
for enc in ENCODERS:
    r = analyse(enc)
    if r:
        RESULTS[enc] = r
        print(f'\n===== {enc} =====')
        print('H1 interaction LR test:', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r['H1'].items()})
        print('H2 Spearman(K*_L, pi_L):', round(r['H2']['rho_point'], 3), 'CI95', np.round(r['H2']['rho_ci95'], 3))
        for l, v in r['H2']['Kstar'].items(): print(f'   K*_{l}: {v["point"]:.0f} tokens, CI95 {np.round(v["ci95"]).tolist()}')
        print('H3 all languages equivalent:', r['H3']['all_equivalent'])
        for l, v in r['H3']['per_language'].items():
            print(f"   {l}: C*_L/C*_en = {v['ratio_point']:.2f}, 90% CI {np.round(v['ratio_ci90'], 2).tolist()}, equivalent={v['equivalent']}"
                  f" | fixed-token: {np.round(v['fixed_token_ratio_ci90'], 2).tolist()}, equivalent={v['fixed_token_equivalent']}")
        print('Exploratory spread (FSB vs fixed):', r['H3']['exploratory_spread'])
print('\nAnalysis files written to', A)

# %% [markdown]
# ## 11. Package the results
# Download this zip and bring it back for the Results section.

# %%
import shutil
if SUMMARY.exists():
    shutil.copy(SUMMARY, A / 'runs_summary.csv')
z = shutil.make_archive(str(WORK / f'results_{RUN_NAME}'), 'zip', root_dir=str(WORK), base_dir='analysis')
print('Download:', z)
if IN_COLAB:
    from google.colab import files
    files.download(z)
elif IN_KAGGLE:
    print('Kaggle: the zip is in the Output tab after Save Version.')
