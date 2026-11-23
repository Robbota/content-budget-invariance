"""Side measurement for Section 5.2: FLORES-200 token premiums under a second tokenizer family.

    pip install transformers
    python qwen3_premiums.py            # writes results/qwen3_premiums.json

pi_L = tokens of the 997 FLORES-200 dev sentences in language L / tokens of the same sentences in
English, counted without special tokens, exactly as in Phase 0 of FSB_Experiment.ipynb. The XLM-R
tokenizer (BGE-M3) is measured again as a check against results/phase0.json. CPU only; no model weights.
"""
import json, tarfile, urllib.request
from pathlib import Path
from transformers import AutoTokenizer

FLORES_URL = 'https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz'
FLORES = dict(english='eng_Latn', indonesian='ind_Latn', finnish='fin_Latn', swahili='swh_Latn', telugu='tel_Telu',
              arabic='arb_Arab', bengali='ben_Beng', russian='rus_Cyrl', japanese='jpn_Jpan', korean='kor_Hang',
              thai='tha_Thai', shona='sna_Latn')
TOKENIZERS = {'qwen3-0.6b': 'Qwen/Qwen3-Embedding-0.6B', 'bge-m3': 'BAAI/bge-m3'}
HERE = Path(__file__).resolve().parent
CACHE = HERE / '.cache'


def flores_dev():
    d = CACHE / 'flores200_dataset' / 'dev'
    if not d.exists():
        CACHE.mkdir(exist_ok=True)
        tgz = CACHE / 'flores200_dataset.tar.gz'
        if not tgz.exists():
            urllib.request.urlretrieve(FLORES_URL, tgz)
        with tarfile.open(tgz) as tf:
            tf.extractall(CACHE, filter='data')
    return d


def main():
    dev = flores_dev()
    sents = {l: (dev / f'{c}.dev').read_text(encoding='utf-8').splitlines() for l, c in FLORES.items()}
    assert all(len(s) == 997 for s in sents.values()), 'expected 997 FLORES-200 dev sentences per language'
    out = {'source': 'FLORES-200 dev (997 sentences)', 'tokenizers': TOKENIZERS, 'premiums': {}}
    for name, repo in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(repo)
        count = lambda ss: sum(len(x) for x in tok(ss, add_special_tokens=False)['input_ids'])
        n_en = count(sents['english'])
        out['premiums'][name] = {l: dict(pi=round(count(s) / n_en, 4), tokens=count(s)) for l, s in sents.items()}
    dst = HERE / 'results' / 'qwen3_premiums.json'
    dst.write_text(json.dumps(out, indent=1))
    print(f'{"language":12s}' + ''.join(f'{n:>12s}' for n in TOKENIZERS))
    for l in FLORES:
        print(f'{l:12s}' + ''.join(f'{out["premiums"][n][l]["pi"]:12.3f}' for n in TOKENIZERS))
    print('written', dst)


if __name__ == '__main__':
    main()
