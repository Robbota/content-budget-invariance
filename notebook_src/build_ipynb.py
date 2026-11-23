"""Convert fsb_experiment.py (# %% cells) into FSB_Experiment.ipynb."""
import json, re, sys
from pathlib import Path
src = Path(__file__).with_name('fsb_experiment.py').read_text()
cells = []
for block in re.split(r'^# %%', src, flags=re.M)[1:]:
    md = block.startswith(' [markdown]')
    body = block.split('\n', 1)[1] if '\n' in block else ''
    body = body.strip('\n')
    if md:
        lines = [l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.splitlines()]
        cells.append(dict(cell_type='markdown', metadata={}, source='\n'.join(lines)))
    else:
        cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=body))
for c in cells:
    lines = c['source'].split('\n'); c['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
nb = dict(nbformat=4, nbformat_minor=5, cells=cells,
          metadata=dict(accelerator='GPU', colab=dict(gpuType='T4', provenance=[]),
                        kernelspec=dict(name='python3', display_name='Python 3'), language_info=dict(name='python')))
out = Path(__file__).parent.parent / 'FSB_Experiment.ipynb'
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print('wrote', out, len(cells), 'cells')
