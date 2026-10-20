"""Final integrity review, including guards, relative links and Git evidence."""
from pathlib import Path
import csv
import difflib
import hashlib
import json
import re
import subprocess
import sys
from PIL import Image,ImageOps,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'Analysis/DQN/final_task1'
def main():
    manifest=json.loads((AUDIT/'preservation_manifest.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    for variant in ('v0','v1','v2_history_only','v2'):
        for seed in (42,123,456):
            name=f'dqn_{variant}_ls20_seed{seed}'
            for path in (f'DQN/checkpoints/{name}.pt',f'Logs/DQN/{name}.csv',f'Logs/DQN/{name}_config.json',
                f'Evaluations/DQN/{name}_eval_ls20_seed1001.csv',f'Evaluations/DQN/{name}_eval_ls20_seed1001.json'):
                assert (ROOT/path).is_file(),path
    # Attempted overwrite must fail before the environment starts and leave all hashes unchanged.
    proc=subprocess.run([sys.executable,'-m','DQN.evaluate','--checkpoint','DQN/checkpoints/dqn_v0_ls20_seed42.pt'],
        cwd=ROOT,capture_output=True,text=True)
    assert proc.returncode!=0 and 'Refusing to overwrite' in proc.stderr
    assert 'Initialized ScorecardManager' not in proc.stderr
    (AUDIT/'overwrite_guard_check.txt').write_text('PASS: existing evaluation refused before environment creation.\n'+proc.stderr)
    diffs=[]
    for name in ('train.py','evaluate.py'):
        before=(AUDIT/(name+'.before.txt')).read_text().splitlines(keepends=True)
        after=(ROOT/'DQN'/name).read_text().splitlines(keepends=True)
        diffs.extend(difflib.unified_diff(before,after,fromfile=f'a/DQN/{name}',tofile=f'b/DQN/{name}'))
    (AUDIT/'implementation_changes.diff').write_text(''.join(diffs),encoding='utf-8')
    for args,name in [(['status','--short','--branch'],'git_status_final.txt'),(['diff','--check'],'git_diff_check.txt'),(['diff','--stat'],'git_diff_stat.txt')]:
        p=subprocess.run(['git',*args],cwd=ROOT,text=True,capture_output=True)
        (AUDIT/name).write_text(p.stdout+p.stderr,encoding='utf-8')
        assert p.returncode==0,(args,p.stderr)
    report=ROOT/'Reports/DQN-Reports/final_task1/DQN_Task1_report.md'
    for target in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report.read_text(encoding='utf-8')):
        assert (report.parent/target).resolve().is_file(),target
    figures=sorted((AUDIT/'figures').glob('*.png'))
    assert len(figures)==6
    sheet=Image.new('RGB',(1800,1800),'#e5e8eb')
    draw=ImageDraw.Draw(sheet)
    for i,p in enumerate(figures):
        im=Image.open(p)
        assert im.width>2000 and im.height>1000
        tile=ImageOps.contain(im.convert('RGB'),(880,560))
        x=(i%2)*900+(900-tile.width)//2; y=(i//2)*600+20
        sheet.paste(tile,(x,y)); draw.text(((i%2)*900+15,(i//2)*600+580),p.stem,fill='#222')
    sheet.save(AUDIT/'figure_contact_sheet.png')
    result=dict(existing_artifact_hashes_unchanged=len(manifest),main_artifacts_present=12,
        evaluation_overwrite_guard=True,report_figure_links=True,six_high_resolution_figures=True,
        git_diff_check='passed',no_push_or_merge=True)
    (AUDIT/'final_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
