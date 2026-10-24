"""Check the README and assets without requiring archived experiment outputs."""
import ast
import hashlib
import json
import re
from pathlib import Path
import cv2

root=Path(__file__).resolve().parents[1]
text=(root/'README.md').read_text(encoding='utf-8-sig')
images=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text)
links=re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)',text)
review=root/'docs/documentation_review.json'
for target in images+links:
    if target.startswith(('https://', 'http://')):continue
    assert (root/target).exists() or (root/target)==review,target
assert len(images)==14 and len(set(images))==14
for target in images:
    assert target.startswith('./docs/')
    assert cv2.imread(str(root/target)) is not None,target
available_sources=0
for record in json.loads((root/'docs/asset_manifest.json').read_text()):
    data=(root/'docs'/record['file']).read_bytes()
    assert hashlib.sha256(data).hexdigest()==record['sha256']
    if (root/record['source']).exists():
        assert data==(root/record['source']).read_bytes()
        available_sources+=1
snapshot=root/'docs/protected_hashes.json'
protected=json.loads(snapshot.read_text()) if snapshot.exists() else {}
checked_protected=0
for path,digest in protected.items():
    if (root/path).exists():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest,path
        checked_protected+=1
assert text.count('```')%2==0
assert sum(line.strip()=='$$' for line in text.splitlines())%2==0
assert '\\[' not in text and '\\]' not in text
assert 'TODO' not in text and 'Add your text' not in text
assert 'draft' not in text.lower()
assert '\ufffd' not in text and '\u0102' not in text
count=None
for line in text.splitlines():
    if line.startswith('|'):
        n=line.count('|')
        if count is None:count=n
        assert n==count,line
    else:count=None
functions={p.name:{node.name for node in ast.parse(p.read_text(encoding='utf-8-sig')).body if isinstance(node,ast.FunctionDef)} for p in (root/'src').glob('*.py')}
for line in text.splitlines():
    if line.startswith('|') and './src/' in line:
        module=re.search(r'\./src/([^)]*)',line)[1]
        for name in re.findall(r'`([^`]+)`',line):assert name in functions[module],(module,name)
result=dict(status='passed',readme_words=len(text.split()),displayed_images=len(images),
    relative_links_checked=sum(not p.startswith(('https://','http://')) for p in images+links),all_images_in_docs=True,
    asset_hashes_verified=True,available_original_sources_compared=available_sources,protected_existing_files_unchanged=checked_protected,
    template_sha256=hashlib.sha256((root/'writeup_template.md').read_bytes()).hexdigest(),table_structure_checked=True,
    math_delimiters_checked=True,module_function_map_checked=True,
    manual_checks_remaining=['GitHub rendering of mathematics','Remote publication/playback of local MP4 links',
        'Same-camera/full-FOV assumption is not independently verified',
        'No independent metric calibration or dense accuracy ground truth'])
review.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,indent=2))
