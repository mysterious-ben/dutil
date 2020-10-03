import os
import subprocess
from pathlib import Path

if os.getenv('GITHUB_REF', '').startswith('refs/tags/'):
    print(os.environ['GITHUB_REF'].split('/')[-1])
else:
    p = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=Path(__file__).parent.parent, stdout=subprocess.PIPE)
    print(p.stdout.decode('utf-8'))