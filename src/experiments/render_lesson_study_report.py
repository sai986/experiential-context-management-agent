"""Render the saved Markdown report into a readable, local HTML artifact."""
from __future__ import annotations

import argparse
import html
from pathlib import Path
import re

from src.experiments.run_openstack_lesson_study import ROOT


def inline(text):
    text = html.escape(text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    return re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)


def render(output: Path):
    rows = (output / 'RESULTS.md').read_text().splitlines()
    body, index = [], 0
    while index < len(rows):
        row = rows[index]
        if row.startswith('|'):
            table = []
            while index < len(rows) and rows[index].startswith('|'):
                table.append([v.strip() for v in rows[index].strip('|').split('|')])
                index += 1
            body.append('<div class="table"><table><thead><tr>' + ''.join('<th>' + inline(v) + '</th>' for v in table[0]) + '</tr></thead><tbody>')
            for cells in table[2:]:
                body.append('<tr>' + ''.join('<td>' + inline(v) + '</td>' for v in cells) + '</tr>')
            body.append('</tbody></table></div>')
            continue
        if row.startswith('#'):
            level = len(row) - len(row.lstrip('#'))
            body.append(f'<h{level}>' + inline(row[level:].strip()) + f'</h{level}>')
        elif row.startswith('- '):
            body.append('<p class="bullet">• ' + inline(row[2:]) + '</p>')
        elif row:
            body.append('<p>' + inline(row) + '</p>')
        index += 1
    if (output / 'rule-selectivity.svg').exists():
        body.insert(2, '<figure><img src="rule-selectivity.svg" alt="Context size and exact fact success across noise levels"><figcaption>Offline controlled probes: frozen OpenStack and synthetic lesson policies.</figcaption></figure>')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Matched lesson study</title><style>'
    page += 'body{max-width:1150px;margin:40px auto;padding:0 24px;font:17px/1.55 system-ui;color:#213645;background:#f7f9fb}h1,h2{color:#235768}h2{margin-top:36px}img{max-width:100%}figure{margin:25px 0;background:white;padding:12px}figcaption{font-size:14px;color:#607078}.table{overflow:auto}table{border-collapse:collapse;width:100%;background:white;margin:18px 0;font-size:15px}th,td{text-align:left;border-bottom:1px solid #d8e1e6;padding:9px}th{background:#e4eff1}code{font:14px monospace;overflow-wrap:anywhere}.bullet{margin:10px 0}a{color:#23706f}</style>'
    page += '<nav><a href="RESULTS.md">Markdown</a> · <a href="rule-selectivity.pdf">Figure PDF</a> · <a href="FULL_CONTEXT_REVIEW.md">Failure review</a> · <a href="verification.json">Verification</a></nav>'
    page += ''.join(body) + '</html>'
    (output / 'RESULTS.html').write_text(page)
    print(output / 'RESULTS.html')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/openstack-lesson-study')
    args = parser.parse_args()
    render(args.output)


if __name__ == '__main__':
    main()
