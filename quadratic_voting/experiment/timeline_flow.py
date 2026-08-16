"""Single self-contained SVG-and-sidebar timeline renderer."""

from __future__ import annotations

import json
from pathlib import Path

from quadratic_voting.experiment.timeline import _read, build_timeline_payload


_DOCUMENT_PREFIX = """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quadratic voting flow timeline</title>
<style>
:root{color-scheme:dark}body{font:14px system-ui;margin:0;background:#101820;color:#edf4f7}header,main{max-width:1540px;margin:auto;padding:1rem}button,select,input{font:inherit;margin:.25rem}:focus-visible{outline:3px solid #edc948}.round{margin:1rem 0;padding:1rem;background:#172933;border-left:5px solid #466879;scroll-margin-top:1rem}.selected{border-color:#edc948;box-shadow:0 0 0 2px #edc948}.canvas{overflow-x:auto}.flow-layout{display:grid;grid-template-columns:850px minmax(390px,1fr);gap:12px;min-width:1260px;align-items:start}svg{width:850px;background:#0d171d}.edge{fill:none;stroke-opacity:.48}.node{fill:#223c4a;stroke:#d8e5eb}.candidate{fill:#172933}.svg-label{fill:#edf4f7;font-size:12px}.candidate-sidebar{display:grid;gap:8px}.candidate-card,.voter-card,.outcome-summary{background:#203740;padding:10px;border-left:4px solid #4e79a7}.candidate-card{min-height:66px}.candidate-meta{display:flex;gap:.5rem;flex-wrap:wrap}.status{font-weight:700}.conversation-note,small{color:#c2d1d8}.turn{display:grid;grid-template-columns:5.5rem minmax(0,1fr);gap:.5rem;margin-top:6px;background:#0d171d;padding:6px}.turn-role{font-weight:700}.turn-text{white-space:pre-wrap;overflow-wrap:anywhere}.voter-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:10px}.allocation-table{width:100%;border-collapse:collapse;margin-top:8px}.allocation-table th,.allocation-table td{border-bottom:1px solid #48606a;padding:4px;text-align:left;vertical-align:top}.statement{white-space:pre-wrap;overflow-wrap:anywhere}.bar{height:12px;display:flex;background:#8a8a8a;margin:6px 0}.seg{height:100%}details{margin-top:10px}summary{cursor:pointer}.provenance-list{white-space:pre-wrap;overflow-wrap:anywhere}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}@media(max-width:600px){header,main{padding:.6rem}.round{padding:.6rem}.flow-layout{grid-template-columns:850px 390px}}
</style>
<header><h1>Quadratic-voting allocation flows</h1><label>Condition / run <select id="run"></select></label><button id="previous">Previous</button><button id="next">Next</button><label>Round <input id="round" type="range"></label><output id="round-label"></output><p id="explanation"></p></header>
<main id="diagrams" aria-live="polite"></main>
<script id="timeline-data" type="application/json">"""

_DOCUMENT_SUFFIX = """</script><script>
const D=JSON.parse(document.querySelector('#timeline-data').textContent),q=s=>document.querySelector(s),run=q('#run'),sl=q('#round'),out=q('#round-label'),root=q('#diagrams');
const e=(name,text)=>{const node=document.createElement(name);if(text!=null)node.textContent=text;return node};
const svg=(name,attributes={})=>{const node=document.createElementNS('http://www.w3.org/2000/svg',name);Object.entries(attributes).forEach(([key,value])=>node.setAttribute(key,value));return node};
const title=text=>{const node=svg('title');node.textContent=text;return node};
const candidateStatus=c=>c.protected?'Protected this round':c.removed?'Removed this round':c.winner?'Final winner':'Active';
const aggregateVotes=c=>c.aggregateVotes==null?'Aggregate votes unavailable':`Aggregate votes: ${c.aggregateVotes}`;
const rudenessLabel=value=>String(value??'Rudeness unavailable').replaceAll('_',' ').replace(/^./,letter=>letter.toUpperCase());
const sourceRatings=c=>c.sourceSeverityN===0?'Source score unavailable':`Source abuse ratings: ${c.sourceSeverityRatings.join(', ')} · Mean: ${c.sourceSeverityMean.toFixed(2)} (n=${c.sourceSeverityN})`;
const malformedSourceStatus=c=>c.sourceSeverityMalformedAnnotatorCount===0?'':` · ${c.sourceSeverityMalformedAnnotatorCount} malformed annotator${c.sourceSeverityMalformedAnnotatorCount===1?'':'s'} excluded`;
const roleLabel=role=>role==='user'?'User':role==='assistant'||role==='agent'?'Model':String(role??'Unknown role');
const ratingLabel=rating=>({"-2":"Strongly prefer not to continue","-1":"Prefer not to continue","0":"Neutral","1":"Prefer to continue","2":"Strongly prefer to continue"}[String(rating)]??'Not applicable')+(rating==null?'':` (code ${rating})`);
const addClass=(node,className)=>{node.className=className;return node};
const activeCandidates=frame=>frame.candidates.filter(candidate=>candidate.active);
function candidateCard(candidate){const card=addClass(e('article'),'candidate-card');card.append(e('h3',candidate.label),addClass(e('div',rudenessLabel(candidate.rudeness)),'candidate-meta'),e('div',sourceRatings(candidate)+malformedSourceStatus(candidate)),e('div',aggregateVotes(candidate)),addClass(e('div',candidateStatus(candidate)),'status'),addClass(e('div',`Conversation actually shown in this pilot (${candidate.sourceTurns.length} messages)`),'conversation-note'));candidate.sourceTurns.forEach(turn=>{const row=addClass(e('div'),'turn');row.append(addClass(e('div',roleLabel(turn.role)),'turn-role'),addClass(e('div',turn.text??'Unavailable'),'turn-text'));card.append(row)});return card}
function diagram(frame,index){const section=addClass(e('section'),'round');section.id='round-diagram-'+index;section.tabIndex=-1;section.setAttribute('aria-label','Allocation flow diagram for round '+frame.round);section.append(e('h2','Round '+frame.round+' allocation flow'));const active=activeCandidates(frame),height=Math.max(260,80+Math.max(frame.voters.length,active.length)*76),layout=addClass(e('div'),'flow-layout'),wrap=addClass(e('div'),'canvas'),image=svg('svg',{viewBox:`0 0 850 ${height}`,height:String(height),role:'img','aria-label':'Voters on the left allocate quadratic credits to active candidate anchors on the right'}),voterY=i=>55+i*76,candidateY=i=>55+i*76;
frame.voters.forEach((voter,index)=>{const y=voterY(index),group=svg('g',{'aria-label':'Voter V'+voter.voter});group.append(svg('rect',{x:25,y:y-18,width:190,height:42,rx:4,class:'node'}),title('V'+voter.voter+': '+(voter.spend==null?'abstained or missing ballot':`spent ${voter.spend} of ${voter.budget}, unspent ${voter.budget-voter.spend}`)));const label=svg('text',{x:34,y:y,class:'svg-label'});label.textContent='V'+voter.voter+' · '+(voter.spend==null?'abstained/missing':voter.spend+'/'+voter.budget);group.append(label);image.append(group);voter.allocations.filter(allocation=>allocation.credits>0).forEach(allocation=>{const target=active.findIndex(candidate=>candidate.label===allocation.label);if(target<0)return;const edge=svg('path',{d:`M215 ${y} C370 ${y}, 490 ${candidateY(target)}, 610 ${candidateY(target)}`,class:'edge',stroke:D.colors[allocation.label]||'#aaa','stroke-width':String(Math.max(1,Math.sqrt(allocation.credits)*1.7))});edge.append(title(`V${voter.voter} → ${allocation.label}: ${allocation.votes} votes; ${allocation.credits} quadratic credits`));image.append(edge)})});
active.forEach((candidate,index)=>{const y=candidateY(index),group=svg('g',{'aria-label':'Candidate '+candidate.label});group.append(svg('rect',{x:610,y:y-18,width:205,height:42,rx:4,class:'candidate',stroke:D.colors[candidate.label]||'#aaa'}),title(`${candidate.label} · ${candidateStatus(candidate)}`));const label=svg('text',{x:620,y:y,class:'svg-label'});label.textContent=candidate.label+' · '+candidateStatus(candidate);group.append(label);image.append(group)});wrap.append(image);layout.append(wrap);const sidebar=addClass(e('aside'),'candidate-sidebar');sidebar.setAttribute('aria-label','Candidate conversations');sidebar.append(e('h3','Candidate conversations'));active.forEach(candidate=>sidebar.append(candidateCard(candidate)));layout.append(sidebar);section.append(layout, voterEvidence(frame), outcome(frame), provenance(frame));return section}
function voterEvidence(frame){const section=e('section');section.setAttribute('aria-label','Voter statements and ballot evidence');section.append(e('h3','Voter statements and ballot evidence'));const grid=addClass(e('div'),'voter-grid');frame.voters.forEach(voter=>{const card=addClass(e('article'),'voter-card');card.append(e('h4',`V${voter.voter}`),e('div',`Ballot: ${voter.ballotStatus}; statement: ${voter.statementStatus}`),e('p',voter.rationale??'Rationale unavailable / abstained'));if(voter.spend==null){card.append(e('strong','Abstained or terminal-missing ballot: no unspent budget is inferred.'))}else{const bar=addClass(e('div'),'bar');voter.allocations.forEach(allocation=>{const segment=addClass(e('span'),'seg');segment.style.width=(100*allocation.credits/voter.budget)+'%';segment.style.background=D.colors[allocation.label]||'#666';segment.title=`${allocation.label}: ${allocation.credits} credits`;bar.append(segment)});const unspent=addClass(e('span'),'seg');unspent.style.width=(100*(voter.budget-voter.spend)/voter.budget)+'%';unspent.style.background=D.unspent;unspent.title=`Unspent: ${voter.budget-voter.spend} credits`;bar.append(unspent);card.append(e('div',`Spent ${voter.spend} / ${voter.budget}; unspent ${voter.budget-voter.spend}`),bar)}const table=addClass(e('table'),'allocation-table'),head=e('thead'),headRow=e('tr');['Candidate','Rating','Statement','Votes','Credits'].forEach(label=>headRow.append(e('th',label)));head.append(headRow);table.append(head);const body=e('tbody');voter.allocations.forEach(allocation=>{const row=e('tr');[allocation.label,ratingLabel(allocation.rating),allocation.statement??'Unavailable',allocation.votes??'—',allocation.credits??'—'].forEach((value,index)=>{const cell=e('td',value);if(index===2)cell.className='statement';row.append(cell)});body.append(row)});table.append(body);card.append(table);grid.append(card)});section.append(grid);return section}
function outcome(frame){const section=addClass(e('section'),'outcome-summary');section.append(e('h3','Round outcome summary'),e('div',`Protected: ${frame.outcome.protectedLabel??'Not applicable'}`),e('div',`Removed: ${frame.outcome.removedLabel??'Not sealed'}`),e('div',`Tie: ${frame.outcome.tie??'Unavailable'}`));return section}
function provenance(frame){const details=e('details');details.append(e('summary','Optional raw source annotation provenance'));activeCandidates(frame).forEach(candidate=>{const block=addClass(e('div',`${candidate.label}: ${sourceRatings(candidate)}${malformedSourceStatus(candidate)}`),'provenance-list');details.append(block)});return details}
function build(){const current=D.runs[run.value];sl.min=0;sl.max=current.frames.length-1;sl.value=0;out.textContent='Round '+current.frames[0].round;q('#explanation').textContent=current.regime+': '+current.explanation;root.replaceChildren(...current.frames.map(diagram))}
function select(){const index=+sl.value,current=D.runs[run.value],selected=q('#round-diagram-'+index);out.textContent='Round '+current.frames[index].round;document.querySelectorAll('.round').forEach(node=>node.classList.remove('selected'));selected.classList.add('selected');selected.scrollIntoView({block:'nearest',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});selected.focus({preventScroll:true})}
D.runs.forEach((current,index)=>run.add(new Option(current.regime+' — '+current.arm.replaceAll('-',' '),index)));run.onchange=()=>{build();select()};sl.oninput=select;q('#previous').onclick=()=>{sl.value=Math.max(0,+sl.value-1);select()};q('#next').onclick=()=>{sl.value=Math.min(+sl.max,+sl.value+1);select()};build();select();
</script>"""


def _write_timeline_document(payload: dict[str, object], path: Path) -> Path:
    """Serialize the payload and write one no-network self-contained document."""
    safe_payload = (
        json.dumps(payload, sort_keys=True, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    path.write_text(
        _DOCUMENT_PREFIX + safe_payload + _DOCUMENT_SUFFIX, encoding="utf-8"
    )
    return path


def export_candidate_labels(export_dir: Path) -> list[dict[str, object]]:
    """Derive deterministic C1..Cn labels from the export's candidate set.

    This matches the analyze snapshot labeling (sorted candidate IDs) so the
    plots-directory timeline and the full analyze dashboard agree.
    """
    candidate_ids = sorted(
        {str(row["candidate_id"]) for row in _read(export_dir, "candidate_analysis")}
    )
    return [
        {"candidate_id": candidate_id, "candidate_label": f"C{index}"}
        for index, candidate_id in enumerate(candidate_ids, start=1)
    ]


def render_timeline_html(export_dir: Path, out_dir: Path) -> Path:
    """Write one no-network timeline using textContent for all persisted text."""
    payload = build_timeline_payload(
        export_dir, _read(out_dir, "snapshot_candidate_labels")
    )
    return _write_timeline_document(payload, out_dir / "timeline.html")


def render_export_timeline(export_dir: Path, out_path: Path) -> Path:
    """Render the timeline directly from an export to an explicit file path.

    Labels are derived from the export itself, so no snapshot tables need to be
    materialized. Used to include the timeline alongside static plots.
    """
    payload = build_timeline_payload(export_dir, export_candidate_labels(export_dir))
    return _write_timeline_document(payload, out_path)
