"""The bill calculator on kalunho.com/experiments/tokens-per-task/, from ``results/summary.json``.

    python analysis/make_calculator.py --site ../kalunho17.github.io

Rewrites the block between ``<!-- CALCULATOR START -->`` and ``<!-- CALCULATOR END -->`` in
``<site>/experiments/tokens-per-task/index.html`` and nothing else, so hand edits elsewhere on the
page stay. The reader picks a task and two measured settings; each fills in its tokens per task
and success rate, and the reader's own prices give the bill, the cost per task and the cost per
successful task. The defaults reproduce Finding 4 (EUR 144 against EUR 18,325 per million school
maths questions). Styles: the ``.calc`` block in the site's ``assets/css/site.css``. Hidden without
JavaScript. Nothing is published: commit and push the site repository for that.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from tokens_per_task.config import MODEL_OF, SETTING_OF  # noqa: E402

PAGE = Path("experiments") / "tokens-per-task" / "index.html"
TASKS = {"T1": "Classify a customer message", "T2": "Calculate (school maths)", "T4": "Hard maths", "T3": "Agent (invoice questions)"}
ORDER = ["S", "S-on", "Q-off", "Q-low", "Q-med", "Q-xhigh", "G-off", "G-on", "O-low", "O-high", "Q-A", "Q-A-low"]
USD_PER_EUR = 1.1165                        # ECB reference rate, 6 November 2026, as in Finding 4
LUNA, ASTRA = (0.10, 0.50), (10.0, 50.0)    # OpenAI list prices, $ per 1M tokens in/out, 8 November 2026
DEFAULT = {"task": "T2", "a": "S", "b": "Q-off", "n": 1_000_000}


def presets(summary: dict) -> dict:
    rows = {(r["config"], r["task"]): r for r in summary["by_config_task"] if r.get("seed", 42) == 42}
    out = {}
    for t in TASKS:
        out[t] = {c: {"label": f"{MODEL_OF[c]}, {SETTING_OF[c]}", "tin": round(rows[(c, t)]["mean_prompt_tokens"], 2),
                      "tout": round(rows[(c, t)]["mean_completion_tokens"], 2), "acc": round(100 * rows[(c, t)]["accuracy"])}
                  for c in ORDER if (c, t) in rows}
    # The defaults must give Finding 4's numbers, as on the page and in the video.
    def per_million(c, t, price):
        p = out[t][c]
        return (p["tin"] * price[0] + p["tout"] * price[1]) / USD_PER_EUR
    got = (round(per_million("S", "T2", LUNA)), round(per_million("Q-off", "T2", ASTRA)))
    assert got == (144, 18325), got
    return out


JS = r"""
(function () {
  var root = document.getElementById('calculator');
  if (!root) return;
  var P = __PRESETS__, D = __DEFAULT__, PRICE = __PRICE__;
  var form = root.querySelector('form'), el = form.elements, nf = new Intl.NumberFormat('en-GB');
  var FIELDS = ['tin', 'tout', 'acc', 'pin', 'pout'];
  function out(name) { return root.querySelector('[data-out="' + name + '"]'); }
  function eur(v) {
    if (v >= 100) return '€' + nf.format(Math.round(v));
    if (v >= 0.1) return '€' + v.toFixed(2);
    return '€' + Number(v.toPrecision(3));
  }
  function options(side, keep) {
    var sel = el[side], list = P[el.task.value];
    sel.innerHTML = '';
    Object.keys(list).forEach(function (c) { sel.add(new Option(list[c].label, c)); });
    sel.value = list[keep] ? keep : Object.keys(list)[side === 'a' ? 0 : 1];
  }
  function fill(side) {
    var p = P[el.task.value][el[side].value];
    el[side + '_tin'].value = p.tin; el[side + '_tout'].value = p.tout; el[side + '_acc'].value = p.acc;
  }
  function read(side) {
    var v = {}, ok = true;
    FIELDS.forEach(function (k) { v[k] = parseFloat(el[side + '_' + k].value); if (!(v[k] >= 0)) ok = false; });
    if (!(v.acc <= 100)) ok = false;
    if (!ok) return null;
    v.task = (v.tin * v.pin + v.tout * v.pout) / 1e6;
    v.ok = v.acc > 0 ? v.task / (v.acc / 100) : Infinity;
    return v;
  }
  function render(share) {
    var n = parseFloat(el.n.value), r = {};
    ['a', 'b'].forEach(function (s) {
      r[s] = n >= 0 ? read(s) : null;
      out(s + '_bill').textContent = r[s] ? eur(n * r[s].task) : '–';
      out(s + '_rest').textContent = !r[s] ? 'Check the inputs: every field needs a number, and correct at most 100.'
        : eur(r[s].task) + ' a task, ' + (isFinite(r[s].ok) ? eur(r[s].ok) + ' a successful task.' : 'and no task succeeded.');
    });
    var a = r.a, b = r.b, v = out('verdict');
    if (!a || !b || !isFinite(a.ok) || !isFinite(b.ok) || !a.ok || !b.ok) v.textContent = '';
    else {
      var hi = b.ok >= a.ok ? 'B' : 'A', lo = hi === 'B' ? 'A' : 'B', x = Math.max(a.ok, b.ok) / Math.min(a.ok, b.ok);
      var d = Math.round((hi === 'B' ? b.acc - a.acc : a.acc - b.acc) * 10) / 10;
      v.textContent = hi + ' costs ' + (x < 10 ? x.toFixed(1) : nf.format(Math.round(x))) + ' times as much as ' + lo + ' per successful task, and gets '
        + (d > 0 ? d + ' points more right.' : d < 0 ? -d + ' points fewer right.' : 'as many right.');
    }
    if (!share) return;  // the address bar changes only once the visitor changes a number
    var q = new URLSearchParams({ task: el.task.value, n: el.n.value, a: el.a.value, b: el.b.value });
    ['a', 'b'].forEach(function (s) {
      var p = P[el.task.value][el[s].value];
      FIELDS.forEach(function (k) {
        var dflt = k === 'pin' ? PRICE[s][0] : k === 'pout' ? PRICE[s][1] : p[k];
        if (parseFloat(el[s + '_' + k].value) !== dflt) q.set(s + '_' + k, el[s + '_' + k].value);
      });
    });
    try { history.replaceState(null, '', '?' + q + '#calculator'); } catch (e) {}
  }
  var q = new URLSearchParams(location.search);
  if (P[q.get('task')]) el.task.value = q.get('task');
  options('a', q.get('a') || D.a); options('b', q.get('b') || D.b); fill('a'); fill('b');
  ['a', 'b'].forEach(function (s) { el[s + '_pin'].value = PRICE[s][0]; el[s + '_pout'].value = PRICE[s][1]; });
  if (q.has('n')) el.n.value = q.get('n');
  ['a', 'b'].forEach(function (s) { FIELDS.forEach(function (k) { if (q.has(s + '_' + k)) el[s + '_' + k].value = q.get(s + '_' + k); }); });
  el.task.addEventListener('change', function () { options('a', el.a.value); options('b', el.b.value); fill('a'); fill('b'); render(true); });
  ['a', 'b'].forEach(function (s) { el[s].addEventListener('change', function () { fill(s); render(true); }); });
  form.addEventListener('input', function (e) { if (e.target.tagName !== 'SELECT') render(true); });
  form.addEventListener('submit', function (e) { e.preventDefault(); });
  root.querySelector('[data-copy]').addEventListener('click', function () {
    var btn = this;
    navigator.clipboard.writeText(location.href).then(function () { btn.textContent = 'Link copied'; },
      function () { btn.textContent = 'Copy the address bar instead'; });
    setTimeout(function () { btn.textContent = 'Copy a link to these numbers'; }, 2500);
  });
  root.hidden = false;
  render();
}());
"""


def side(s: str, title: str, p: dict, price: tuple) -> str:
    def f(k, label, value, step):
        return (f'<label>{label}<input name="{s}_{k}" type="number" inputmode="decimal" min="0" step="{step}" '
                f'value="{value:g}"></label>')
    return f'''<div class="calc__side">
              <p class="calc__legend">{title}</p>
              <label>Model and setting<select name="{s}"></select></label>
              <div class="calc__row">{f("tin", "Input tokens", p["tin"], "any")}{f("tout", "Output tokens", p["tout"], "any")}{f("acc", "Correct, %", p["acc"], "any")}</div>
              <div class="calc__row">{f("pin", "€ per million input", price[0], "any")}{f("pout", "€ per million output", price[1], "any")}</div>
              <div class="calc__result">
                <p class="kicker">Your bill</p>
                <p class="calc__big" data-out="{s}_bill"></p>
                <p data-out="{s}_rest"></p>
              </div>
            </div>'''


def block(summary: dict) -> str:
    pre = presets(summary)
    price = {s: [round(x / USD_PER_EUR, 6) for x in pr] for s, pr in (("a", LUNA), ("b", ASTRA))}
    task_opts = "".join(f'<option value="{t}"{" selected" if t == DEFAULT["task"] else ""}>{html.escape(n)}</option>' for t, n in TASKS.items())
    js = (JS.replace("__PRESETS__", json.dumps(pre, ensure_ascii=False, separators=(",", ":")))
            .replace("__DEFAULT__", json.dumps(DEFAULT)).replace("__PRICE__", json.dumps(price)).strip())
    t = DEFAULT["task"]
    return f'''<!-- CALCULATOR START: written by yt-experiment-001/analysis/make_calculator.py; edit there, not here -->
  <section class="section section--light" aria-labelledby="calc-title">
    <div class="container">
      <div class="section-head">
        <p class="kicker">Your own numbers</p>
        <h2 id="calc-title">Your bill is price × tokens per task × tasks.</h2>
        <p class="lead">Tokens per task and the success rate decide what one answer that works costs you.</p>
      </div>
      <p class="cite">Cost per task = (input tokens × input price + output tokens × output price) ÷ 1 million. Cost per successful task = cost per task ÷ share correct.</p>
      <div class="calc" id="calculator" hidden>
        <h3>Compare two settings with your prices.</h3>
        <p class="calc__intro">Pick a task and two settings from the test. Each fills in its measured tokens and success rate. Then put in your provider’s prices.</p>
        <form novalidate>
          <div class="calc__row">
            <label>Task<select name="task">{task_opts}</select></label>
            <label>Tasks a month<input name="n" type="number" inputmode="decimal" min="0" step="1" value="{DEFAULT["n"]}"></label>
          </div>
          <div class="calc__pair">
            {side("a", "Setting A", pre[t][DEFAULT["a"]], price["a"])}
            {side("b", "Setting B", pre[t][DEFAULT["b"]], price["b"])}
          </div>
        </form>
        <p class="calc__verdict" data-out="verdict" aria-live="polite"></p>
        <p class="caption">Starting prices: OpenAI’s list prices of 8 November 2026 for gpt-6-luna (A) and gpt-6-astra (B), at 1 € = {USD_PER_EUR} $, as in Finding 4. Tokens and success rates are those of the open models in this test, so pricing a commercial model with them is a projection.</p>
        <button class="btn btn--outline btn--small" type="button" data-copy>Copy a link to these numbers</button>
      </div>
    </div>
  </section>
  <script>
{js}
  </script>
  <!-- CALCULATOR END -->'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(HERE.parent / "results"))
    ap.add_argument("--site", required=True, help="path to the kalunho.com repository")
    a = ap.parse_args()
    summary = json.loads((Path(a.results) / "summary.json").read_text(encoding="utf-8"))
    page = Path(a.site) / PAGE
    text = page.read_text(encoding="utf-8")
    pat = re.compile(r"<!-- CALCULATOR START.*?<!-- CALCULATOR END -->", re.S)
    if not pat.search(text):
        sys.exit(f"no <!-- CALCULATOR START --> ... <!-- CALCULATOR END --> markers in {page}: add them once where the section goes")
    page.write_text(pat.sub(lambda _: block(summary), text), encoding="utf-8", newline="\n")
    print(f"wrote the calculator into {page}")


if __name__ == "__main__":
    main()
