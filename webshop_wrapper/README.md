# `webshop_wrapper`

Bridge between [princeton-nlp/WebShop](https://github.com/princeton-nlp/WebShop)
and the existing `browsergym_wrapper` pipeline.  Lets us reuse all
three schema heads (heuristic / vision VLM / OmniParser-v2), the
multi-turn tool registry, and the 116 anti-thrash / anti-repeat /
search-first regression tests *without modification* — by pointing
BrowserGym's Chromium-based AXTree extractor at WebShop's Flask
server and only replacing the goal injection + reward function.

> **Status (2026-05-03 5:38 PM, post-spike):** 🟢 **AXTree spike PASSED.**
> All 5 representative pages (search, results, item, description,
> done) come back with **100% bid coverage** and clean accessibility
> roles (`button`, `radio`, `link`, `textbox`, `heading`, `image`).
> The existing `browsergym_wrapper.heuristic.obs_to_schema` produces
> 0 entities on the terminal `done_page` (expected) and 2-12 entities
> on every other page.  Decision: **proceed to full install.**

> **Why this exists.** VWA was archived on 2026-05-03 after 10 distinct
> non-research bugs across one week; see
> [`legacy/visualwebarena/README.md`](../legacy/visualwebarena/README.md).
> WebShop has dramatically simpler infra (single Flask process, no
> Docker fleet, no per-step LLM judge, rule-based reward) but does not
> natively expose AXTree / `bid` / `extra_element_properties`.  The
> bridge approach ("treat WebShop as a `browsergym/openended` URL")
> sidesteps both the WebShop wrapper-rewrite cost and the VWA infra
> cost.

---

## What's inside

| File | What it does |
|------|--------------|
| `__init__.py`     | Public API — `WebShopTask`, `register_webshop_tasks`. |
| `task.py`         | `WebShopTask(OpenEndedTask)` — goal injected via `/fixed_<idx>`, reward read from `/__bridge/session/<id>` on `validate()`.  Registers as `browsergym/webshop.<idx>`. |
| `server.py`       | `start_webshop_server(mode={stub,full,external})` + `ServerHandle.stop()`.  Subprocess management mirroring `env_wrappers/subprocess_env.py`. |
| `stub_app.py`     | Mini Flask app serving WebShop's *real* HTML templates with a 5-product fake catalogue.  Zero install (only Flask required).  Used by the AXTree smoke. |
| `smoke_axtree.py` | The make-or-break test: boots stub server + `browsergym/webshop.0`, walks 5 pages, reports per-page `interactive_nodes`, `bid_coverage`, `schema_entities`.  Exits 0 on PASS. |
| `templates/*.html`| Verbatim copy of `princeton-nlp/WebShop/web_agent_site/templates/`.  Stays in lock-step with upstream — used by `stub_app.py`. |
| `static/`         | Auto-created by stub_app.py, holds an empty `style.css` to silence Playwright's wait-for-load on the missing static asset. |

---

## Three install levels

### Level 1 — stub mode (no install, ~30 s wall-clock)

For verifying the bridge works at all, e.g. before paying the full
install cost or in CI.  Only needs Flask + the existing `browsergym`
conda env.

```bash
conda activate browsergym
cd Multi-hop-Reasoning-VLM-Agent
python -m webshop_wrapper.smoke_axtree
```

Expected output (verbatim from the spike on 2026-05-03):

```
[info] spawning stub server on :3000
[info] registered 5 envs; using browsergym/webshop.0
  [PASS] search_page    interactive=2   bid_cov=100%  schema_ents=3   (..., button=1, textbox=1)
  [PASS] results_page   interactive=4   bid_cov=100%  schema_ents=6   (..., button=2, link=2)
  [PASS] item_page      interactive=12  bid_cov=100%  schema_ents=12  (..., button=6, radio=6)
  [PASS] description_page interactive=2   bid_cov=100%  schema_ents=2   (..., button=2)
  [PASS] done_page      interactive=0   bid_cov=100%  schema_ents=0   (terminal page)

VERDICT: PASS  (all 5 pages clean)
```

If this fails on your machine, **stop and don't proceed to Level 2** —
something is wrong with the BrowserGym install (Chromium, Playwright,
or the AXTree extractor), not with WebShop.

### Level 2 — full mode (lite install, ~10 min, ~50 MB on disk)

Real WebShop with the 1k-product split, BM25 search via `rank_bm25`
(no pyserini / Java / faiss / spaCy `en_core_web_lg`).  See
[`install/install_webshop.sh`](../install/install_webshop.sh).

```bash
bash install/install_webshop.sh        # default: lite mode
source cold_start/webshop_env.sh        # exports WEBSHOP_BASE_URL etc.

# Boot the server in a separate terminal:
conda activate webshop
cd $WEBSHOP_DIR && python -m web_agent_site.app

# Then run the same smoke against the real server:
conda activate browsergym
cd Multi-hop-Reasoning-VLM-Agent
python -m webshop_wrapper.smoke_axtree --base-url $WEBSHOP_BASE_URL
```

The lite-mode patch is idempotent (re-running `install_webshop.sh`
detects `_LITE_BM25_PATCH_APPLIED` and skips), so you can rebuild
without nuking `$WEBSHOP_DIR` first.

### Level 3 — full mode (Lucene + BERT ranker, ~30 min, ~3 GB on disk)

Reproduces the published WebShop pass-rate numbers exactly.  Pulls in
pyserini (Java 11 + Lucene), faiss, the BERT search ranker, and
spaCy's `en_core_web_lg`.  Use only if you're comparing against a
specific WebShop paper baseline.

```bash
WEBSHOP_LITE=0 bash install/install_webshop.sh
```

---

## How the bridge works

### Goal injection

WebShop's Flask app uses `session_id="fixed_<idx>"` URLs to pin a
specific goal: `/fixed_42` always selects goal #42 from the
`get_goals()` list (post-shuffle with seed=233).  The
`WebShopTask.setup()` method sets `start_url = f"{base_url}/fixed_{goal_idx}"`,
so when BrowserGym does its initial `page.goto(start_url)`, WebShop
auto-creates the session and returns the search page with the goal
text at the top.

### Goal text extraction

Before the first `page.goto()`, the task hits
`{base_url}/__bridge/session/fixed_<idx>` to pull the
`instruction_text` field out as JSON.  This endpoint is part of the
stub server natively, and gets monkey-patched into the real WebShop
`app.py` by `install/install_webshop.sh` step 5.  Bypasses the need to
parse the goal text out of rendered HTML.

### Reward signal

`WebShopTask.validate(page, chat_messages)` checks if the current
`page.url` contains `/done/<session_id>`; if so, fetches
`/__bridge/session/<id>` and returns the `reward` field.  Reward
calculation itself happens server-side in WebShop's
`web_agent_site/app.py:done()` route (the existing rule-based attribute
matcher in `web_agent_site/engine/goal.py:get_reward`), so the bridge
inherits whatever reward function WebShop ships.

In stub mode the reward is a simplified asin-match-with-options bonus
(`stub_app.py:_reward`) — sufficient for AXTree smoke testing,
**not** sufficient for actual training/eval (use full mode for that).

### Action space

Native BrowserGym actions (`click("bid_47")`, `fill("bid_3", "...")`,
`press("Enter")`, etc.) — *not* WebShop's `search[query]` /
`click[<text>]` action language.  The agent never sees the WebShop
action format, so the existing tool registry, anti-thrash / anti-repeat
overrides, and search-first heuristic in
[`cold_start/generate_cold_start_actor_browsergym.py`](../cold_start/generate_cold_start_actor_browsergym.py)
all keep working unmodified.  This is the main reason we picked the
bridge approach over wrapping WebShop's native gym env.

---

## Spike result (2026-05-03)

**Question.** Does Chromium's accessibility tree (as exposed via
BrowserGym's `obs["axtree_object"]` + `obs["extra_element_properties"]`)
pick up clean interactive roles + bids on WebShop's HTML pages, or
does it collapse everything into `role="generic"` with no addressable
bids?

**Method.** `webshop_wrapper.stub_app` (5-product fake catalogue
served through WebShop's *real* templates) + `webshop_wrapper.smoke_axtree`
walking the 5 representative page types.

**Result.** 100% bid coverage on every interactive node, role
identification matches exactly what we'd expect from semantic HTML:

| Page             | Interactive | Bid cov | Schema ents | Roles seen |
|---              | ---:        | ---:    | ---:        | --- |
| search_page      | 2           | 100%    | 3           | textbox, button |
| results_page     | 4           | 100%    | 6           | button×2 (Back/Next), link×2 (products) |
| item_page        | 12          | 100%    | 12          | button×6 (Description, Features, Reviews, Buy Now, Back, Prev), radio×6 (size/color options) |
| description_page | 2           | 100%    | 2           | button×2 |
| done_page        | 0           | (n/a)   | 0           | terminal — only headings + static text by design |

**Why this matters.** It means the 3-head schema generator
(`browsergym_wrapper.heuristic` / `.adapter` / `.grounding`) and the
multi-turn tool registry (`browsergym_wrapper.tools`) all work on
WebShop pages without any wrapper changes.  Specifically:

- The radio-button option selector (`size = 20oz | 32oz | 40oz`,
  `color = black | blue | silver`) — historically the page where naive
  HTML scrapers struggle most — comes through as 6 distinct
  `role="radio"` nodes with `name`s set to the option labels and
  `properties.checked` reflecting the current selection.  This is
  exactly the structure `_extract_entities` in `heuristic.py` is built
  to consume.
- The `Buy Now` button is a `role="button"` (not `role="generic"`
  inside a styled div), so the terminal-action plumbing in
  `tests/test_browsergym_terminal_actions.py` will recognise it.

**Decision.** Proceed to Level 2 install.  Total estimated cost: **0.5
day** for the lite install + bridge polish, vs. **1+ week** for VWA
(per `legacy/visualwebarena/README.md`).

---

## Limitations / known caveats

1. **Site diversity** — WebShop is a single shopping domain.  If your
   paper story requires multi-site cross-domain transfer, this bridge
   won't carry the story alone — pair it with AssistantBench (already
   primary in `main`) or MiniWoB++ for non-shopping coverage.

2. **Action space mismatch with the published WebShop literature.**
   We use BrowserGym's bid-based actions (`click(bid="47")`), not
   WebShop's text-action format (`click[Buy Now]`).  Pass-rate
   comparisons against papers using the native action space will be
   off-by-an-action-cost; calibrate before publishing.

3. **Reward calibration** — only validated on the stub.  Before
   training, run a 50-task smoke through the full-mode server and
   confirm reward distribution matches the published WebShop reward
   histogram (`mean ≈ 0.5`, mass at `1.0` for exact-match purchases).

4. **Lite-mode search differs from full-mode.** Lite uses `rank_bm25`
   over `Title+category+query+product_category+asin`; full uses
   pyserini's BM25 over a Lucene-tokenised richer text field, and
   optionally a BERT cross-encoder re-ranker on top.  Expected
   <2 pp pass-rate delta per the WebShop paper §6.3, well within our
   cold-start noise floor — but worth a side-by-side smoke before any
   numbers ship.

5. **5-product stub != production goal pool.** The stub registers 5
   `browsergym/webshop.0..4` envs; full mode can register up to ~12k
   (1k-product split) or ~50k (full split).  Set
   `WEBSHOP_NUM_GOALS=N` env var before
   `register_webshop_tasks(num_goals=N)` to scale.

---

## Tests

The spike ships with one regression test (the AXTree smoke itself).
Add it to CI by symlinking into `tests/`:

```bash
ln -s ../webshop_wrapper/smoke_axtree.py tests/test_webshop_axtree.py
```

The test takes ~7 s wall-clock on the dev box (most of it Chromium
startup; the stub itself starts in <1 s and serves each page in <50 ms).

---

## Cross-refs

- [`browsergym_wrapper/README.md`](../browsergym_wrapper/README.md) — three-head schema generator we reuse
- [`legacy/visualwebarena/README.md`](../legacy/visualwebarena/README.md) — why we dropped VWA, what survived in `main`
- [`env_wrappers/subprocess_env.py`](../env_wrappers/subprocess_env.py) — same subprocess-isolation pattern `server.py` mirrors
- [`install/install_webshop.sh`](../install/install_webshop.sh) — full Level 2/3 installer
- [`install/webshop.environment.yml`](../install/webshop.environment.yml) — webshop conda env spec
