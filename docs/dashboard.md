# Dashboard notes

`dashboard/` is a static page with no build step. It fetches
`dashboard/data/runs.json` — the same document `llmft eval` writes — and renders
it. Serve it with:

```
make dashboard   # python -m http.server 8080 --directory dashboard
```

Opening `index.html` directly off the filesystem will not work: `fetch()` on a
`file://` URL is blocked by every current browser. The page says so when it
fails rather than sitting blank.

## Refreshing the data

```
llmft eval --config configs/eval.yaml
```

`eval.dashboard_data` in the config controls where the copy lands. Set it to
`null` if you only want the files under `reports/`.

The checked-in `dashboard/data/runs.json` holds the reference run's numbers so
the page renders on a fresh clone. `python scripts/seed_dashboard.py` rewrites
it through the same `build_report` path the harness uses, so the bundled file
can never drift from the real schema.

## Charts

| Chart | Encoding | Why it is the way it is |
|---|---|---|
| Metric by training step | multi-series line | All the metrics share a 0–1 scale, so they share one axis. |
| Train/validation loss | separate card, own axis | Loss is not on the metric scale. Two scales on one chart would be a dual axis, which is never worth it. |
| Score by checkpoint | zero-anchored bars | Bar length encodes magnitude; a truncated baseline would make a 10-point gain look like a 3x one. |
| All checkpoints | table | The exact numbers, and the accessible fallback for the charts. |

## Colours

The five series colours in `assets/styles.css` (`--series-1` … `--series-5`)
are a palette validated for colour-vision deficiency — adjacent pairs clear a
CVD ΔE of 8 and a normal-vision ΔE of 15 in both light and dark mode. They are
assigned to metrics **by name**, never by position in the report, so hiding one
metric never repaints the others.

Three of the light-mode colours sit below 3:1 contrast against the surface. The
mitigation is that identity never rests on colour alone: every line carries a
direct label at its end (dropped below 560px, where the legend takes over), a
legend is always present, and the table view repeats every number.

If you change a `--series-*` value, re-check the palette rather than eyeballing
it — the failure mode is invisible to anyone with normal colour vision.

## Browser support

Plain ES5-style JavaScript in one IIFE, no modules, no bundler, no runtime
dependencies. Charts are hand-built SVG. Everything works from a stock
`http.server`, which is the point — the dashboard has to be openable by someone
who just cloned the repo.
