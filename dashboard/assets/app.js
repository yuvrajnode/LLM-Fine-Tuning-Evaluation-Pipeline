/* Dashboard behaviour.
 *
 * Plain ES modules-free JavaScript on purpose: the page is opened straight from
 * the filesystem or served by `python -m http.server`, and a build step for four
 * charts is not worth the maintenance. Charts are hand-built SVG - no library -
 * which keeps the repo dependency-free and the markup inspectable.
 *
 * Everything is driven by the report JSON the eval harness writes. If a field is
 * missing the chart that needs it is skipped rather than crashing the page.
 */

(function () {
  "use strict";

  var DATA_URL = "data/runs.json";

  // Fixed slot order, matching the validated palette in styles.css. Metrics are
  // assigned by name, never by index in the report, so hiding one metric never
  // repaints the others.
  var SERIES = {
    exact_match: { color: "var(--series-1)", label: "Exact match" },
    token_f1: { color: "var(--series-2)", label: "Token F1" },
    rouge_l: { color: "var(--series-3)", label: "ROUGE-L" },
    contains_answer: { color: "var(--series-4)", label: "Contains answer" },
    length_ratio: { color: "var(--series-5)", label: "Length ratio" }
  };

  // Higher is not better for these, so they never win "best checkpoint" and are
  // off by default in the line chart - they sit on a different scale.
  var NON_DIRECTIONAL = ["length_ratio"];

  var SVG_NS = "http://www.w3.org/2000/svg";

  var state = {
    report: null,
    activeMetrics: [],
    rows: []
  };

  /* ------------------------------------------------------------------ utils */

  function el(tag, attrs, text) {
    var node = document.createElement(tag);
    applyAttrs(node, attrs);
    if (text != null) node.textContent = text;
    return node;
  }

  function svg(tag, attrs, text) {
    var node = document.createElementNS(SVG_NS, tag);
    applyAttrs(node, attrs);
    if (text != null) node.textContent = text;
    return node;
  }

  function applyAttrs(node, attrs) {
    if (!attrs) return;
    Object.keys(attrs).forEach(function (key) {
      if (attrs[key] == null) return;
      if (key === "class") node.setAttribute("class", attrs[key]);
      else node.setAttribute(key, attrs[key]);
    });
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function fmt(value, digits) {
    if (value == null || isNaN(value)) return "—";
    return Number(value).toFixed(digits == null ? 4 : digits);
  }

  function metricLabel(name) {
    return (SERIES[name] && SERIES[name].label) || name;
  }

  function metricColor(name) {
    return (SERIES[name] && SERIES[name].color) || "var(--text-secondary)";
  }

  function duration(seconds) {
    if (seconds == null) return "—";
    var s = Math.round(seconds);
    if (s < 90) return s + "s";
    var m = Math.floor(s / 60);
    if (m < 60) return m + "m " + (s % 60) + "s";
    return Math.floor(m / 60) + "h " + (m % 60) + "m";
  }

  /* --------------------------------------------------------------- scaling */

  function makeScale(domain, range) {
    var d0 = domain[0];
    var d1 = domain[1];
    var span = d1 - d0 || 1;
    return function (value) {
      return range[0] + ((value - d0) / span) * (range[1] - range[0]);
    };
  }

  /* Round a domain outward to readable tick boundaries, with a little padding
     so the top line is never welded to the frame. */
  function niceDomain(values, opts) {
    opts = opts || {};
    var finite = values.filter(function (v) {
      return v != null && isFinite(v);
    });
    if (!finite.length) return [0, 1];

    var min = Math.min.apply(null, finite);
    var max = Math.max.apply(null, finite);
    if (min === max) {
      min -= 0.5;
      max += 0.5;
    }
    var pad = (max - min) * (opts.pad == null ? 0.12 : opts.pad);
    min = opts.zero ? 0 : min - pad;
    max = max + pad;

    var step = tickStep(min, max, opts.ticks || 5);
    return [Math.floor(min / step) * step, Math.ceil(max / step) * step];
  }

  function tickStep(min, max, count) {
    var raw = (max - min) / Math.max(count, 1);
    var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var norm = raw / mag;
    var nice = norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1;
    return nice * mag;
  }

  function ticksFor(domain, count) {
    var step = tickStep(domain[0], domain[1], count);
    var out = [];
    for (var v = domain[0]; v <= domain[1] + step / 2; v += step) {
      out.push(Math.abs(v) < step / 1e6 ? 0 : v);
    }
    return out;
  }

  function tickFormat(value, domain) {
    var span = Math.abs(domain[1] - domain[0]);
    var digits = span >= 10 ? 0 : span >= 1 ? 1 : 2;
    return value.toFixed(digits);
  }

  /* --------------------------------------------------------------- tooltip */

  function attachTooltip(container) {
    var tip = el("div", { class: "tooltip", role: "status" });
    container.appendChild(tip);
    return {
      show: function (x, y, html) {
        tip.innerHTML = html;
        tip.style.left = x + "px";
        tip.style.top = y + "px";
        tip.setAttribute("data-open", "true");
      },
      hide: function () {
        tip.setAttribute("data-open", "false");
      }
    };
  }

  function tooltipRows(entries) {
    return entries
      .map(function (entry) {
        var swatch = entry.color
          ? '<span class="tip-swatch" style="background:' + entry.color + '"></span>'
          : "<span></span>";
        return (
          "<dt>" +
          swatch +
          "<span>" +
          entry.label +
          '</span><span class="tip-value">' +
          entry.value +
          "</span></dt>"
        );
      })
      .join("");
  }

  /* ------------------------------------------------------------ line charts */

  function drawLineChart(container, opts) {
    clear(container);
    var width = Math.max(container.clientWidth || 640, 320);
    var height = opts.height || 300;
    // Below ~560px there is no room for end labels beside the plot; the legend
    // underneath carries identity instead.
    var showEndLabels = width >= 560;
    var margin = {
      top: 16,
      right: showEndLabels ? opts.rightGutter || 104 : 14,
      bottom: 38,
      left: 46
    };
    var innerW = width - margin.left - margin.right;
    var innerH = height - margin.top - margin.bottom;

    var series = opts.series.filter(function (s) {
      return s.points.length > 0;
    });
    if (!series.length) {
      container.appendChild(el("p", { class: "note" }, "Nothing to plot for this selection."));
      return;
    }

    var xs = [];
    var ys = [];
    series.forEach(function (s) {
      s.points.forEach(function (p) {
        xs.push(p.x);
        ys.push(p.y);
      });
    });

    var xDomain = [Math.min.apply(null, xs), Math.max.apply(null, xs)];
    var yDomain = niceDomain(ys, { ticks: 5, pad: 0.14, zero: opts.zero });
    var x = makeScale(xDomain, [0, innerW]);
    var y = makeScale(yDomain, [innerH, 0]);

    var root = svg("svg", {
      viewBox: "0 0 " + width + " " + height,
      width: width,
      height: height,
      role: "img",
      "aria-label": opts.ariaLabel || "line chart"
    });
    var plot = svg("g", { transform: "translate(" + margin.left + "," + margin.top + ")" });
    root.appendChild(plot);

    // gridlines + y ticks
    ticksFor(yDomain, 5).forEach(function (value) {
      var yy = y(value);
      plot.appendChild(svg("line", { class: "grid-line", x1: 0, x2: innerW, y1: yy, y2: yy }));
      plot.appendChild(
        svg(
          "text",
          { class: "tick-label", x: -10, y: yy, "text-anchor": "end", "dominant-baseline": "middle" },
          tickFormat(value, yDomain)
        )
      );
    });

    // x axis
    plot.appendChild(
      svg("line", { class: "axis-line", x1: 0, x2: innerW, y1: innerH, y2: innerH })
    );
    var xTicks = uniqueSteps(series[0].points.map(function (p) { return p.x; }), 6);
    xTicks.forEach(function (value) {
      plot.appendChild(
        svg(
          "text",
          { class: "tick-label", x: x(value), y: innerH + 18, "text-anchor": "middle" },
          String(value)
        )
      );
    });
    plot.appendChild(
      svg(
        "text",
        { class: "axis-title", x: innerW / 2, y: innerH + 34, "text-anchor": "middle" },
        opts.xTitle || "training step"
      )
    );

    // series
    var endLabels = [];
    series.forEach(function (s) {
      var d = s.points
        .map(function (p, i) {
          return (i === 0 ? "M" : "L") + x(p.x) + "," + y(p.y);
        })
        .join(" ");
      plot.appendChild(svg("path", { class: "series-line", d: d, stroke: s.color }));

      // Direct label at the line end. With three or more series this is what
      // keeps identity off colour alone, which the palette's contrast warning
      // requires in light mode.
      var last = s.points[s.points.length - 1];
      endLabels.push({ y: y(last.y), x: x(last.x) + 10, text: s.label, color: s.color });
    });

    // Converging series put their labels on top of each other, so push them
    // apart before drawing. Nudging only - each label stays nearest its line.
    spreadLabels(endLabels, 15, innerH);
    (showEndLabels ? endLabels : []).forEach(function (label) {
      plot.appendChild(
        svg(
          "text",
          {
            class: "end-label",
            x: label.x,
            y: label.y,
            "dominant-baseline": "middle",
            fill: label.color
          },
          label.text
        )
      );
    });

    // hover layer: one crosshair snapped to the nearest x
    var crosshair = svg("line", {
      class: "crosshair",
      y1: 0,
      y2: innerH,
      x1: 0,
      x2: 0,
      opacity: 0
    });
    plot.appendChild(crosshair);

    var dots = svg("g", { opacity: 0 });
    plot.appendChild(dots);

    var hit = svg("rect", { class: "hit", x: 0, y: 0, width: innerW, height: innerH });
    plot.appendChild(hit);
    container.appendChild(root);

    var tip = attachTooltip(container);
    var stepsAvailable = series[0].points.map(function (p) { return p.x; });

    hit.addEventListener("mousemove", function (event) {
      var bounds = root.getBoundingClientRect();
      // Guard against the SVG being rendered at a different size than its
      // viewBox (a resize between draws), which would offset every reading.
      var scale = bounds.width / width || 1;
      var localX = (event.clientX - bounds.left) / scale - margin.left;
      var target = nearest(stepsAvailable, xDomain, localX, innerW);

      crosshair.setAttribute("x1", x(target));
      crosshair.setAttribute("x2", x(target));
      crosshair.setAttribute("opacity", 1);

      clear(dots);
      var entries = [];
      series.forEach(function (s) {
        var point = s.points.filter(function (p) { return p.x === target; })[0];
        if (!point) return;
        dots.appendChild(
          svg("circle", { class: "series-dot", cx: x(point.x), cy: y(point.y), r: 4.5, fill: s.color })
        );
        entries.push({ label: s.label, color: s.color, value: fmt(point.y, s.digits) });
      });
      dots.setAttribute("opacity", 1);

      tip.show(
        (margin.left + x(target)) * scale,
        (margin.top + 8) * scale,
        '<div class="tip-title">' + (opts.pointTitle || "step ") + target + "</div><dl>" +
          tooltipRows(entries) +
          "</dl>"
      );
    });

    hit.addEventListener("mouseleave", function () {
      crosshair.setAttribute("opacity", 0);
      dots.setAttribute("opacity", 0);
      tip.hide();
    });
  }

  function nearest(values, domain, pixelX, innerW) {
    var ratio = Math.min(Math.max(pixelX / innerW, 0), 1);
    var target = domain[0] + ratio * (domain[1] - domain[0]);
    return values.reduce(function (best, value) {
      return Math.abs(value - target) < Math.abs(best - target) ? value : best;
    }, values[0]);
  }

  /* Resolve overlapping direct labels: sort by y, then walk down enforcing a
     minimum gap, and walk back up if the last one has been pushed off. */
  function spreadLabels(labels, minGap, maxY) {
    labels.sort(function (a, b) {
      return a.y - b.y;
    });
    for (var i = 1; i < labels.length; i++) {
      if (labels[i].y - labels[i - 1].y < minGap) {
        labels[i].y = labels[i - 1].y + minGap;
      }
    }
    var overflow = labels.length ? labels[labels.length - 1].y - maxY : 0;
    if (overflow > 0) {
      for (var j = labels.length - 1; j >= 0; j--) {
        labels[j].y -= overflow;
        if (j > 0 && labels[j].y - labels[j - 1].y >= minGap) break;
      }
    }
    return labels;
  }

  function uniqueSteps(values, count) {
    if (values.length <= count) return values;
    var stride = Math.ceil(values.length / count);
    return values.filter(function (_, i) {
      return i % stride === 0 || i === values.length - 1;
    });
  }

  /* ------------------------------------------------------------- bar charts */

  function drawBarChart(container, opts) {
    clear(container);
    var width = Math.max(container.clientWidth || 520, 320);
    var rows = opts.rows;
    var height = opts.height || 300;
    var margin = { top: 16, right: 16, bottom: 52, left: 46 };
    var innerW = width - margin.left - margin.right;
    var innerH = height - margin.top - margin.bottom;

    // Bar length encodes magnitude, so the axis starts at zero. A truncated
    // baseline would make a 10-point gain look like a 3x one.
    var values = rows.map(function (r) { return r.value; });
    var yDomain = niceDomain(values, { ticks: 5, pad: 0.08, zero: true });
    var y = makeScale(yDomain, [innerH, 0]);

    // 2px surface gap between neighbouring bars, per the mark spec.
    var slot = innerW / rows.length;
    var barW = Math.max(slot - 4, 3);

    var root = svg("svg", {
      viewBox: "0 0 " + width + " " + height,
      width: width,
      height: height,
      role: "img",
      "aria-label": opts.ariaLabel || "bar chart"
    });
    var plot = svg("g", { transform: "translate(" + margin.left + "," + margin.top + ")" });
    root.appendChild(plot);

    ticksFor(yDomain, 5).forEach(function (value) {
      var yy = y(value);
      plot.appendChild(svg("line", { class: "grid-line", x1: 0, x2: innerW, y1: yy, y2: yy }));
      plot.appendChild(
        svg(
          "text",
          { class: "tick-label", x: -10, y: yy, "text-anchor": "end", "dominant-baseline": "middle" },
          tickFormat(value, yDomain)
        )
      );
    });

    plot.appendChild(svg("line", { class: "axis-line", x1: 0, x2: innerW, y1: innerH, y2: innerH }));

    container.appendChild(root);
    var tip = attachTooltip(container);

    rows.forEach(function (row, i) {
      var bx = i * slot + (slot - barW) / 2;
      var by = y(row.value);
      var group = svg("g", { class: "bar" });

      group.appendChild(
        svg("path", {
          d: roundedTopBar(bx, by, barW, innerH - by, 4),
          fill: row.highlight ? opts.color : opts.mutedColor || opts.color,
          opacity: row.highlight ? 1 : 0.5
        })
      );
      plot.appendChild(group);

      group.addEventListener("mouseenter", function () {
        var bounds = root.getBoundingClientRect();
        var scale = bounds.width / width || 1;
        tip.show(
          (margin.left + bx + barW / 2) * scale,
          (margin.top + by) * scale,
          '<div class="tip-title">' + row.label + "</div><dl>" +
            tooltipRows([
              { label: opts.metricLabel, color: opts.color, value: fmt(row.value, 4) },
              { label: "step", value: String(row.step) }
            ]) +
            "</dl>"
        );
      });
      group.addEventListener("mouseleave", tip.hide);

      if (row.showTick) {
        plot.appendChild(
          svg(
            "text",
            {
              class: "tick-label",
              x: bx + barW / 2,
              y: innerH + 16,
              "text-anchor": "end",
              transform: "rotate(-40," + (bx + barW / 2) + "," + (innerH + 16) + ")"
            },
            row.tickLabel
          )
        );
      }
    });

    // The winner gets a value label so the answer is readable without hovering.
    var best = rows.filter(function (r) { return r.highlight; })[0];
    if (best) {
      var bi = rows.indexOf(best);
      plot.appendChild(
        svg(
          "text",
          {
            class: "end-label",
            x: bi * slot + slot / 2,
            y: y(best.value) - 8,
            "text-anchor": "middle",
            fill: opts.color
          },
          fmt(best.value, 3)
        )
      );
    }
  }

  function roundedTopBar(x, y, w, h, r) {
    var radius = Math.min(r, w / 2, Math.max(h, 0));
    if (h <= 0) return "M" + x + "," + y + "h" + w;
    return (
      "M" + x + "," + (y + h) +
      "V" + (y + radius) +
      "a" + radius + "," + radius + " 0 0 1 " + radius + ",-" + radius +
      "h" + (w - 2 * radius) +
      "a" + radius + "," + radius + " 0 0 1 " + radius + "," + radius +
      "V" + (y + h) +
      "Z"
    );
  }

  /* ----------------------------------------------------------- render parts */

  function renderMasthead(report) {
    var run = report.run || {};
    document.getElementById("run-title").textContent = run.name || "Checkpoint sweep";
    document.title = (run.name || "Checkpoint sweep") + " · llmft";

    var meta = document.getElementById("run-meta");
    clear(meta);
    meta.appendChild(document.createTextNode("Base "));
    meta.appendChild(el("code", null, run.base_model || "unknown"));
    meta.appendChild(
      document.createTextNode(
        " · " + (run.num_examples || 0) + " eval examples · LoRA r=" +
        ((run.lora && run.lora.r) || "?") + " · generated " + (report.generated_at || "—")
      )
    );
  }

  function renderTiles(report) {
    var host = document.getElementById("tiles");
    clear(host);

    var summary = report.summary || {};
    var timing = report.timing || {};
    var best = summary.best;

    var tiles = [];

    if (best) {
      var deltaPct = summary.delta_pct;
      tiles.push({
        label: "Best " + metricLabel(summary.primary_metric),
        value: fmt(best.score, 4),
        sub:
          deltaPct == null
            ? "no base model in this sweep"
            : '<span class="delta ' + (deltaPct >= 0 ? "up" : "down") + '">' +
              (deltaPct >= 0 ? "▲" : "▼") + " " + Math.abs(deltaPct).toFixed(1) +
              "%</span> vs base " + fmt(summary.baseline && summary.baseline.score, 4)
      });
      tiles.push({
        label: "Best checkpoint",
        value: "step " + best.step,
        sub: best.name + (best.stage ? " · " + best.stage : "")
      });
    }

    tiles.push({
      label: "Checkpoints scored",
      value: String((report.checkpoints || []).length),
      sub: timing.from_cache
        ? timing.evaluated + " evaluated, " + timing.from_cache + " from cache"
        : "all evaluated this run"
    });

    tiles.push({
      label: "Sweep wall clock",
      value: duration(timing.wall_seconds),
      sub:
        duration(timing.mean_seconds_per_checkpoint) +
        " per checkpoint" +
        (timing.estimated_seconds_saved
          ? " · " + duration(timing.estimated_seconds_saved) + " saved by cache"
          : "")
    });

    tiles.forEach(function (tile) {
      var node = el("article", { class: "tile" });
      node.appendChild(el("p", { class: "tile-label" }, tile.label));
      node.appendChild(el("p", { class: "tile-value" }, tile.value));
      if (tile.sub) {
        var sub = el("p", { class: "tile-sub" });
        sub.innerHTML = tile.sub;
        node.appendChild(sub);
      }
      host.appendChild(node);
    });
  }

  function renderMetricFilters(report) {
    var host = document.getElementById("metric-filters");
    // keep the "Metrics" label, drop previously rendered chips
    Array.prototype.slice.call(host.querySelectorAll("button")).forEach(function (b) {
      host.removeChild(b);
    });

    (report.tasks || []).forEach(function (task) {
      var active = state.activeMetrics.indexOf(task) !== -1;
      var chip = el("button", {
        class: "chip",
        type: "button",
        "aria-pressed": String(active)
      });
      chip.style.color = metricColor(task);
      chip.appendChild(el("span", { class: "swatch" }));
      chip.appendChild(el("span", null, metricLabel(task)));
      chip.querySelector("span:last-child").style.color = "var(--text-primary)";

      chip.addEventListener("click", function () {
        var index = state.activeMetrics.indexOf(task);
        if (index === -1) state.activeMetrics.push(task);
        else if (state.activeMetrics.length > 1) state.activeMetrics.splice(index, 1);
        renderMetricFilters(report);
        renderMetricChart();
      });

      host.appendChild(chip);
    });
  }

  function renderLegend(hostId, items) {
    var host = document.getElementById(hostId);
    clear(host);
    items.forEach(function (item) {
      var node = el("span", { class: "item" });
      var swatch = el("span", { class: "swatch" });
      swatch.style.background = item.color;
      node.appendChild(swatch);
      node.appendChild(el("span", null, item.label));
      host.appendChild(node);
    });
  }

  function renderMetricChart() {
    var container = document.getElementById("plot-metrics");
    var series = state.activeMetrics.map(function (task) {
      return {
        label: metricLabel(task),
        color: metricColor(task),
        digits: 4,
        points: state.rows
          .filter(function (row) {
            return row.metrics && row.metrics[task] != null;
          })
          .map(function (row) {
            return { x: row.step, y: row.metrics[task] };
          })
      };
    });

    drawLineChart(container, {
      series: series,
      ariaLabel: "Evaluation metrics against training step",
      height: 320
    });
    renderLegend(
      "legend-metrics",
      series.map(function (s) {
        return { label: s.label, color: s.color };
      })
    );
  }

  function renderLossChart() {
    var container = document.getElementById("plot-loss");
    var trained = state.rows.filter(function (row) {
      return !row.is_base;
    });

    var series = [
      {
        label: "Train loss",
        color: "var(--series-1)",
        digits: 4,
        points: trained
          .filter(function (r) { return r.train_loss != null; })
          .map(function (r) { return { x: r.step, y: r.train_loss }; })
      },
      {
        label: "Validation loss",
        color: "var(--series-2)",
        digits: 4,
        points: trained
          .filter(function (r) { return r.eval_loss != null; })
          .map(function (r) { return { x: r.step, y: r.eval_loss }; })
      }
    ].filter(function (s) {
      return s.points.length > 0;
    });

    if (!series.length) {
      clear(container);
      container.appendChild(
        el("p", { class: "note" }, "No loss recorded — checkpoints.jsonl was not written for this run.")
      );
      renderLegend("legend-loss", []);
      return;
    }

    drawLineChart(container, {
      series: series,
      ariaLabel: "Training and validation loss against step",
      height: 260,
      rightGutter: 92
    });
    renderLegend(
      "legend-loss",
      series.map(function (s) {
        return { label: s.label, color: s.color };
      })
    );
  }

  function renderBarChart() {
    var report = state.report;
    var summary = report.summary || {};
    var primary = summary.primary_metric || state.activeMetrics[0];
    if (!primary) return;

    document.getElementById("bars-title").textContent = metricLabel(primary) + " by checkpoint";

    var bestName = summary.best && summary.best.name;
    var rows = state.rows
      .filter(function (row) {
        return row.metrics && row.metrics[primary] != null;
      })
      .map(function (row, i, all) {
        return {
          label: row.name,
          step: row.step,
          value: row.metrics[primary],
          highlight: row.name === bestName,
          // Every third tick, plus the first and last, or the labels collide.
          showTick: i % 3 === 0 || i === all.length - 1,
          tickLabel: row.is_base ? "base" : String(row.step)
        };
      });

    drawBarChart(document.getElementById("plot-bars"), {
      rows: rows,
      color: metricColor(primary),
      metricLabel: metricLabel(primary),
      ariaLabel: metricLabel(primary) + " for each checkpoint",
      height: 260
    });
  }

  function renderTable() {
    var report = state.report;
    var tasks = report.tasks || [];
    var bestName = (report.summary && report.summary.best && report.summary.best.name) || null;

    var thead = document.querySelector("#results-table thead");
    var tbody = document.querySelector("#results-table tbody");
    clear(thead);
    clear(tbody);

    var headRow = el("tr");
    ["Checkpoint", "Step", "Train loss", "Val loss"].forEach(function (title) {
      headRow.appendChild(el("th", { scope: "col" }, title));
    });
    tasks.forEach(function (task) {
      headRow.appendChild(el("th", { scope: "col" }, metricLabel(task)));
    });
    thead.appendChild(headRow);

    state.rows.forEach(function (row) {
      var tr = el("tr", { "data-best": String(row.name === bestName) });
      var nameCell = el("td");
      nameCell.appendChild(document.createTextNode(row.name));
      if (row.is_base) nameCell.appendChild(el("span", { class: "tag" }, "base"));
      else if (row.name === bestName) nameCell.appendChild(el("span", { class: "tag" }, "best"));
      tr.appendChild(nameCell);

      tr.appendChild(el("td", null, String(row.step)));
      tr.appendChild(el("td", null, fmt(row.train_loss, 4)));
      tr.appendChild(el("td", null, fmt(row.eval_loss, 4)));
      tasks.forEach(function (task) {
        tr.appendChild(el("td", null, fmt(row.metrics && row.metrics[task], 4)));
      });
      tbody.appendChild(tr);
    });
  }

  function renderSamples() {
    var card = document.getElementById("samples-card");
    var host = document.getElementById("samples");
    clear(host);

    var withSamples = state.rows.filter(function (row) {
      return (row.samples || []).length > 0;
    });
    if (withSamples.length < 1) {
      card.hidden = true;
      return;
    }
    card.hidden = false;

    withSamples.slice(0, 2).forEach(function (row) {
      var sample = row.samples[0];
      var node = el("article", { class: "sample" });
      node.appendChild(el("h3", null, row.is_base ? "Base model" : row.name));
      node.appendChild(el("p", null, "Reference: " + sample.reference));
      node.appendChild(el("p", { class: "prediction" }, sample.prediction));
      host.appendChild(node);
    });
  }

  function renderFooter(report) {
    var run = report.run || {};
    var decoding = run.decoding || {};
    document.getElementById("footer").textContent =
      "Report v" + (report.report_version || "?") +
      " · " + (run.checkpoint_dir || "") +
      " · decoding: " + (decoding.temperature ? "T=" + decoding.temperature : "greedy") +
      ", max_new_tokens=" + (decoding.max_new_tokens || "?") +
      " · regenerate with `llmft eval --config configs/eval.yaml`";
  }

  function renderAll() {
    renderMetricChart();
    renderLossChart();
    renderBarChart();
  }

  /* ------------------------------------------------------------------ theme */

  function initTheme() {
    var button = document.getElementById("theme-toggle");
    var stored = null;
    try {
      stored = localStorage.getItem("llmft-theme");
    } catch (err) {
      /* private browsing - fall back to the OS setting */
    }
    if (stored) document.documentElement.setAttribute("data-theme", stored);

    function label() {
      var current = document.documentElement.getAttribute("data-theme");
      var dark =
        current === "dark" ||
        (!current && window.matchMedia("(prefers-color-scheme: dark)").matches);
      button.textContent = dark ? "Light" : "Dark";
    }

    button.addEventListener("click", function () {
      var current = document.documentElement.getAttribute("data-theme");
      var dark =
        current === "dark" ||
        (!current && window.matchMedia("(prefers-color-scheme: dark)").matches);
      var next = dark ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try {
        localStorage.setItem("llmft-theme", next);
      } catch (err) {
        /* ignore */
      }
      label();
      // SVG strokes read the custom properties at draw time, so redraw.
      if (state.report) renderAll();
    });

    label();
  }

  /* ------------------------------------------------------------------- boot */

  function showError(message, detail) {
    var host = document.getElementById("state");
    clear(host);
    host.appendChild(el("p", null, message));
    if (detail) {
      var p = el("p");
      p.innerHTML = detail;
      host.appendChild(p);
    }
  }

  function start(report) {
    state.report = report;
    state.rows = (report.checkpoints || []).slice().sort(function (a, b) {
      return a.step - b.step;
    });

    state.activeMetrics = (report.tasks || []).filter(function (task) {
      return NON_DIRECTIONAL.indexOf(task) === -1;
    });
    if (!state.activeMetrics.length) state.activeMetrics = (report.tasks || []).slice(0, 1);

    // Reveal before drawing: a hidden container reports clientWidth 0, the
    // charts would fall back to a guessed width, and CSS would then scale the
    // SVG - which silently desyncs every hover coordinate.
    document.getElementById("state").hidden = true;
    document.getElementById("content").hidden = false;

    renderMasthead(report);
    renderTiles(report);
    renderMetricFilters(report);
    renderAll();
    renderTable();
    renderSamples();
    renderFooter(report);

    var resizeTimer = null;
    window.addEventListener("resize", function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(renderAll, 120);
    });
  }

  initTheme();

  fetch(DATA_URL, { cache: "no-store" })
    .then(function (response) {
      if (!response.ok) throw new Error(response.status + " " + response.statusText);
      return response.json();
    })
    .then(function (report) {
      if (!report || !report.checkpoints || !report.checkpoints.length) {
        showError(
          "The report has no checkpoints in it.",
          "Run <code>llmft eval --config configs/eval.yaml</code> to generate one."
        );
        return;
      }
      start(report);
    })
    .catch(function (error) {
      showError(
        "Could not load " + DATA_URL + " (" + error.message + ").",
        "Serve the folder rather than opening the file directly — <code>make dashboard</code> — " +
          "then run <code>llmft eval --config configs/eval.yaml</code> to refresh the data."
      );
    });
})();
