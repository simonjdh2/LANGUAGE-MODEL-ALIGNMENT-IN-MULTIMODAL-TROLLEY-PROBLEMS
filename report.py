"""
report.py — HTML report generator.

Loads text and image arm CSVs and produces a self-contained HTML file.
Face images are embedded as base64 data URIs so the report is portable.

Each scenario card shows:
  - Which group is hit by inaction vs action (framing)
  - Per-provider choice badges for text and image arms
  - Perception validity summary for the image arm
  - Position-bias indicators (base vs mirror comparison)
  - Collapsible raw responses and reasoning

Summary cards at the top show aggregate statistics per provider:
  - Answer / refusal rates
  - Action vs inaction rates
  - Image arm perception validity and choice rates
  - Text–image agreement rates
  - Position bias rates

The paired analysis section adjusts for inaction and position bias to
surface genuine demographic preferences (scenarios where the model saves
the same demographic group in both the base and mirrored versions).
"""

import base64
import csv
from pathlib import Path

from scenario_generator import _TEXT_CSV_DIR, _IMAGE_CSV_DIR, _HTML_DIR, RUN_PREFIX


def _img_tag(path, height=100):
    try:
        with open(path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("utf-8")
        return (f'<img src="data:image/jpeg;base64,{b64}" height="{height}" '
                f'style="border-radius:4px;margin:2px;">')
    except Exception:
        return f'<span style="color:red">[missing: {Path(path).name}]</span>'


def _load_text_csv(path):
    data = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            is_mirror = row.get("is_mirror", "False") == "True"
            key = (int(row["index"]), is_mirror, row["provider"])
            data[key] = {
                "left_label":     row["left_label"],
                "right_label":    row["right_label"],
                "left_count":     row.get("left_count", ""),
                "right_count":    row.get("right_count", ""),
                "choice":         row["choice"],
                "choice_raw":     row["choice_raw"],
                "reasoning":      row.get("reasoning", ""),
                "refused":        row.get("refused", ""),
                "latency":        row.get("latency_s", ""),
                "model":          row["model"],
                "category":       row["category"],
                "system_role":    row["system_role"],
                "scenario_start": row.get("scenario_start", ""),
                "is_mirror":      is_mirror,
                "inaction_harms": row.get("inaction_harms", ""),
                "option_order":   row.get("option_order", ""),
                "scenario_end":   row.get("scenario_end", ""),
            }
    return data


def _load_image_csv(path):
    data, scenarios = {}, {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx       = int(row["index"])
            is_mirror = row.get("is_mirror", "False") == "True"
            key       = (idx, is_mirror, row["provider"])
            scen_key  = (idx, is_mirror)
            if scen_key not in scenarios:
                scenarios[scen_key] = {
                    "left_image_paths":  [p for p in row["left_image_paths"].split("|") if p],
                    "right_image_paths": [p for p in row["right_image_paths"].split("|") if p],
                    "left_label":        row["left_label"],
                    "right_label":       row["right_label"],
                }
            data[key] = {
                "perception_refused":     row.get("perception_refused", ""),
                "perception_valid":       row.get("perception_valid", ""),
                "perception_raw":         row.get("perception_raw", ""),
                "perception_latency":     row.get("perception_latency_s", ""),
                "left_perceived_count":   row.get("left_perceived_count", ""),
                "left_perceived_age":     row.get("left_perceived_age", ""),
                "left_perceived_race":    row.get("left_perceived_race", ""),
                "left_perceived_gender":  row.get("left_perceived_gender", ""),
                "right_perceived_count":  row.get("right_perceived_count", ""),
                "right_perceived_age":    row.get("right_perceived_age", ""),
                "right_perceived_race":   row.get("right_perceived_race", ""),
                "right_perceived_gender": row.get("right_perceived_gender", ""),
                "choice":            row.get("choice", ""),
                "choice_raw":        row.get("choice_raw", ""),
                "choice_reasoning":  row.get("choice_reasoning", ""),
                "choice_refused":    row.get("choice_refused", ""),
                "choice_latency":    row.get("choice_latency_s", ""),
                "model":          row["model"],
                "is_mirror":      is_mirror,
            }
    return data, scenarios


def _choice_badge(choice, refused=""):
    if refused == "rate_limit":
        return '<span class="badge rate-limit">rate limit</span>'
    if refused == "api_error":
        return '<span class="badge refused">api error</span>'
    if refused not in ("", "False"):
        return f'<span class="badge refused">{refused}</span>'
    if choice.lower() == "left":
        return '<span class="badge group-a-badge">Group A</span>'
    if choice.lower() == "right":
        return '<span class="badge group-b-badge">Group B</span>'
    if choice.lower() in ("refused", "no_choice"):
        return '<span class="badge model-refused">no choice</span>'
    if choice == "":
        return '<span class="badge grey-badge">—</span>'
    return f'<span class="badge grey-badge">{choice[:30]}</span>'


def _saved_badge(choice, refused):
    if refused == "rate_limit":
        return '<span class="badge rate-limit">rate limit</span>'
    if refused == "api_error":
        return '<span class="badge refused">api error</span>'
    if refused == "refused":
        return '<span class="badge model-refused">refused to choose</span>'
    if refused not in ("", "False"):
        return f'<span class="badge refused">{refused}</span>'
    if choice.lower() == "left":
        return ('<span class="badge group-b-badge">Saves Group B</span>'
                '<span class="hit-label"> hits Group A</span>')
    if choice.lower() == "right":
        return ('<span class="badge group-a-badge">Saves Group A</span>'
                '<span class="hit-label"> hits Group B</span>')
    if choice == "":
        return '<span class="badge grey-badge">—</span>'
    return f'<span class="badge grey-badge">{choice[:30]}</span>'


def _agreement_badge(text_choice, text_refused, img_choice, img_refused, img_percep_valid):
    if img_percep_valid != "True":
        return '<span class="badge grey-badge">n/a</span>'
    t, i = text_choice.lower(), img_choice.lower()
    t_answered = t in ("left", "right")
    i_answered = i in ("left", "right")
    t_refused  = text_refused not in ("", "False") or t in ("refused", "no_choice")
    i_refused  = img_refused  not in ("", "False") or i in ("refused", "no_choice")
    if t_answered and i_answered:
        if t == i:
            return '<span class="badge agree">&#10003; agree</span>'
        return '<span class="badge disagree">&#10007; disagree</span>'
    if (t_answered and i_refused) or (t_refused and i_answered):
        return '<span class="badge disagree">&#10007; disagree (refusal)</span>'
    if t_refused and i_refused:
        return '<span class="badge agree">&#10003; agree (both refused)</span>'
    return '<span class="badge grey-badge">unparsed</span>'


def _perception_summary(img_row):
    lc, la, lr, lg = (img_row[k] for k in
        ["left_perceived_count", "left_perceived_age",
         "left_perceived_race",  "left_perceived_gender"])
    rc, ra, rr, rg = (img_row[k] for k in
        ["right_perceived_count", "right_perceived_age",
         "right_perceived_race",  "right_perceived_gender"])
    if not any([lc, la, lr, lg, rc, ra, rr, rg]):
        return "—"
    return (f"<b>Group A:</b> {lc} · {la} · {lr} · {lg} &nbsp;&nbsp; "
            f"<b>Group B:</b> {rc} · {ra} · {rr} · {rg}")


def _percep_badge(img_row):
    pr = img_row["perception_refused"]
    if pr == "rate_limit":
        return '<span class="badge rate-limit">rate limit</span>'
    if pr not in ("", "False"):
        return '<span class="badge refused">api error</span>'
    if img_row["perception_valid"] == "True":
        return '<span class="badge valid">&#10003; valid</span>'
    if img_row["perception_valid"] == "False":
        return '<span class="badge invalid">&#10007; invalid</span>'
    return ""


def _compute_summary(text_data, image_data, providers):
    stats = {p: {
        "total": 0, "text_answered": 0, "text_refused": 0, "text_model_refused": 0,
        "text_inaction": 0, "text_action": 0,
        "img_percep_valid": 0, "img_percep_refused": 0, "img_percep_invalid": 0,
        "img_choice_answered": 0, "img_choice_refused": 0,
        "img_inaction": 0, "img_action": 0,
        "agree": 0, "disagree": 0, "comparable": 0,
        "pos_consistent": 0, "pos_biased": 0, "pos_pairs": 0,
        "img_pos_consistent": 0, "img_pos_biased": 0, "img_pos_pairs": 0,
    } for p in providers}

    all_indices = sorted(set(idx for (idx, _mirror, _p) in text_data))

    for idx in all_indices:
        for p in providers:
            orig_key = (idx, False, p)
            mirr_key = (idx, True,  p)

            for is_mirror, key in [(False, orig_key), (True, mirr_key)]:
                if key not in text_data:
                    continue
                s = stats[p]
                s["total"] += 1
                t = text_data[key]
                i = image_data.get(key, {})

                if t["refused"] not in ("", "False"):
                    s["text_refused"] += 1
                elif t["choice"].lower() in ("refused", "no_choice"):
                    s["text_model_refused"] += 1
                elif t["choice"].lower() in ("left", "right"):
                    s["text_answered"] += 1

                tc_lower = t["choice"].lower()
                ih = t.get("inaction_harms", "")
                if tc_lower in ("left", "right"):
                    chose_inaction = (ih == "left"  and tc_lower == "left") or \
                                     (ih == "right" and tc_lower == "right")
                    if chose_inaction:
                        s["text_inaction"] += 1
                    else:
                        s["text_action"] += 1

                if i.get("perception_refused") not in ("", "False"):
                    s["img_percep_refused"] += 1
                elif i.get("perception_valid") == "True":
                    s["img_percep_valid"] += 1
                elif i.get("perception_valid") == "False":
                    s["img_percep_invalid"] += 1

                if i.get("choice_refused") not in ("", "False"):
                    s["img_choice_refused"] += 1
                elif i.get("choice", "").lower() in ("left", "right"):
                    s["img_choice_answered"] += 1

                ic = i.get("choice", "").lower()
                if ic in ("left", "right") and i.get("perception_valid") == "True":
                    img_inaction = (ih == "left"  and ic == "right") or \
                                   (ih == "right" and ic == "left")
                    if img_inaction:
                        s["img_inaction"] += 1
                    else:
                        s["img_action"] += 1

                tc = t["choice"].lower()
                t_answered = tc in ("left", "right")
                i_answered = ic in ("left", "right")
                t_refused  = t["refused"] not in ("", "False") or t["choice"].lower() in ("refused", "no_choice")
                i_refused  = i.get("choice_refused", "") not in ("", "False") or i.get("choice", "").lower() in ("refused", "no_choice")

                if i.get("perception_valid") == "True":
                    if t_answered and i_answered:
                        s["comparable"] += 1
                        if tc == ic:
                            s["agree"] += 1
                        else:
                            s["disagree"] += 1
                    elif (t_answered and i_refused) or (t_refused and i_answered):
                        s["comparable"] += 1
                        s["disagree"] += 1
                    elif t_refused and i_refused:
                        s["comparable"] += 1
                        s["agree"] += 1

            s = stats[p]
            if orig_key in text_data and mirr_key in text_data:
                oc = text_data[orig_key]["choice"].lower()
                mc = text_data[mirr_key]["choice"].lower()
                if oc in ("left", "right") and mc in ("left", "right"):
                    s["pos_pairs"] += 1
                    if oc != mc:
                        s["pos_consistent"] += 1
                    else:
                        s["pos_biased"] += 1

            orig_img = image_data.get(orig_key, {})
            mirr_img = image_data.get(mirr_key, {})
            if (orig_img.get("perception_valid") == "True" and
                    mirr_img.get("perception_valid") == "True"):
                oc_img = orig_img.get("choice", "").lower()
                mc_img = mirr_img.get("choice", "").lower()
                if oc_img in ("left", "right") and mc_img in ("left", "right"):
                    s["img_pos_pairs"] += 1
                    if oc_img != mc_img:
                        s["img_pos_consistent"] += 1
                    else:
                        s["img_pos_biased"] += 1

    return stats


def _pct(n, total):
    return "—" if total == 0 else f"{n}/{total} ({round(n / total * 100)}%)"


def _demographic_from_label(label, category):
    label_l = label.lower()
    if category == 'Race':
        for race in ['White', 'Black', 'East Asian', 'South Asian']:
            if race.lower() in label_l:
                return race
    elif category == 'Gender':
        if 'woman' in label_l or 'girl' in label_l or 'women' in label_l:
            return 'Female'
        if 'man' in label_l or 'boy' in label_l or 'men' in label_l:
            return 'Male'
    elif category == 'Age':
        if 'elderly' in label_l:
            return 'elderly'
        if 'boy' in label_l or 'girl' in label_l:
            return 'child'
        if 'man' in label_l or 'woman' in label_l or 'men' in label_l or 'women' in label_l:
            return 'adult'
    return None


def _compute_paired_analysis(text_data, providers):
    """
    For each scenario pair (base + mirror), identify the model's genuine
    demographic preference — the group saved consistently across both versions.

    A model with inaction bias but no demographic bias will save different
    demographic groups in base vs mirror (because inaction always saves the
    same side, but groups swap sides between versions). Only when a model saves
    the same demographic group in both versions is a genuine preference counted.
    """
    all_indices = sorted(set(idx for (idx, _mirror, _p) in text_data))
    result = {}
    for p in providers:
        result[p] = {}
        for idx in all_indices:
            orig_key = (idx, False, p)
            mirr_key = (idx, True,  p)
            if orig_key not in text_data or mirr_key not in text_data:
                continue
            orig = text_data[orig_key]
            mirr = text_data[mirr_key]
            cat  = orig.get('category', '')
            if cat not in result[p]:
                result[p][cat] = {'pairs_total': 0, 'with_signal': 0,
                                   'inaction_only': 0, 'prefs': {}}
            oc = orig['choice'].lower()
            mc = mirr['choice'].lower()
            if oc not in ('left', 'right') or mc not in ('left', 'right'):
                continue
            result[p][cat]['pairs_total'] += 1
            orig_saved = orig['left_label'] if oc == 'left' else orig['right_label']
            mirr_saved = mirr['left_label'] if mc == 'left' else mirr['right_label']
            od = _demographic_from_label(orig_saved, cat)
            md = _demographic_from_label(mirr_saved, cat)
            if od and md and od == md:
                result[p][cat]['with_signal'] += 1
                result[p][cat]['prefs'][od] = result[p][cat]['prefs'].get(od, 0) + 1
            else:
                result[p][cat]['inaction_only'] += 1
    return result


def _compute_category_stats(text_data, image_data, providers):
    cats = {}
    all_indices = sorted(set(idx for (idx, _mirror, _p) in text_data))
    for idx in all_indices:
        cat = None
        for p in providers:
            if (idx, False, p) in text_data:
                cat = text_data[(idx, False, p)].get("category", "Unknown")
                break
        if not cat:
            continue
        if cat not in cats:
            cats[cat] = {"scenarios": 0, "pos_pairs": 0, "pos_biased": 0,
                         "agree": 0, "comparable": 0,
                         "text_inaction": 0, "text_answered": 0}
        s = cats[cat]
        s["scenarios"] += 1
        for p in providers:
            orig_key = (idx, False, p)
            mirr_key = (idx, True,  p)
            if orig_key in text_data and mirr_key in text_data:
                oc = text_data[orig_key]["choice"].lower()
                mc = text_data[mirr_key]["choice"].lower()
                if oc in ("left", "right") and mc in ("left", "right"):
                    s["pos_pairs"] += 1
                    if oc == mc:
                        s["pos_biased"] += 1
            for is_mirror in [False, True]:
                key = (idx, is_mirror, p)
                t = text_data.get(key, {})
                i = image_data.get(key, {})
                tc = t.get("choice", "").lower()
                ih = t.get("inaction_harms", "")
                if tc in ("left", "right"):
                    s["text_answered"] += 1
                    chose_inaction = (ih == "left" and tc == "left") or (ih == "right" and tc == "right")
                    if chose_inaction:
                        s["text_inaction"] += 1
                if i.get("perception_valid") == "True":
                    ic = i.get("choice", "").lower()
                    if tc in ("left", "right") and ic in ("left", "right"):
                        s["comparable"] += 1
                        if tc == ic:
                            s["agree"] += 1
    return cats


def find_latest_pair(text_csv_dir=None, image_csv_dir=None):
    """Return the most recent run CSV from each arm independently."""
    def _latest(d):
        n, found = 1, None
        while (d / f"run_{n}_results.csv").exists():
            found = d / f"run_{n}_results.csv"
            n += 1
        return found

    t_dir = Path(text_csv_dir  or _TEXT_CSV_DIR)
    i_dir = Path(image_csv_dir or _IMAGE_CSV_DIR)
    t = _latest(t_dir)
    i = _latest(i_dir)
    if t is None:
        raise FileNotFoundError(f"No run_N_results.csv found in {t_dir}")
    if i is None:
        raise FileNotFoundError(f"No run_N_results.csv found in {i_dir}")
    return t, i


def generate_report(text_csv_path, image_csv_path, output_dir=None):
    """
    Generate a self-contained HTML report from a matched text + image CSV pair.

    Writes to results/HTMLs/V9/ by default; override with output_dir.
    Returns the output path.
    """
    text_csv_path  = Path(text_csv_path)
    image_csv_path = Path(image_csv_path)

    text_data, (image_data, img_scenarios) = (
        _load_text_csv(text_csv_path),
        _load_image_csv(image_csv_path),
    )
    all_indices    = sorted(set(idx for (idx, _mirror, _p) in text_data))
    providers      = ["claude", "openai", "gemini"]
    summary        = _compute_summary(text_data, image_data, providers)
    category_stats = _compute_category_stats(text_data, image_data, providers)
    paired_analysis = _compute_paired_analysis(text_data, providers)

    summary_cards = ""
    for p in providers:
        s = summary[p]
        summary_cards += f"""
        <div class="summary-card">
          <h3>{p}</h3>
          <table class="summary-table">
            <tr><td>Total scenarios</td><td>{s['total']}</td></tr>
            <tr><td>Text: answered</td><td>{_pct(s['text_answered'], s['total'])}</td></tr>
            <tr><td>Text: model refused</td><td>{_pct(s['text_model_refused'], s['total'])}</td></tr>
            <tr><td>Text: api error</td><td>{_pct(s['text_refused'], s['total'])}</td></tr>
            <tr><td colspan="2" style="padding-top:8px;font-weight:bold;font-size:.85em">Action / Inaction (text arm)</td></tr>
            <tr><td>Continue straight <span style="color:#9ca3af;font-size:.85em">(inaction — car stays on path)</span></td><td>{_pct(s['text_inaction'], s['text_answered'])}</td></tr>
            <tr><td>Swerve <span style="color:#9ca3af;font-size:.85em">(action — car diverts)</span></td><td>{_pct(s['text_action'], s['text_answered'])}</td></tr>
            <tr><td colspan="2" style="padding-top:8px;font-weight:bold;font-size:.85em">Image arm</td></tr>
            <tr><td>Perception valid</td><td>{_pct(s['img_percep_valid'], s['total'])}</td></tr>
            <tr><td>Perception invalid</td><td>{_pct(s['img_percep_invalid'], s['total'])}</td></tr>
            <tr><td>Perception refused</td><td>{_pct(s['img_percep_refused'], s['total'])}</td></tr>
            <tr><td>Choice answered</td><td>{_pct(s['img_choice_answered'], s['total'])}</td></tr>
            <tr><td>Choice refused</td><td>{_pct(s['img_choice_refused'], s['total'])}</td></tr>
            <tr><td colspan="2" style="padding-top:8px;font-weight:bold;font-size:.85em">Action / Inaction (image arm)</td></tr>
            <tr><td>Continue straight <span style="color:#9ca3af;font-size:.85em">(inaction)</span></td><td>{_pct(s['img_inaction'], s['img_choice_answered'])}</td></tr>
            <tr><td>Swerve <span style="color:#9ca3af;font-size:.85em">(action)</span></td><td>{_pct(s['img_action'], s['img_choice_answered'])}</td></tr>
            <tr><td colspan="2" style="padding-top:8px;font-weight:bold;font-size:.85em">Agreement</td></tr>
            <tr><td>Agree</td><td>{_pct(s['agree'], s['comparable'])}</td></tr>
            <tr><td>Disagree</td><td>{_pct(s['disagree'], s['comparable'])}</td></tr>
            <tr><td colspan="2" style="padding-top:8px;font-weight:bold;font-size:.85em">Position bias (text)</td></tr>
            <tr><td>Consistent (same group)</td><td>{_pct(s['pos_consistent'], s['pos_pairs'])}</td></tr>
            <tr><td>Biased (same side)</td><td>{_pct(s['pos_biased'], s['pos_pairs'])}</td></tr>
            <tr><td colspan="2" style="padding-top:8px;font-weight:bold;font-size:.85em">Position bias (image)</td></tr>
            <tr><td>Consistent (same group)</td><td>{_pct(s['img_pos_consistent'], s['img_pos_pairs'])}</td></tr>
            <tr><td>Biased (same side)</td><td>{_pct(s['img_pos_biased'], s['img_pos_pairs'])}</td></tr>
          </table>
        </div>"""

    cat_rows = ""
    for cat, s in sorted(category_stats.items()):
        cat_rows += f"""
        <tr>
          <td>{cat}</td>
          <td>{s['scenarios']}</td>
          <td>{_pct(s['text_inaction'], s['text_answered'])}</td>
          <td>{_pct(s['pos_biased'], s['pos_pairs'])}</td>
          <td>{_pct(s['agree'], s['comparable'])}</td>
        </tr>"""
    category_table = f"""
    <table class="category-table">
      <tr>
        <th>Category</th><th>Scenarios</th>
        <th>Inaction rate (text) <span style="font-weight:400;font-size:.85em">chose continue straight</span></th>
        <th>Position bias (text) <span style="font-weight:400;font-size:.85em">same side in base+mirror</span></th>
        <th>Text–image agreement</th>
      </tr>{cat_rows}
    </table>"""

    _CAT_DEMOGRAPHICS = {
        'Race':          ['White', 'Black', 'East Asian', 'South Asian'],
        'Gender':        ['Female', 'Male'],
        'Age':           ['child', 'adult', 'elderly'],
        'Utilitarianism': [],
    }
    paired_html = ""
    for cat, demos in _CAT_DEMOGRAPHICS.items():
        if not demos:
            continue
        demo_headers = "".join(f"<th>Prefers {d}</th>" for d in demos)
        paired_html += f"""
    <h3 style="font-size:.95em;color:#374151;margin:20px 0 8px">{cat}</h3>
    <table class="category-table">
      <tr>
        <th>Provider</th><th>Pairs</th><th>Signal rate</th>
        {demo_headers}
      </tr>"""
        for p in providers:
            pdata  = paired_analysis.get(p, {}).get(cat, {})
            total  = pdata.get('pairs_total', 0)
            signal = pdata.get('with_signal', 0)
            prefs  = pdata.get('prefs', {})
            sig_pct = f"{signal}/{total} ({round(signal/total*100)}%)" if total else "—"
            demo_cells = ""
            for d in demos:
                n = prefs.get(d, 0)
                pct = round(n / signal * 100) if signal else 0
                bar_width = pct
                demo_cells += (
                    f'<td><span style="font-weight:600">{n}</span>'
                    f'<span style="color:#9ca3af;font-size:.8em"> /{signal} ({pct}%)</span>'
                    f'<div style="height:4px;width:{bar_width}%;background:#6366f1;'
                    f'border-radius:2px;margin-top:3px;"></div></td>'
                )
            paired_html += f"""
      <tr>
        <td><strong>{p}</strong></td>
        <td style="color:#6b7280">{total}</td>
        <td>{sig_pct}</td>
        {demo_cells}
      </tr>"""
        paired_html += "\n    </table>"

    all_categories = sorted(category_stats.keys())
    all_roles      = sorted(set(
        text_data[k].get("system_role", "")
        for k in text_data if not k[1]
    ) - {""})
    cat_buttons = "".join(
        f'<button class="filter-btn" data-filter="cat:{c}" onclick="applyFilter(\'cat:{c}\')">{c}</button>'
        for c in all_categories
    )
    role_buttons = "".join(
        f'<button class="filter-btn" data-filter="role:{r}" onclick="applyFilter(\'role:{r}\')">{r}</button>'
        for r in all_roles
    )
    filter_bar = f"""
    <div class="filter-bar">
      <strong>Category:</strong>
      <button class="filter-btn active" data-filter="all" onclick="applyFilter('all')">All</button>
      {cat_buttons}
      <span style="color:#d1d5db;margin:0 4px">|</span>
      <strong>Role:</strong>
      {role_buttons}
      <span style="margin-left:auto;display:flex;gap:6px;">
        <button class="ctrl-btn" onclick="collapseAll(true)">Collapse all</button>
        <button class="ctrl-btn" onclick="collapseAll(false)">Expand all</button>
      </span>
    </div>"""

    scenario_cards = ""
    all_indices = sorted(set(idx for (idx, _mirror, _p) in text_data))

    for idx in all_indices:
        sample_key = (idx, False, "claude") if (idx, False, "claude") in text_data \
                     else (idx, False, providers[0])
        meta = text_data.get(sample_key, {})
        inaction_harms = meta.get("inaction_harms", "")
        option_order   = meta.get("option_order", "")

        if inaction_harms == "left":
            framing_str = ("Continue straight (inaction) → <strong>Group A hit</strong>, Group B saved &nbsp;|&nbsp; "
                           "Swerve (action) → <strong>Group B hit</strong>, Group A saved")
        elif inaction_harms == "right":
            framing_str = ("Continue straight (inaction) → <strong>Group B hit</strong>, Group A saved &nbsp;|&nbsp; "
                           "Swerve (action) → <strong>Group A hit</strong>, Group B saved")
        else:
            framing_str = ""

        order_label = "inaction listed first" if option_order == "inaction_first" else "action listed first"

        scenario_cards += f"""
<div class="scenario" data-category="{meta.get('category', '')}" data-role="{meta.get('system_role', '')}">
  <div class="meta" onclick="toggleCard(this)" style="cursor:pointer">
    <span class="arrow">▼</span>
    <span><strong>Scenario {idx}</strong></span>
    <span class="cat-badge">{meta.get('category', '')}</span>
    <span class="role-badge">{meta.get('system_role', '')}</span>
  </div>
  <div class="scenario-content">
  <div class="framing-bar">
    <span class="framing-label">Framing:</span> {framing_str}
    <span class="order-badge">{order_label}</span>
  </div>"""

        for is_mirror in [False, True]:
            mirror_label = "↔ mirror (groups swapped)" if is_mirror else "original"
            mirror_style = "background:#f0f4ff;" if is_mirror else ""

            sub_key    = (idx, is_mirror, "claude") if (idx, is_mirror, "claude") in text_data \
                         else (idx, is_mirror, providers[0])
            sub_meta   = text_data.get(sub_key, {})
            img_meta   = img_scenarios.get((idx, is_mirror), {})

            sub_ih    = sub_meta.get("inaction_harms", "")
            sub_order = sub_meta.get("option_order", "")
            sub_end   = sub_meta.get("scenario_end", "")
            if sub_ih == "left":
                sub_framing = ("Continue straight → <strong>Group A hit</strong> | "
                               "Swerve → <strong>Group B hit</strong>")
            elif sub_ih == "right":
                sub_framing = ("Continue straight → <strong>Group B hit</strong> | "
                               "Swerve → <strong>Group A hit</strong>")
            else:
                sub_framing = ""
            sub_order_label = "inaction listed first" if sub_order == "inaction_first" else "action listed first"

            if sub_ih == "left":
                group_a_header = "Group A — hit if car continues straight (inaction)"
                group_b_header = "Group B — hit if car swerves (action)"
            elif sub_ih == "right":
                group_a_header = "Group A — hit if car swerves (action)"
                group_b_header = "Group B — hit if car continues straight (inaction)"
            else:
                group_a_header = "Group A"
                group_b_header = "Group B"

            left_imgs  = "".join(_img_tag(p) for p in img_meta.get("left_image_paths", []))
            right_imgs = "".join(_img_tag(p) for p in img_meta.get("right_image_paths", []))

            provider_rows = ""
            for p in providers:
                key   = (idx, is_mirror, p)
                t_row = text_data.get(key, {})
                i_row = image_data.get(key, {})

                counterpart_key = (idx, not is_mirror, p)
                cp_text  = text_data.get(counterpart_key, {})
                cp_image = image_data.get(counterpart_key, {})
                tc = t_row.get("choice", "").lower()
                cc = cp_text.get("choice", "").lower()
                if tc in ("left", "right") and cc in ("left", "right"):
                    if tc != cc:
                        pos_badge = '<span class="badge agree" title="Chose same group from both sides">&#10003; consistent</span>'
                    else:
                        pos_badge = '<span class="badge disagree" title="Chose same side in both — position bias">&#9888; pos. bias</span>'
                else:
                    pos_badge = ""

                ic       = i_row.get("choice", "").lower() if i_row else ""
                cp_ic    = cp_image.get("choice", "").lower()
                cp_valid = cp_image.get("perception_valid", "")
                this_valid = i_row.get("perception_valid", "") if i_row else ""
                if (ic in ("left", "right") and cp_ic in ("left", "right")
                        and this_valid == "True" and cp_valid == "True"):
                    if ic != cp_ic:
                        img_pos_badge = '<span class="badge agree" title="Image: same group from both sides">&#10003; img consistent</span>'
                    else:
                        img_pos_badge = '<span class="badge disagree" title="Image: same side in both — position bias">&#9888; img pos. bias</span>'
                elif ((this_valid == "False" and cp_valid == "True" and cp_ic in ("left", "right")) or
                      (cp_valid == "False" and this_valid == "True" and ic in ("left", "right"))):
                    img_pos_badge = '<span class="badge refused" title="Image: invalid perception on one side — cannot assess position bias">&#9888; img percep. mismatch</span>'
                else:
                    img_pos_badge = ""

                t_reasoning = t_row.get('reasoning', '')
                i_reasoning = i_row.get('choice_reasoning', '') if i_row else ''
                t_reasoning_html = (
                    f'<details><summary class="raw-toggle">reasoning</summary>'
                    f'<div class="reasoning-text">{t_reasoning}</div></details>'
                ) if t_reasoning else ''
                i_reasoning_html = (
                    f'<details><summary class="raw-toggle">reasoning</summary>'
                    f'<div class="reasoning-text">{i_reasoning}</div></details>'
                ) if i_reasoning else ''

                provider_rows += f"""
            <tr>
              <td><strong>{p}</strong><br>
                <span class="model-name">{t_row.get('model', '')}</span></td>
              <td>
                {_saved_badge(t_row.get('choice', ''), t_row.get('refused', ''))}
                {pos_badge}
                {t_reasoning_html}
                <details><summary class="raw-toggle">raw response</summary>
                  <pre class="raw-text">{t_row.get('choice_raw', '')}</pre></details>
                <span class="latency">{t_row.get('latency', '')}s</span>
              </td>
              <td>
                {_percep_badge(i_row) if i_row else ''}
                <div class="perception-summary">{_perception_summary(i_row) if i_row else '—'}</div>
              </td>
              <td>
                {_saved_badge(i_row.get('choice', ''), i_row.get('choice_refused', '')) if i_row else ''}
                {img_pos_badge}
                {i_reasoning_html}
                <details><summary class="raw-toggle">raw response</summary>
                  <pre class="raw-text">{i_row.get('choice_raw', '') if i_row else ''}</pre></details>
                <span class="latency">{i_row.get('choice_latency', '') if i_row else ''}s</span>
              </td>
              <td>{_agreement_badge(
                    t_row.get('choice', ''),
                    t_row.get('refused', ''),
                    i_row.get('choice', '') if i_row else '',
                    i_row.get('choice_refused', '') if i_row else '',
                    i_row.get('perception_valid', '') if i_row else '',
                  )}</td>
            </tr>"""

            scenario_cards += f"""
  <div style="margin-bottom:16px;{mirror_style}border-radius:6px;padding:10px;">
    <div style="font-size:.75em;font-weight:600;color:#6b7280;text-transform:uppercase;
                letter-spacing:.05em;margin-bottom:6px;">{mirror_label}</div>
    <div class="sub-framing">{sub_framing} &nbsp;<span class="order-badge">{sub_order_label}</span></div>
    <div class="groups">
      <div class="group">
        <div class="group-header">{group_a_header}</div>
        <div class="group-label">{sub_meta.get('left_label', '')}</div>
        <div class="images">{left_imgs or '<span class="no-img">no images</span>'}</div>
      </div>
      <div class="group">
        <div class="group-header">{group_b_header}</div>
        <div class="group-label">{sub_meta.get('right_label', '')}</div>
        <div class="images">{right_imgs or '<span class="no-img">no images</span>'}</div>
      </div>
    </div>
    {f'<details style="margin-bottom:8px;"><summary class="raw-toggle">show prompt framing</summary><pre class="raw-text">{sub_end}</pre></details>' if sub_end else ''}
    <table>
      <tr>
        <th>Provider</th>
        <th>Text: group saved <span style="font-weight:400">(hits other)</span></th>
        <th>Image: perception</th>
        <th>Image: group saved</th>
        <th>Agreement</th>
      </tr>
      {provider_rows}
    </table>
  </div>"""

        scenario_cards += "\n  </div>\n</div>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Unified Experiment Report</title>
<style>
  body              {{ font-family: system-ui, sans-serif; background:#f0f2f5; margin:0; padding:24px; }}
  h1                {{ color:#1a1a2e; margin-bottom:4px; }}
  .subtitle         {{ color:#666; font-size:.9em; margin-bottom:24px; }}
  h2                {{ color:#333; margin:32px 0 12px; font-size:1.1em; text-transform:uppercase; letter-spacing:.05em; }}
  .summary-row      {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:32px; }}
  .summary-card     {{ background:#fff; border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,.1); padding:16px; flex:1; min-width:200px; }}
  .summary-card h3  {{ margin:0 0 10px; font-size:1em; color:#444; text-transform:capitalize; }}
  .summary-table    {{ width:100%; font-size:.82em; border-collapse:collapse; }}
  .summary-table td {{ padding:3px 0; color:#555; }}
  .summary-table td:last-child {{ text-align:right; font-weight:500; color:#222; }}
  .scenario         {{ background:#fff; border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,.1); margin-bottom:20px; padding:20px; }}
  .meta             {{ display:flex; align-items:center; gap:10px; margin-bottom:14px; flex-wrap:wrap; }}
  .cat-badge        {{ background:#e0e7ff; color:#3730a3; border-radius:4px; padding:2px 8px; font-size:.8em; font-weight:600; }}
  .role-badge       {{ background:#f3f4f6; color:#374151; border-radius:4px; padding:2px 8px; font-size:.8em; }}
  .groups           {{ display:flex; gap:16px; margin-bottom:16px; }}
  .group            {{ flex:1; background:#f9fafb; border-radius:6px; padding:12px; }}
  .group-header     {{ font-size:.8em; font-weight:600; color:#6b7280; text-transform:uppercase; letter-spacing:.05em; margin-bottom:4px; }}
  .group-label      {{ font-size:.95em; color:#111; margin-bottom:8px; font-weight:500; }}
  .images           {{ display:flex; flex-wrap:wrap; gap:4px; }}
  .no-img           {{ font-size:.8em; color:#aaa; }}
  table             {{ width:100%; border-collapse:collapse; font-size:.88em; }}
  th                {{ background:#f3f4f6; padding:8px 10px; text-align:left; font-size:.8em; color:#4b5563; text-transform:uppercase; letter-spacing:.04em; }}
  td                {{ padding:10px; border-bottom:1px solid #f0f0f0; vertical-align:top; }}
  tr:last-child td  {{ border-bottom:none; }}
  .model-name       {{ font-size:.75em; color:#9ca3af; }}
  .latency          {{ font-size:.75em; color:#9ca3af; margin-left:6px; }}
  .badge            {{ display:inline-block; border-radius:4px; padding:2px 8px; font-size:.82em; font-weight:600; }}
  .group-a-badge    {{ background:#dbeafe; color:#1d4ed8; }}
  .group-b-badge    {{ background:#fce7f3; color:#9d174d; }}
  .agree            {{ background:#dcfce7; color:#166534; }}
  .disagree         {{ background:#fee2e2; color:#991b1b; }}
  .valid            {{ background:#dcfce7; color:#166534; }}
  .invalid          {{ background:#fee2e2; color:#991b1b; }}
  .refused          {{ background:#fff7ed; color:#c2410c; }}
  .rate-limit       {{ background:#fef2f2; color:#dc2626; }}
  .model-refused    {{ background:#f3e8ff; color:#7c3aed; }}
  .grey-badge       {{ background:#f3f4f6; color:#6b7280; }}
  .perception-summary {{ font-size:.78em; color:#555; margin-top:4px; line-height:1.7; }}
  .perception-summary b {{ color:#374151; }}
  .raw-toggle       {{ font-size:.72em; color:#9ca3af; cursor:pointer; margin-top:3px; }}
  .raw-text         {{ font-size:.75em; color:#6b7280; white-space:pre-wrap; margin:4px 0 0; background:#f9fafb; border-radius:4px; padding:6px; }}
  .reasoning-text   {{ font-size:.8em; color:#374151; font-style:italic; margin:4px 0 0; background:#fefce8; border-left:3px solid #fbbf24; border-radius:0 4px 4px 0; padding:6px 8px; line-height:1.5; }}
  .filter-bar       {{ display:flex; align-items:center; gap:8px; margin-bottom:16px; flex-wrap:wrap; background:#fff; border-radius:8px; padding:12px 16px; box-shadow:0 1px 4px rgba(0,0,0,.1); }}
  .filter-btn       {{ border:1px solid #d1d5db; background:#fff; border-radius:4px; padding:4px 12px; cursor:pointer; font-size:.85em; transition:all .15s; }}
  .filter-btn:hover {{ background:#f3f4f6; }}
  .filter-btn.active {{ background:#1d4ed8; color:#fff; border-color:#1d4ed8; }}
  .ctrl-btn         {{ border:1px solid #d1d5db; background:#f9fafb; border-radius:4px; padding:4px 10px; cursor:pointer; font-size:.82em; }}
  .ctrl-btn:hover   {{ background:#e5e7eb; }}
  .inaction-badge   {{ background:#f3f4f6; color:#6b7280; border-radius:4px; padding:2px 7px; font-size:.75em; margin-left:4px; }}
  .arrow            {{ font-size:.75em; color:#9ca3af; margin-right:4px; user-select:none; }}
  .framing-bar      {{ background:#f0f9ff; border-left:3px solid #0ea5e9; border-radius:0 6px 6px 0; padding:8px 12px; font-size:.83em; color:#0c4a6e; margin-bottom:12px; display:flex; align-items:center; gap:6px; flex-wrap:wrap; }}
  .framing-label    {{ font-weight:700; margin-right:4px; white-space:nowrap; }}
  .sub-framing      {{ font-size:.8em; color:#374151; background:#f8fafc; border-radius:4px; padding:5px 8px; margin-bottom:8px; }}
  .order-badge      {{ display:inline-block; background:#e0e7ff; color:#3730a3; border-radius:4px; padding:1px 7px; font-size:.75em; font-weight:600; }}
  .hit-label        {{ font-size:.78em; color:#6b7280; font-weight:400; }}
  .category-table   {{ width:100%; border-collapse:collapse; font-size:.88em; margin-bottom:32px; background:#fff; border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,.1); overflow:hidden; }}
  .category-table th {{ background:#f3f4f6; padding:8px 14px; text-align:left; font-size:.8em; color:#4b5563; text-transform:uppercase; letter-spacing:.04em; }}
  .category-table td {{ padding:9px 14px; border-bottom:1px solid #f0f0f0; color:#374151; }}
  .category-table tr:last-child td {{ border-bottom:none; }}
</style>
</head>
<body>
<h1>Unified Experiment Report</h1>
<p class="subtitle">
  Text arm: <strong>{text_csv_path.name}</strong> &nbsp;|&nbsp;
  Image arm: <strong>{image_csv_path.name}</strong>
</p>
<h2>Summary by provider</h2>
<div class="summary-row">{summary_cards}</div>
<h2>Category breakdown</h2>
{category_table}
<h2>Paired Analysis — Demographic Preferences (inaction-bias adjusted)</h2>
<p class="subtitle" style="margin-bottom:12px">
  Each scenario pair (original + mirror) controls for inaction and position bias — the groups
  swap sides between versions, so only a model that saves the <em>same demographic group</em>
  in both versions registers as a genuine preference. "Signal rate" = % of pairs showing a
  consistent preference. The preference breakdown shows which demographic was favoured among
  signal pairs only.
</p>
{paired_html}
<h2>Scenarios ({len(all_indices)} total)</h2>
{filter_bar}
{scenario_cards}
<script>
function applyFilter(f) {{
  document.querySelectorAll('.scenario').forEach(el => {{
    if (f === 'all') {{ el.style.display = ''; return; }}
    const [type, val] = f.split(':');
    const match = type === 'cat'
      ? el.dataset.category === val
      : el.dataset.role === val;
    el.style.display = match ? '' : 'none';
  }});
  document.querySelectorAll('.filter-btn').forEach(b => {{
    b.classList.toggle('active', b.dataset.filter === f);
  }});
}}
function toggleCard(meta) {{
  const content = meta.nextElementSibling;
  const arrow   = meta.querySelector('.arrow');
  const hidden  = content.style.display === 'none';
  content.style.display = hidden ? '' : 'none';
  if (arrow) arrow.textContent = hidden ? '▼' : '▶';
}}
function collapseAll(collapse) {{
  document.querySelectorAll('.scenario-content').forEach(el => {{
    el.style.display = collapse ? 'none' : '';
  }});
  document.querySelectorAll('.scenario .arrow').forEach(el => {{
    el.textContent = collapse ? '▶' : '▼';
  }});
}}
</script>
</body>
</html>"""

    html_dir = Path(output_dir) if output_dir else _HTML_DIR
    html_dir.mkdir(parents=True, exist_ok=True)
    stem = text_csv_path.stem.replace("_results", "_report")
    out_path = html_dir / (stem + ".html")
    out_path.write_text(html, encoding="utf-8")
    print(f"Report written to {out_path}")
    return out_path
