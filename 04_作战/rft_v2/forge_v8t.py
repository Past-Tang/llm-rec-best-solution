#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v8t 数据构造：懂用户×1（发射卡0715 卡8；官方种子四线最后一格定价实验）。
普查在档（08:45 + 09:0x 关A判例补测）：2,892 行、面谱 no_think 2,290/think 602、response 全带
<think> 头、final 双形态——list=SID JSON 数组 1,588 行 / dict={"logic_chain":…} 演化逻辑链 1,304 行，
与 prompt 尾部输出要求一一对应（零错位）；交叉 dict/think 602、dict/no_think 702、list/think 0。
配方：data_v8t = data_v8m 471,539 + 懂用户剥壳×1 2,892 = 474,431。
单变量账：vs v8m = 仅 +懂用户×1（+104MB ≈ 字节面 +12%，缓于懂推荐 +53%）。
剥壳承 forge_v8m 双面逻辑（/think|/no_think 尾判 + think 块剥离 + 面谱虚实互锁断言）；
final QC 为本件新增（关A 判例后双形态版）：JSON 可解析→list 走 SID 三元组逐元断言、
dict 走 logic_chain 结构断言（单键/name 非空/events 键集恒等），形态谱与交叉普查恒等。
"""
import json, random, re, sys, time

BASE = "/workspace/LLM-Rec"
SEED = f"{BASE}/02_数据/种子SFT数据/懂用户.jsonl"
V8M = f"{BASE}/04_作战/rft_v2/data_v8m.jsonl"
OUT = f"{BASE}/04_作战/rft_v2/data_v8t.jsonl"
STATS = f"{BASE}/04_作战/rft_v2/forge_v8t.stats.json"
DSI = f"{BASE}/02_数据/data/Explorer_LLM_Rec_Competition/demo/LLaMA-Factory/data/dataset_info.json"
KEYS = {"instruction", "input", "output", "system", "history"}
THINK_RE = re.compile(r'^<think>.*?</think>\s*', re.S)
SID_FULL_RE = re.compile(r'^<\|(prod|ad|video|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>$')
SID_ANY_RE = re.compile(r'<\|(prod|ad|video|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>')
FACE_EXP = {"think": 602, "no_think": 2290}
FORM_EXP = {"list": 1588, "dict": 1304}  # 关A 判例 0715: final 双形态普查恒等
EV_NOSID_EXP = 15                        # dict events 中 action 无 SID 的普查账
N_SEED, N_BASE, N_TOTAL = 2892, 471539, 474431


def log(m):
    print(f"[{time.strftime('%F %T')}] {m}", flush=True)


def die(m):
    log(f"FAIL: {m}")
    with open(STATS, "w", encoding="utf-8") as f:
        json.dump({"verdict": "FAIL", "reason": m}, f, ensure_ascii=False, indent=1)
    sys.exit(1)


def strip_row(d, i):
    p = d["prompt"].rstrip()
    if p.endswith("/no_think"):
        face, p = "no_think", p[: -len("/no_think")].rstrip()
    elif p.endswith("/think"):
        face, p = "think", p[: -len("/think")].rstrip()
    else:
        die(f"懂用户 行{i} prompt 尾缀非 /think|/no_think: {p[-60:]!r}")
    r = d["response"]
    m = THINK_RE.match(r)
    if not m:
        die(f"懂用户 行{i} response 无 <think> 头: {r[:60]!r}")
    blk = m.group(0)
    inner = blk[len("<think>"):blk.rfind("</think>")].strip()
    if (face == "think") != bool(inner):
        die(f"懂用户 行{i} 面谱互锁破裂: {face} 面配{'空' if not inner else '实'} think 块")
    return p, r[m.end():], face


def main():
    usr_rows, fc = [], {"think": 0, "no_think": 0}
    fm, dom_cnt, sid_n_dist, ev_n_dist, n_ev_nosid = {"list": 0, "dict": 0}, {}, {}, {}, 0
    n = 0
    for line in open(SEED, encoding="utf-8"):
        d = json.loads(line)
        p, r, face = strip_row(d, n)
        fc[face] += 1
        if "<think>" in r or "/think" in p or "/no_think" in p:
            die(f"懂用户 行{n} 剥壳残留")
        try:
            obj = json.loads(r)
        except Exception as e:
            die(f"懂用户 行{n} final 非合法 JSON: {e} | {r[:80]!r}")
        if isinstance(obj, list):
            form = "list"
            if not obj:
                die(f"懂用户 行{n} final 空数组")
            for x in obj:
                mm = SID_FULL_RE.match(x if isinstance(x, str) else "")
                if not mm:
                    die(f"懂用户 行{n} final 元素非 SID 三元组: {x!r}")
                dom_cnt[mm.group(1)] = dom_cnt.get(mm.group(1), 0) + 1
            sid_n_dist[len(obj)] = sid_n_dist.get(len(obj), 0) + 1
        elif isinstance(obj, dict):
            form = "dict"
            lc = obj.get("logic_chain")
            if set(obj) != {"logic_chain"} or not isinstance(lc, dict):
                die(f"懂用户 行{n} dict final 非 logic_chain 单键: {sorted(obj)}")
            evs = lc.get("events")
            if not (isinstance(lc.get("name"), str) and lc["name"].strip()
                    and isinstance(evs, list) and 1 <= len(evs) <= 10):
                die(f"懂用户 行{n} logic_chain 结构异常: keys={sorted(lc)}")
            for ev in evs:
                if not (isinstance(ev, dict) and set(ev) == {"date", "action", "logic"}):
                    die(f"懂用户 行{n} event 键集异常")
                if not SID_ANY_RE.search(ev["action"]):
                    n_ev_nosid += 1
            ev_n_dist[len(evs)] = ev_n_dist.get(len(evs), 0) + 1
        else:
            die(f"懂用户 行{n} final 类型异常: {type(obj)}")
        fm[form] += 1
        if face == "think" and form == "list":
            die(f"懂用户 行{n} 交叉异常: list/think（普查恒 0）")
        if not (0 < len(r) <= 3000):
            die(f"懂用户 行{n} final 长度异常 {len(r)}")
        usr_rows.append({"instruction": "", "input": p, "output": r,
                         "system": d.get("system", ""), "history": []})
        n += 1
    if n != N_SEED:
        die(f"懂用户 行数 {n} != {N_SEED}")
    if fc != FACE_EXP:
        die(f"懂用户 面谱 {fc} != 普查 {FACE_EXP}")
    if fm != FORM_EXP:
        die(f"懂用户 形态谱 {fm} != 普查 {FORM_EXP}")
    if n_ev_nosid != EV_NOSID_EXP:
        die(f"dict events 无SID action 数 {n_ev_nosid} != 普查 {EV_NOSID_EXP}")
    log(f"懂用户剥壳 OK n={n:,} 面谱={fc} 形态谱={fm} list域账={dom_cnt} "
        f"SID数top5={sorted(sid_n_dist.items(), key=lambda kv: -kv[1])[:5]} "
        f"events长top5={sorted(ev_n_dist.items(), key=lambda kv: -kv[1])[:5]} 无SID_event={n_ev_nosid}")

    rows = list(usr_rows)
    n_base = 0
    for line in open(V8M, encoding="utf-8"):
        d = json.loads(line)
        d.setdefault("system", "")
        d.setdefault("history", [])
        if set(d) != KEYS:
            die(f"base 行键集异常: {sorted(d)}")
        rows.append(d)
        n_base += 1
    if n_base != N_BASE:
        die(f"base(data_v8m) 行数 {n_base} != {N_BASE}")
    log(f"base=data_v8m 读入 {n_base:,}")

    random.Random(20260701).shuffle(rows)
    with open(OUT, "w", encoding="utf-8") as f:
        for d in rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    log(f"落盘 {len(rows):,} -> {OUT}")

    n_chk = 0
    for line in open(OUT, encoding="utf-8"):
        d = json.loads(line)
        if set(d) != KEYS:
            die(f"复验键集异常@{n_chk}")
        n_chk += 1
    if n_chk != N_TOTAL:
        die(f"复验行数 {n_chk} != {N_TOTAL}")

    info = json.load(open(DSI, encoding="utf-8"))
    info["data_v8t"] = {"file_name": OUT, "formatting": "alpaca",
                        "columns": {"prompt": "instruction", "query": "input", "response": "output",
                                    "system": "system", "history": "history"}}
    json.dump(info, open(DSI, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log("dataset_info.json 已注册 data_v8t")

    with open(STATS, "w", encoding="utf-8") as f:
        json.dump({"verdict": "PASS", "total": n_chk, "base_rows": n_base, "seed_rows": n,
                   "faces": fc, "forms": fm, "final_dom_list": dom_cnt,
                   "sid_n_top": sorted(sid_n_dist.items(), key=lambda kv: -kv[1])[:8],
                   "ev_n_top": sorted(ev_n_dist.items(), key=lambda kv: -kv[1])[:8],
                   "ev_nosid": n_ev_nosid,
                   "ts": time.strftime("%F %T")}, f, ensure_ascii=False, indent=1)
    log(f"QC 全检 PASS: total={n_chk:,} (base 471,539 + 懂用户 {n:,}); stats 落盘")


if __name__ == "__main__":
    main()
