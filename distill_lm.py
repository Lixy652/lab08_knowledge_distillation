# -*- coding: utf-8 -*-
"""
自制语言模型的知识蒸馏 —— 实验作业八参考实现（纯 CPU 可运行）
对应教材第 15 章《模型量化与压缩技术》+ 第 5 章《Transformer 模型》

单文件、零外部数据：
* 内置「仿古诗句」合成语料生成器（可复现，不依赖网络与版权）
* 字符级 GPT（教师 / 学生同架构、不同宽度）
* 三种训练：教师监督训练、学生从零训练（hard label）、学生蒸馏训练（soft label）
* 温度 T / 权重 alpha 消融、中间层特征蒸馏（FitNet）、动态量化导出、自蒸馏、教师规模扫描

常用命令（在本文件所在目录执行）：
python distill_lm.py --mode quick          # 30 秒自检，确认环境没问题
python distill_lm.py --mode teacher        # 只训练教师并保存 teacher.pt（约 5 分钟）
python distill_lm.py --mode all            # 教师 -> 从零学生 -> 蒸馏学生 -> 画图
python distill_lm.py --mode ablate         # 温度/权重 4x4 消融
python distill_lm.py --mode feat --feat 0.1    # 加特征蒸馏
python distill_lm.py --mode heatmap        # 由消融结果画 4x4 热力图
python distill_lm.py --mode quant          # 动态量化导出 int8 并对比体积/延迟
python distill_lm.py --mode selfdistill    # 自蒸馏（教师 = 训练更久的同构学生）
python distill_lm.py --mode teacherscan    # 教师规模扫描 C in {96,160,256}
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_distill")

# ============================================================
# 1. 语料：可复现的合成「仿古诗句」
# 四个槽位 + 两个收尾模板，使下一个字的分布具有真实的多样性与结构，
# 这正是教师 soft label 能传给学生的「暗知识」。
# ============================================================
SUBJ = ["小明", "小红", "阿远", "书生", "渔夫", "牧童", "旅人", "诗人",
        "少年", "故人", "归客", "山僧", "孤客", "游子"]
PLACE = ["江边", "山寺", "长亭", "渡口", "竹林", "荒村", "城楼", "野渡",
         "溪桥", "松间", "篱下", "窗前", "陌上", "舟中"]
VERB = ["望见", "听见", "想起", "送去", "寻到", "梦见", "独对", "遥看",
        "静听", "又逢", "恰遇", "空忆"]
OBJ = ["明月", "落花", "孤舟", "归雁", "残灯", "细雨", "青山", "晚钟",
       "白雪", "寒江", "疏星", "暮云", "秋草", "远帆"]
MOOD = ["心中惆怅", "不觉天明", "欣然归去", "泪湿衣襟", "长叹一声",
        "默然无语", "忽忆旧游", "欲语还休", "此意难平", "一笑而别"]
POEMS = [
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
    "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
    "空山不见人，但闻人语响。返景入深林，复照青苔上。",
    "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
    "独坐幽篁里，弹琴复长啸。深林人不知，明月来相照。",
    "月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。",
    "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
    "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
    "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。",
    "渭城朝雨浥轻尘，客舍青青柳色新。劝君更尽一杯酒，西出阳关无故人。",
    "寒雨连江夜入吴，平明送客楚山孤。洛阳亲友如相问，一片冰心在玉壶。",
    "独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。",
    "清明时节雨纷纷，路上行人欲断魂。借问酒家何处有，牧童遥指杏花村。",
    "朱雀桥边野草花，乌衣巷口夕阳斜。旧时王谢堂前燕，飞入寻常百姓家。",
]


def build_corpus(n_lines=3000, seed=0):
    """用固定随机种子生成训练语料；留出集换一个种子，检验泛化而非记忆。"""
    rng = random.Random(seed)
    out = []
    for i in range(n_lines):
        s, p, v, o, m = (rng.choice(x) for x in (SUBJ, PLACE, VERB, OBJ, MOOD))
        if i % 3 == 0:
            out.append("%s%s%s%s，%s。" % (s, p, v, o, m))
        elif i % 3 == 1:
            out.append("%s在%s%s了一夜%s，%s。" % (s, p, o, rng.choice(["看", "听", "守", "忆"]), m))
        else:
            out.append("%s%s道：“%s%s。”%s。" % (s, v, o, rng.choice(["正好", "可怜", "可惜", "依旧"]), m))
        out.append("\n")
    out.extend(POEMS)
    text = "".join(out)
    return text


# ============================================================
# 2. 字符级分词
# ============================================================
class Tokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = chars
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


# ============================================================
# 3. 最小 GPT（Pre-Norm，与作业三同构，只改宽度/深度）
# ============================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg["n_embd"]
        assert C % cfg["n_head"] == 0
        self.n_head = cfg["n_head"]
        self.c_attn = nn.Linear(C, 3 * C)
        self.c_proj = nn.Linear(C, C)
        self.drop = nn.Dropout(cfg["dropout"])
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg["block_size"], cfg["block_size"]))
            .view(1, 1, cfg["block_size"], cfg["block_size"]),
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg["n_embd"]
        self.fc = nn.Linear(C, 4 * C)
        self.proj = nn.Linear(4 * C, C)
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["n_embd"])
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg["n_embd"])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg["n_embd"])
        self.pos_emb = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], vocab_size, bias=False)
        self.apply(self._init)
        self.n_param = sum(p.numel() for p in self.parameters())

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, need_hidden=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        if need_hidden:
            return logits, loss, x
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg["block_size"]:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ============================================================
# 4. 批采样：scratch 与 kd 使用同一 generator 种子，保证数据顺序一致
# ============================================================
def get_batch(data, bs, block, gen):
    ix = torch.randint(len(data) - block - 1, (bs,), generator=gen)
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(model, data, bs, block, iters=20, seed=1234):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    tot = 0.0
    for _ in range(iters):
        x, y = get_batch(data, bs, block, gen)
        _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / iters


@torch.no_grad()
def agreement(student, teacher, data, bs, block, iters=10, seed=4321):
    """学生与教师在留出集上的 top-1 一致率，衡量「学到了多少」"""
    student.eval()
    teacher.eval()
    gen = torch.Generator().manual_seed(seed)
    hit = tot = 0
    for _ in range(iters):
        x, _ = get_batch(data, bs, block, gen)
        a = student(x)[0].argmax(-1)
        b = teacher(x)[0].argmax(-1)
        hit += (a == b).sum().item()
        tot += b.numel()
    student.train()
    teacher.train()
    return hit / tot


# ============================================================
# 5. 训练：hard label / soft label（KD）
# ============================================================
def train(model, data, iters, bs, block, lr=3e-3, seed=0,
          teacher=None, T=4.0, alpha=0.5, feat_weight=0.0,
          log_every=50, tag=""):
    """
    alpha = 0 -> 纯交叉熵（从零训练的对照组）
    alpha > 0 -> L = (1-alpha)*CE + alpha*T^2*KL(教师软标签 || 学生)
    feat_weight -> 额外加最后一层隐状态 MSE（FitNet 式特征蒸馏）
    """
    proj = None
    if teacher is not None and feat_weight > 0:
        proj = nn.Linear(model.cfg["n_embd"], teacher.cfg["n_embd"])
    params = list(model.parameters()) + (list(proj.parameters()) if proj is not None else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed + 1)
    hist = []

    if teacher is not None:
        teacher.eval()
    model.train()
    t0 = time.time()
    for it in range(iters):
        x, y = get_batch(data, bs, block, gen)
        logits = model(x)[0]
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        loss = ce
        if teacher is not None and alpha > 0:
            with torch.no_grad():
                t_logits = teacher(x)[0]
            # 关键：先 reshape 成 (B*T, V) 再算 KL。
            # 若直接对 (B,T,V) 调用 kl_div(reduction="batchmean")，它只除以 B，
            # 序列维会被重复累加，KL 被放大 T 倍（T=96 时数值大得离谱）。
            V = logits.size(-1)
            s_lp = F.log_softmax(logits.reshape(-1, V) / T, dim=-1)
            t_p = F.softmax(t_logits.reshape(-1, V) / T, dim=-1)
            kd = F.kl_div(s_lp, t_p, reduction="batchmean") * (T * T)
            loss = (1 - alpha) * ce + alpha * kd
        if proj is not None:
            _, _, hs = model(x, need_hidden=True)
            with torch.no_grad():
                _, _, ht = teacher(x, need_hidden=True)
            loss = loss + feat_weight * F.mse_loss(proj(hs), ht)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist.append(loss.item())
        if (it + 1) % log_every == 0 or it == 0:
            extra = ""
            if teacher is not None and alpha > 0:
                extra = " ce %.4f kd(T^2) %.4f" % (ce.item(), kd.item())
            print("  [%s] iter %4d/%d loss %.4f%s (%.1fs)"
                  % (tag or "train", it + 1, iters, loss.item(), extra, time.time() - t0),
                  flush=True)
    return hist, time.time() - t0


# ============================================================
# 6. 主流程
# ============================================================
TEACHER_CFG = dict(n_layer=2, n_head=4, n_embd=160, block_size=96, dropout=0.0)
STUDENT_CFG = dict(n_layer=2, n_head=4, n_embd=64, block_size=96, dropout=0.0)


def make_data(block):
    train_text = build_corpus(3000, seed=0)
    val_text = build_corpus(300, seed=999)  # 同语法、不同种子 -> 留出集
    tok = Tokenizer(train_text)
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    return tok, train_ids, val_ids


def run(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    tok, train_ids, val_ids = make_data(TEACHER_CFG["block_size"])
    bs, block = args.batch_size, TEACHER_CFG["block_size"]
    print("[data] vocab=%d train_chars=%d val_chars=%d block=%d batch=%d"
          % (tok.vocab_size, len(train_ids), len(val_ids), block, bs), flush=True)
    teacher_path = os.path.join(OUT_DIR, "teacher.pt")
    res = {"vocab": tok.vocab_size, "block": block, "batch": bs,
           "train_chars": int(len(train_ids)), "val_chars": int(len(val_ids)),
           "seed": args.seed, "lr": args.lr, "threads": torch.get_num_threads()}

    # ---------- (1) 教师：已训练过就直接复用，避免重复等待 ----------
    torch.manual_seed(args.seed)
    teacher = GPT(tok.vocab_size, TEACHER_CFG)
    print("[teacher] params = %.3f M" % (teacher.n_param / 1e6), flush=True)
    if os.path.exists(teacher_path) and not args.retrain and args.mode != "teacher":
        teacher.load_state_dict(torch.load(teacher_path, map_location="cpu"))
        tv = evaluate(teacher, val_ids, bs, block)
        res["teacher"] = {"params": teacher.n_param, "val_loss": tv,
                          "ppl": math.exp(tv), "time_s": 0.0, "curve": [],
                          "iters": 0, "cached": True}
        print("[teacher] 复用 %s val_loss=%.4f ppl=%.2f"
              % (os.path.basename(teacher_path), tv, math.exp(tv)), flush=True)
    else:
        th, ttime = train(teacher, train_ids, args.teacher_iters, bs, block,
                          lr=args.lr, seed=args.seed, log_every=args.log_every, tag="teacher")
        tv = evaluate(teacher, val_ids, bs, block)
        torch.save(teacher.state_dict(), teacher_path)
        res["teacher"] = {"params": teacher.n_param, "val_loss": tv,
                          "ppl": math.exp(tv), "time_s": ttime,
                          "curve": th, "iters": args.teacher_iters, "cached": False}
        print("[teacher] val_loss=%.4f ppl=%.2f %.1fs" % (tv, math.exp(tv), ttime), flush=True)

    if args.mode == "teacher":
        save(res, teacher, tok, tag="teacher")
        return

    if args.mode == "ablate":
        ablate(args, tok, train_ids, teacher, val_ids, bs, block, res)
        save(res, teacher, tok, tag="ablate")
        print("\n[done] 结果已写入 %s" % OUT_DIR, flush=True)
        return

    # ---------- (2) 学生配置 ----------
    torch.manual_seed(args.seed + 7)
    probe = GPT(tok.vocab_size, STUDENT_CFG)
    sp = probe.n_param
    print("[student] params = %.3f M (压缩率 %.1fx)"
          % (sp / 1e6, teacher.n_param / sp), flush=True)
    res["student_params"] = sp
    res["compress_ratio"] = teacher.n_param / sp

    def new_student():
        torch.manual_seed(args.seed + 7)  # 同初始化，保证公平对比
        return GPT(tok.vocab_size, STUDENT_CFG)

    def eval_student(m, hist, tcost, name, **extra):
        v = evaluate(m, val_ids, bs, block)
        row = {"params": sp, "val_loss": v, "ppl": math.exp(v), "time_s": tcost,
               "curve": hist, "iters": args.student_iters,
               "agree": agreement(m, teacher, val_ids, bs, block)}
        row.update(extra)
        print("[%s] val_loss=%.4f ppl=%.2f agree=%.3f %.1fs"
              % (name, v, math.exp(v), row["agree"], tcost), flush=True)
        res[name] = row
        return row

    # ---------- (3) 从零训练（hard label 对照组） ----------
    s0 = new_student()
    h0, t0 = train(s0, train_ids, args.student_iters, bs, block, lr=args.lr,
                   seed=args.seed, log_every=args.log_every, tag="scratch")
    row0 = eval_student(s0, h0, t0, "scratch", T=None, alpha=0.0)

    # ---------- (4) 蒸馏训练（soft label） ----------
    s1 = new_student()
    h1, t1 = train(s1, train_ids, args.student_iters, bs, block, lr=args.lr,
                   seed=args.seed, teacher=teacher, T=args.T, alpha=args.alpha,
                   feat_weight=args.feat, log_every=args.log_every, tag="kd")
    row1 = eval_student(s1, h1, t1, "kd", T=args.T, alpha=args.alpha, feat=args.feat)
    print("\n[对比] 从零 %.4f -> 蒸馏 %.4f 相对下降 %.1f%%"
          % (row0["val_loss"], row1["val_loss"],
             100 * (row0["val_loss"] - row1["val_loss"]) / row0["val_loss"]), flush=True)

    # ---------- (5) 生成样例（教师 / 从零 / 蒸馏同一提示词，便于横向比较） ----------
    prompt = "月"
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    samples = {}
    for name, m in (("teacher", teacher), ("scratch", s0), ("kd", s1)):
        with torch.no_grad():
            g = m.generate(ids.clone(), 60, temperature=0.9, top_k=20)
        samples[name] = tok.decode(g[0].tolist())
        print("[generate:%s] %s" % (name, samples[name].replace("\n", " / ")), flush=True)
    res["samples"] = samples
    res["sample"] = samples["kd"]
    res["prompt"] = prompt

    # ---------- (6) 画图 ----------
    plot_curves(res, os.path.join(OUT_DIR, "curves%s.png" % args.run_name))
    save(res, s1, tok, tag=args.mode + args.run_name)
    print("\n[done] 结果与曲线已写入 %s" % OUT_DIR, flush=True)


def ablate(args, tok, train_ids, teacher, val_ids, bs, block, res):
    """T x alpha 网格扫描。
    教师前向只算一次并缓存 32 个批次，之后每组只训练学生，
    相比每组重新跑教师前向可省掉约一半时间。"""
    grid_T = args.grid_T
    grid_a = args.grid_alpha
    V = tok.vocab_size
    gen = torch.Generator().manual_seed(2024)
    cache = []
    teacher.eval()
    with torch.no_grad():
        for _ in range(32):
            x, y = get_batch(train_ids, bs, block, gen)
            cache.append((x, y, teacher(x)[0]))
    rows = []
    for T in grid_T:
        for a in grid_a:
            torch.manual_seed(args.seed + 7)
            m = GPT(tok.vocab_size, STUDENT_CFG)
            opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01)
            m.train()
            t0 = time.time()
            for it in range(args.abl_iters):
                x, y, tl = cache[it % len(cache)]
                logits = m(x)[0]
                ce = F.cross_entropy(logits.view(-1, V), y.reshape(-1))
                s_lp = F.log_softmax(logits.reshape(-1, V) / T, dim=-1)
                t_p = F.softmax(tl.reshape(-1, V) / T, dim=-1)
                kd = F.kl_div(s_lp, t_p, reduction="batchmean") * (T * T)
                loss = (1 - a) * ce + a * kd
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            v = evaluate(m, val_ids, bs, block)
            ag = agreement(m, teacher, val_ids, bs, block)
            print("  T=%d alpha=%.1f -> val %.4f ppl %.2f agree %.3f (%.1fs)"
                  % (T, a, v, math.exp(v), ag, time.time() - t0), flush=True)
            rows.append({"T": T, "alpha": a, "val_loss": v, "ppl": math.exp(v),
                         "agree": ag, "time_s": time.time() - t0})
    res["ablation"] = rows
    res["abl_iters"] = args.abl_iters
    best = min(rows, key=lambda r: r["val_loss"])
    worst = max(rows, key=lambda r: r["val_loss"])
    print("[ablate] 最优 T=%d alpha=%.1f val=%.4f ｜ 最差 T=%d alpha=%.1f val=%.4f"
          % (best["T"], best["alpha"], best["val_loss"],
             worst["T"], worst["alpha"], worst["val_loss"]), flush=True)
    plot_curves(res, os.path.join(OUT_DIR, "curves.png"))


def setup_cjk_font(plt):
    """让图里的中文标题正常显示（Windows 自带雅黑/黑体，缺失时自动回落）。"""
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian",
                                       "Noto Sans CJK SC", "DejaVu Sans"]


def plot_curves(res, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[plot] 跳过绘图（%s）" % e)
        return
    setup_cjk_font(plt)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, lab in (("teacher", "Teacher"), ("scratch", "Student-scratch"), ("kd", "Student-KD")):
        if k in res and res[k]["curve"]:
            c = res[k]["curve"]
            ax[0].plot(range(1, len(c) + 1), c, label=lab, linewidth=1.2)
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("training loss")
    ax[0].set_title("training loss")
    if ax[0].get_legend_handles_labels()[0]:
        ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    if "ablation" in res:
        for T in sorted({r["T"] for r in res["ablation"]}):
            xs, ys = [], []
            for r in res["ablation"]:
                if r["T"] == T:
                    xs.append(r["alpha"])
                    ys.append(r["val_loss"])
            ax[1].plot(xs, ys, marker="o", label="T=%d" % T, linewidth=1.2)
        ax[1].set_xlabel("alpha")
        ax[1].set_ylabel("val loss")
        ax[1].set_title("temperature / weight ablation")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
    elif "scratch" in res and "kd" in res:
        bars = ax[1].bar(["scratch", "kd"], [res["scratch"]["val_loss"], res["kd"]["val_loss"]])
        for b in bars:
            ax[1].text(b.get_x() + b.get_width() / 2, b.get_height(),
                       "%.4f" % b.get_height(), ha="center", va="bottom", fontsize=8)
        ax[1].set_title("val loss")
        ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("[plot] saved %s" % path, flush=True)


def save(res, student, tok, tag="run"):
    with open(os.path.join(OUT_DIR, "results_%s.json" % tag), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    torch.save(student.state_dict(), os.path.join(OUT_DIR, "student_%s.pt" % tag))


# ============================================================
# 7. 拓展实验
# ============================================================
def mode_figures(args):
    """把三次运行的 JSON 合成报告用的两张图：
    图 1 = 训练损失曲线 + 留出集 loss 柱状对比；图 2 = T x alpha 消融曲线 + 热力图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def load(name):
        p = os.path.join(OUT_DIR, "results_%s.json" % name)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    rt, ra, rb = load("teacher"), load("all"), load("ablate")
    setup_cjk_font(plt)

    # ---------- 图 1 ----------
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8),
                           gridspec_kw={"width_ratios": [1.55, 1]})
    curves = [("teacher", rt, "教师 Teacher (0.737 M)"),
              ("scratch", ra, "学生·从零训练 (0.147 M)"),
              ("kd", ra, "学生·蒸馏训练 (0.147 M)")]
    for key, res, lab in curves:
        if res and res.get(key, {}).get("curve"):
            c = res[key]["curve"]
            ax[0].plot(range(1, len(c) + 1), c, label=lab, linewidth=1.2)
    if ax[0].get_legend_handles_labels()[0]:
        ax[0].legend(fontsize=8)
    ax[0].set_xlabel("迭代步 iteration")
    ax[0].set_ylabel("训练损失 training loss")
    ax[0].set_title("训练损失曲线")
    ax[0].grid(alpha=0.3)

    if ra:
        names, vals, cols = ["教师", "从零", "蒸馏"], [], ["#4c72b0", "#dd8452", "#55a868"]
        for k in ("teacher", "scratch", "kd"):
            vals.append(ra["teacher"]["val_loss"] if k == "teacher" else ra[k]["val_loss"])
        bars = ax[1].bar(names, vals, color=cols)
        for b, v in zip(bars, vals):
            ax[1].text(b.get_x() + b.get_width() / 2, b.get_height(), "%.4f" % v,
                       ha="center", va="bottom", fontsize=8)
        ax[1].set_ylabel("留出集 loss")
        ax[1].set_title("留出集损失对比")
        ax[1].set_ylim(0, max(vals) * 1.18)
        ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "fig1_train_curves.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print("[figures] saved %s" % p1, flush=True)

    if not (rb and "ablation" in rb):
        print("[figures] 缺少 results_ablate.json，跳过图 2")
        return
    rows = rb["ablation"]
    Ts = sorted({r["T"] for r in rows})
    As = sorted({r["alpha"] for r in rows})

    def grid(key):
        return [[next(r[key] for r in rows if r["T"] == T and abs(r["alpha"] - a) < 1e-9)
                 for a in As] for T in Ts]

    # ---------- 图 2 ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    for T in Ts:
        xs = [r["alpha"] for r in rows if r["T"] == T]
        ys = [r["val_loss"] for r in rows if r["T"] == T]
        ax[0].plot(xs, ys, marker="o", label="T=%d" % T, linewidth=1.3)
    ax[0].set_xlabel("权重 α")
    ax[0].set_ylabel("留出集 loss")
    ax[0].set_title("T × α 消融曲线（越低越好）")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    M = grid("val_loss")
    im = ax[1].imshow(M, cmap="viridis", aspect="auto")
    ax[1].set_xticks(range(len(As)))
    ax[1].set_xticklabels(["α=%.1f" % a for a in As], fontsize=8)
    ax[1].set_yticks(range(len(Ts)))
    ax[1].set_yticklabels(["T=%d" % T for T in Ts], fontsize=8)
    mn = min(min(r) for r in M)
    for i in range(len(Ts)):
        for j in range(len(As)):
            ax[1].text(j, i, "%.4f" % M[i][j], ha="center", va="center",
                       color="w" if M[i][j] > mn + 0.5 * (max(map(max, M)) - mn) else "k",
                       fontsize=8)
    ax[1].set_title("留出集 loss 热力图")
    fig.colorbar(im, ax=ax[1])
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "fig2_ablation.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print("[figures] saved %s" % p2, flush=True)


def mode_sanity(args):
    """数值自检：(a) KL 未 reshape 时被放大多少倍；(b) T^2 缩放对梯度的影响。"""
    torch.manual_seed(args.seed)
    tok, train_ids, val_ids = make_data(TEACHER_CFG["block_size"])
    bs, block = args.batch_size, TEACHER_CFG["block_size"]
    torch.manual_seed(args.seed)
    teacher = GPT(tok.vocab_size, TEACHER_CFG)
    p = os.path.join(OUT_DIR, "teacher.pt")
    if os.path.exists(p):
        teacher.load_state_dict(torch.load(p, map_location="cpu"))
    torch.manual_seed(args.seed + 7)
    student = GPT(tok.vocab_size, STUDENT_CFG)
    gen = torch.Generator().manual_seed(args.seed + 1)
    x, y = get_batch(train_ids, bs, block, gen)
    V = tok.vocab_size
    T = args.T

    with torch.no_grad():
        t_logits = teacher(x)[0]
    s_logits = student(x)[0]
    s_lp = F.log_softmax(s_logits.reshape(-1, V) / T, dim=-1)
    t_p = F.softmax(t_logits.reshape(-1, V) / T, dim=-1)
    kd_ok = F.kl_div(s_lp, t_p, reduction="batchmean") * (T * T)
    # 错误写法：直接对 (B, T, V) 求 batchmean，只除以 B，序列维被重复累加
    kd_bad = F.kl_div(F.log_softmax(s_logits / T, dim=-1),
                      F.softmax(t_logits / T, dim=-1), reduction="batchmean") * (T * T)
    print("[sanity-a] 正确 KL = %.4f ；未 reshape 的 KL = %.4f ；放大 %.1f 倍（约等于序列长 %d）"
          % (kd_ok.item(), kd_bad.item(), kd_bad.item() / kd_ok.item(), block), flush=True)

    # (b) 同一组 logits，KL 项对 logits 的梯度范数随 T 的变化
    rows = []
    for Tt in (1, 4, 8, 16):
        z = s_logits.detach().clone().requires_grad_(True)
        with torch.no_grad():
            pt = F.softmax(t_logits.reshape(-1, V) / Tt, dim=-1)
        kl = F.kl_div(F.log_softmax(z.reshape(-1, V) / Tt, dim=-1), pt, reduction="batchmean")
        g = torch.autograd.grad(kl, z)[0]
        scaled = g * (Tt * Tt)
        rows.append({"T": Tt, "grad_no_T2": g.norm().item() / z.numel(),
                     "grad_with_T2": scaled.norm().item() / z.numel()})
        print("[sanity-b] T=%2d 不加 T^2 的梯度 %.3e ；乘 T^2 后 %.3e"
              % (Tt, rows[-1]["grad_no_T2"], rows[-1]["grad_with_T2"]), flush=True)
    # 硬标签 CE 的梯度量级做参照
    ce = F.cross_entropy(s_logits.reshape(-1, V), y.reshape(-1))
    ce_g = torch.autograd.grad(ce, s_logits)[0].norm().item() / s_logits.numel()
    print("[sanity-b] 硬标签 CE 的梯度量级 %.3e（与乘 T^2 后的 KD 同量级）" % ce_g, flush=True)
    with open(os.path.join(OUT_DIR, "results_sanity.json"), "w", encoding="utf-8") as f:
        json.dump({"T": T, "kd_correct": kd_ok.item(), "kd_no_reshape": kd_bad.item(),
                   "bug_scale": kd_bad.item() / kd_ok.item(), "block": block,
                   "grad_rows": rows, "ce_grad": ce_g}, f, ensure_ascii=False, indent=2)


def mode_randteacher(args):
    """对照实验：把教师换成"未训练的随机教师"。
    用于区分蒸馏收益来自「教师的暗知识」还是仅仅来自「软标签的正则/平滑作用」。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    tok, train_ids, val_ids = make_data(TEACHER_CFG["block_size"])
    bs, block = args.batch_size, TEACHER_CFG["block_size"]
    torch.manual_seed(args.seed)
    real = GPT(tok.vocab_size, TEACHER_CFG)
    p = os.path.join(OUT_DIR, "teacher.pt")
    if os.path.exists(p):
        real.load_state_dict(torch.load(p, map_location="cpu"))
    rnd = GPT(tok.vocab_size, TEACHER_CFG)  # 未训练：输出近似均匀的随机分布
    rv = evaluate(rnd, val_ids, bs, block)
    print("[rand-teacher] val_loss=%.4f ppl=%.2f（≈均匀分布 ln V = %.4f）"
          % (rv, math.exp(rv), math.log(tok.vocab_size)), flush=True)

    res = {"rand_teacher_val_loss": rv, "ln_vocab": math.log(tok.vocab_size)}
    for name, t_net in (("scratch", None), ("kd_real", real), ("kd_random", rnd)):
        torch.manual_seed(args.seed + 7)
        m = GPT(tok.vocab_size, STUDENT_CFG)
        _, tt = train(m, train_ids, args.student_iters, bs, block, lr=args.lr,
                      seed=args.seed, teacher=t_net, T=args.T, alpha=args.alpha,
                      log_every=args.log_every, tag=name)
        v = evaluate(m, val_ids, bs, block)
        res[name] = {"val_loss": v, "ppl": math.exp(v), "time_s": tt,
                     "agree_with_real": agreement(m, real, val_ids, bs, block)}
        print("[%s] val_loss=%.4f ppl=%.2f agree(real)=%.3f"
              % (name, v, math.exp(v), res[name]["agree_with_real"]), flush=True)
    with open(os.path.join(OUT_DIR, "results_randteacher%s.json" % args.run_name),
              "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


def mode_heatmap(args):
    """B：把 4x4 消融结果画成热力图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(OUT_DIR, "results_ablate.json"), "r", encoding="utf-8") as f:
        res = json.load(f)
    setup_cjk_font(plt)
    rows = res["ablation"]
    Ts = sorted({r["T"] for r in rows})
    As = sorted({r["alpha"] for r in rows})
    M = [[next(r["val_loss"] for r in rows if r["T"] == T and abs(r["alpha"] - a) < 1e-9)
          for a in As] for T in Ts]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    im = axes[0].imshow(M, cmap="viridis", aspect="auto")
    axes[0].set_xticks(range(len(As)))
    axes[0].set_xticklabels(["alpha=%.1f" % a for a in As], fontsize=8)
    axes[0].set_yticks(range(len(Ts)))
    axes[0].set_yticklabels(["T=%d" % T for T in Ts], fontsize=8)
    for i in range(len(Ts)):
        for j in range(len(As)):
            axes[0].text(j, i, "%.4f" % M[i][j], ha="center", va="center",
                         color="w", fontsize=8)
    axes[0].set_title("val loss heatmap (T x alpha)")
    fig.colorbar(im, ax=axes[0])
    im2 = axes[1].imshow([[r["agree"] for r in rows if r["T"] == T] for T in Ts],
                         cmap="magma", aspect="auto")
    axes[1].set_xticks(range(len(As)))
    axes[1].set_xticklabels(["alpha=%.1f" % a for a in As], fontsize=8)
    axes[1].set_yticks(range(len(Ts)))
    axes[1].set_yticklabels(["T=%d" % T for T in Ts], fontsize=8)
    for i, T in enumerate(Ts):
        for j in range(len(As)):
            v = [r["agree"] for r in rows if r["T"] == T][j]
            axes[1].text(j, i, "%.3f" % v, ha="center", va="center", color="w", fontsize=8)
    axes[1].set_title("agreement with teacher")
    fig.colorbar(im2, ax=axes[1])
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "heatmap.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    best = min(rows, key=lambda r: r["val_loss"])
    print("[heatmap] saved %s ; best T=%d alpha=%.1f val=%.4f"
          % (p, best["T"], best["alpha"], best["val_loss"]), flush=True)


def mode_quant(args):
    """D：动态量化导出 int8，对比体积 / 前向延迟 / 留出集损失"""
    import copy

    torch.manual_seed(args.seed)
    tok, train_ids, val_ids = make_data(TEACHER_CFG["block_size"])
    bs, block = args.batch_size, TEACHER_CFG["block_size"]
    fp32 = GPT(tok.vocab_size, STUDENT_CFG)
    path = os.path.join(OUT_DIR, "student_all.pt")
    fp32.load_state_dict(torch.load(path, map_location="cpu"))
    fp32.eval()
    v32 = evaluate(fp32, val_ids, bs, block)

    try:
        import torch.ao.quantization as tq
    except Exception:  # 老版本 PyTorch
        import torch.quantization as tq
    q = tq.quantize_dynamic(copy.deepcopy(fp32).eval(), {nn.Linear}, dtype=torch.qint8)
    v8 = evaluate(q, val_ids, bs, block)

    def model_size(m, name):
        p = os.path.join(OUT_DIR, name)
        torch.save(m.state_dict(), p)
        sz = os.path.getsize(p)
        os.remove(p)
        return sz

    s32 = model_size(fp32, "_tmp_fp32.pt")
    s8 = model_size(q, "_tmp_int8.pt")

    def latency(m, reps=30):
        m.eval()
        x = torch.randint(0, tok.vocab_size, (1, block))
        with torch.no_grad():
            for _ in range(5):
                m(x)
            t0 = time.time()
            for _ in range(reps):
                m(x)
        return (time.time() - t0) / reps * 1000

    l32, l8 = latency(fp32), latency(q)
    out = {"fp32": {"val_loss": v32, "ppl": math.exp(v32), "bytes": s32, "ms_per_fwd": l32},
           "int8": {"val_loss": v8, "ppl": math.exp(v8), "bytes": s8, "ms_per_fwd": l8}}
    out["size_ratio"] = s32 / s8
    out["params"] = fp32.n_param
    with open(os.path.join(OUT_DIR, "results_quant.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[quant] fp32 %.4f (%.1f KB, %.2f ms) -> int8 %.4f (%.1f KB, %.2f ms) 压缩 %.2fx"
          % (v32, s32 / 1024, l32, v8, s8 / 1024, l8, s32 / s8), flush=True)


def mode_selfdistill(args):
    """C：自蒸馏 —— 教师是训练更久的同构学生（C=64，与目标学生同宽）"""
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    tok, train_ids, val_ids = make_data(TEACHER_CFG["block_size"])
    bs, block = args.batch_size, TEACHER_CFG["block_size"]
    print("[data] vocab=%d train_chars=%d val_chars=%d"
          % (tok.vocab_size, len(train_ids), len(val_ids)), flush=True)
    torch.manual_seed(args.seed)
    big_teacher = GPT(tok.vocab_size, TEACHER_CFG)
    tp = os.path.join(OUT_DIR, "teacher.pt")
    if os.path.exists(tp):
        big_teacher.load_state_dict(torch.load(tp, map_location="cpu"))

    res = {}
    # 1) 自蒸馏教师：同结构同宽度，只训练更久
    torch.manual_seed(args.seed + 11)
    st = GPT(tok.vocab_size, STUDENT_CFG)
    _, t_self = train(st, train_ids, args.self_teacher_iters, bs, block, lr=args.lr,
                      seed=args.seed + 11, log_every=args.log_every, tag="self-teacher")
    sv = evaluate(st, val_ids, bs, block)
    res["self_teacher"] = {"params": st.n_param, "val_loss": sv, "ppl": math.exp(sv),
                           "iters": args.self_teacher_iters, "time_s": t_self,
                           "agree_with_big": agreement(st, big_teacher, val_ids, bs, block)}
    print("[self-teacher] val_loss=%.4f ppl=%.2f (%.1fs)"
          % (sv, math.exp(sv), t_self), flush=True)

    # 2) 对照组：从零训练（同种子同数据顺序）
    torch.manual_seed(args.seed + 7)
    s0 = GPT(tok.vocab_size, STUDENT_CFG)
    _, t0 = train(s0, train_ids, args.student_iters, bs, block, lr=args.lr,
                  seed=args.seed, log_every=args.log_every, tag="scratch")
    v0 = evaluate(s0, val_ids, bs, block)
    res["scratch"] = {"val_loss": v0, "ppl": math.exp(v0), "time_s": t0,
                      "iters": args.student_iters}
    print("[scratch] val_loss=%.4f ppl=%.2f (%.1fs)" % (v0, math.exp(v0), t0), flush=True)

    # 3) 自蒸馏学生
    torch.manual_seed(args.seed + 7)
    s1 = GPT(tok.vocab_size, STUDENT_CFG)
    _, t1 = train(s1, train_ids, args.student_iters, bs, block, lr=args.lr,
                  seed=args.seed, teacher=st, T=args.T, alpha=args.alpha,
                  log_every=args.log_every, tag="self-kd")
    v1 = evaluate(s1, val_ids, bs, block)
    res["self_kd"] = {"val_loss": v1, "ppl": math.exp(v1), "time_s": t1,
                      "iters": args.student_iters,
                      "agree_with_self_teacher": agreement(s1, st, val_ids, bs, block),
                      "agree_with_big": agreement(s1, big_teacher, val_ids, bs, block)}
    print("[self-kd] val_loss=%.4f ppl=%.2f agree(self-t)=%.3f (%.1fs)"
          % (v1, math.exp(v1), res["self_kd"]["agree_with_self_teacher"], t1), flush=True)
    res["gain_pct"] = 100 * (v0 - v1) / v0
    print("[self-distill] 从零 %.4f -> 自蒸馏 %.4f 相对下降 %.1f%%"
          % (v0, v1, res["gain_pct"]), flush=True)
    with open(os.path.join(OUT_DIR, "results_selfdistill.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


def mode_teacherscan(args):
    """E：教师规模扫描 C in {96,160,256}，看蒸馏收益是否随教师变强而增大"""
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    tok, train_ids, val_ids = make_data(TEACHER_CFG["block_size"])
    bs, block = args.batch_size, TEACHER_CFG["block_size"]
    print("[data] vocab=%d train_chars=%d val_chars=%d"
          % (tok.vocab_size, len(train_ids), len(val_ids)), flush=True)
    rows = []
    sp = None
    # 基线：从零训练的学生（与其它实验同种子同数据顺序）
    torch.manual_seed(args.seed + 7)
    s0 = GPT(tok.vocab_size, STUDENT_CFG)
    sp = s0.n_param
    _, t0 = train(s0, train_ids, args.student_iters, bs, block, lr=args.lr,
                  seed=args.seed, log_every=args.log_every, tag="scratch")
    v0 = evaluate(s0, val_ids, bs, block)
    print("[scratch] val_loss=%.4f (%.1fs)" % (v0, t0), flush=True)

    for C in args.scan_widths:
        cfg = dict(n_layer=2, n_head=4, n_embd=C, block_size=96, dropout=0.0)
        torch.manual_seed(args.seed)
        tm = GPT(tok.vocab_size, cfg)
        p = os.path.join(OUT_DIR, "teacher_c%d.pt" % C)
        if C == 160 and os.path.exists(os.path.join(OUT_DIR, "teacher.pt")):
            tm.load_state_dict(torch.load(os.path.join(OUT_DIR, "teacher.pt"),
                                          map_location="cpu"))
            tv, ttime = evaluate(tm, val_ids, bs, block), 0.0
        elif os.path.exists(p) and not args.retrain:
            tm.load_state_dict(torch.load(p, map_location="cpu"))
            tv, ttime = evaluate(tm, val_ids, bs, block), 0.0
        else:
            _, ttime = train(tm, train_ids, args.teacher_iters, bs, block, lr=args.lr,
                             seed=args.seed, log_every=args.log_every, tag="teacher_c%d" % C)
            tv = evaluate(tm, val_ids, bs, block)
            torch.save(tm.state_dict(), p)
        torch.manual_seed(args.seed + 7)
        s1 = GPT(tok.vocab_size, STUDENT_CFG)
        _, t1 = train(s1, train_ids, args.student_iters, bs, block, lr=args.lr,
                      seed=args.seed, teacher=tm, T=args.T, alpha=args.alpha,
                      log_every=args.log_every, tag="kd_c%d" % C)
        v1 = evaluate(s1, val_ids, bs, block)
        row = {"teacher_n_embd": C, "teacher_params": tm.n_param, "teacher_val_loss": tv,
               "teacher_time_s": ttime, "kd_val_loss": v1, "kd_ppl": math.exp(v1),
               "kd_time_s": t1, "scratch_val_loss": v0,
               "gain_pct": 100 * (v0 - v1) / v0,
               "kd_agree": agreement(s1, tm, val_ids, bs, block)}
        rows.append(row)
        print("[scan C=%d] teacher %.4f (%.3fM) -> kd %.4f 收益 %.1f%%"
              % (C, tv, tm.n_param / 1e6, v1, row["gain_pct"]), flush=True)
    with open(os.path.join(OUT_DIR, "results_teacherscan.json"), "w", encoding="utf-8") as f:
        json.dump({"scratch_val_loss": v0, "student_params": sp, "rows": rows},
                  f, ensure_ascii=False, indent=2)


# ============================================================
# 8. 入口
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["quick", "teacher", "all", "ablate", "feat",
                             "heatmap", "quant", "selfdistill", "teacherscan",
                             "figures", "sanity", "randteacher"])
    ap.add_argument("--teacher-iters", type=int, default=200)
    ap.add_argument("--student-iters", type=int, default=120)
    ap.add_argument("--abl-iters", type=int, default=80)
    ap.add_argument("--self-teacher-iters", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--feat", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retrain", action="store_true",
                    help="忽略已缓存的 teacher.pt，重新训练教师")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--run-name", type=str, default="",
                    help="结果文件后缀，避免覆盖（如 _feat0.1）")
    ap.add_argument("--scan-widths", type=int, nargs="+", default=[96, 160, 256])
    args = ap.parse_args()

    if args.mode == "quick":
        global OUT_DIR
        OUT_DIR = os.path.join(OUT_DIR, "quick")
        args.teacher_iters, args.student_iters, args.abl_iters = 30, 20, 10
        args.batch_size, args.log_every = 8, 10
        args.mode = "all"

    if args.mode == "feat":
        args.mode = "all"
        if not args.run_name:
            args.run_name = "_feat%g" % args.feat

    if args.mode == "heatmap":
        mode_heatmap(args)
        return
    if args.mode == "figures":
        mode_figures(args)
        return
    if args.mode == "sanity":
        mode_sanity(args)
        return
    if args.mode == "randteacher":
        mode_randteacher(args)
        return
    if args.mode == "quant":
        mode_quant(args)
        return
    if args.mode == "selfdistill":
        mode_selfdistill(args)
        return
    if args.mode == "teacherscan":
        mode_teacherscan(args)
        return

    # 消融网格可用 --abl-iters 缩短
    args.grid_T = [1, 2, 4, 8]
    args.grid_alpha = [0.3, 0.5, 0.7, 0.9]
    run(args)


if __name__ == "__main__":
    main()
