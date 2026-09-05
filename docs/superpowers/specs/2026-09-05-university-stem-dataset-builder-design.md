# University STEM Dataset Builder V1 设计规范

- 日期：2026-09-05
- 状态：已批准（2026-09-05）
- 项目代号：University STEM Dataset Builder
- 目标：生产不少于 500,000 条大学理工/偏理工类高质量 SFT 试题数据
- 正式输出：`questions.jsonl + image/`
- 兼容契约：复用 `lingyi9909/shiti` 的 `QuestionRecordV1` exact 19-field Contract，并新增 `UniversitySTEMProfileV1`

---

## 1. 背景与目标

本项目用于从公开数据集、开放教材、Stack Exchange STEM 社区、真实大学课程/考试数据及商业供应商数据中，持续获取、筛选、验证并结构化大学 STEM 试题。

目标不是“尽量收集 50 万条 STEM 问答”，而是生产：

> **≥ 500,000 条 University STEM Problem-Solution Records，每条都必须有原始题目、原始答案、原始解析、可追溯来源，并通过严格质量门禁，最终映射为现有 19 字段 QuestionRecordV1。**

项目服务于大模型 SFT 数据生产，因此核心优化目标按以下优先级排序：

1. 题目—答案对应正确；
2. 答案与解析确实来自来源数据；
3. 大学层级与课程题属性准确；
4. 解析完整且支持答案；
5. 来源与许可信息可追溯；
6. 去重准确；
7. 最终 19 字段合法；
8. 最终数据量 ≥ 500,000；
9. 成本与吞吐。

核心质量原则：

- 宁可 Reject，也不让错题、错答案、伪解析进入 Accepted；
- LLM 不得生成、补写、改写或纠正正式答案与正式解析；
- LLM 可以用于分类、元数据提取、完整性判断、语义一致性验证和独立正确性验证；
- 任何 Accepted 数据都必须可追溯到 Raw Source；
- 任何关键判断都要保存 Evidence；
- 版权/许可当前不作为第一阶段采集阻断条件，但从第一天完整保留 provenance 与 license metadata；
- 数据规模目标以 Final Accepted 数量计算，不以 Raw 数量计算。

---

## 2. 项目边界

### 2.1 与现有 `shiti / Question Builder` 的关系

现有 Question Builder 负责：

```text
DOCX
  ↓
DOCX Parse / OCR / Formula / Table
  ↓
Question Split
  ↓
Answer Extract / Match
  ↓
Quality Gates
  ↓
QuestionRecordV1 exact 19 fields
```

University STEM Dataset Builder 负责：

```text
Dataset / JSON / JSONL / HTML / API / Open Textbook / Vendor Data
  ↓
Source Acquisition
  ↓
Normalization
  ↓
University STEM Filtering
  ↓
Problem / Answer / Analysis Filtering
  ↓
Quality Verification
  ↓
Dedup
  ↓
QuestionRecordV1 + UniversitySTEMProfileV1
```

两者为独立生产流水线，不把 Hugging Face、StackExchange、OpenStax、Vendor Adapter 等逻辑塞入现有 DOCX Question Builder。

两者只共享：

- Final 19-field Contract；
- `slim_md5_v1`；
- Markdown / LaTeX / 图片引用规范；
- 部分通用 Quality / Traceability 原则。

### 2.2 V1 明确做

- Hugging Face 数据集接入；
- GitHub 托管 JSON/JSONL 数据接入；
- StackMathQA / StackExchange STEM 接入；
- OpenStax / 开放教材 Problem-Solution 接入；
- 商业数据 Vendor Adapter 预留；
- University STEM 分类；
- Problem / Exercise 分类；
- 原始答案存在与抽取验证；
- 原始解析存在与完整性验证；
- Question / Answer / Analysis 一致性验证；
- 独立正确性验证；
- exact / near / semantic duplicate 检测；
- 19-field University Profile Export；
- provenance、license metadata、quality evidence；
- 5K Dry Run、50K Validation Pilot、最终 500K+ 分批生产。

### 2.3 V1 明确不做

- AI 生成解析；
- AI 补答案；
- AI 纠正原始答案；
- 为凑数量而生成 synthetic questions；
- 立即搭建 Web 管理后台；
- Kafka / Flink / Spark / Kubernetes 等重型数据平台；
- 首版解决全部版权法律结论；
- 首版覆盖所有大学学科；
- 把 StackExchange 普通技术问答直接当作大学试题；
- 直接对 200 万 Raw 全量调用强模型双审。

---

## 3. 成功标准

### 3.1 最终数据规模

正式 Release：

```text
Final Accepted >= 500,000
```

建议目标学科分布：

| 学科 | 目标数量 |
|---|---:|
| 数学 | 160,000 |
| 物理 | 90,000 |
| 计算机 | 90,000 |
| 统计学 | 50,000 |
| 电子/电气 | 50,000 |
| 化学 | 30,000 |
| 其他理工 | 30,000 |
| **合计** | **500,000** |

该分布为生产目标，不是硬性均衡要求；Pilot 后可按实际可获得性和训练价值调整。

### 3.2 正式质量 Gate

最低要求：

- exact 19-field Contract Valid = 100%；
- `answer_analysis` 非空且不为 `略` = 100%；
- Answer Source Traceable = 100%；
- Provenance Available = 100%；
- Exact Duplicate = 0；
- Gold Regression = PASS；
- Accepted 抽检中已知 Question/Answer Wrong Match = 0；
- Accepted 抽检中 University Precision >= 99%；
- Accepted 抽检中 Problem Precision >= 99%；
- Accepted 抽检中 Analysis Valid Precision >= 99%。

---

## 4. 数据来源策略

采用 Hybrid Strategy：

1. 公开数据承担主体；
2. Gold 数据承担标定；
3. Vendor 数据承担缺口补充与质量对标。

### 4.1 P0 公开主力：StackMathQA

StackMathQA 当前公开说明包含：

- Math Stack Exchange：827,439 个独立问题；
- MathOverflow：90,645；
- Statistics：103,024；
- Physics：117,318；
- 合计：1,138,426 个独立问题；
- one-question-one-answer 形式约 1,957,006 个 QA pair；
- 同时提供 100K / 200K / 400K / 800K / 1600K 等 selected subsets。

V1 不直接把 StackMathQA 视为“大学试题数据”，它只作为大规模候选池，必须经过 University / Problem / Answer / Analysis / Quality Gates。

### 4.2 P0 高质量大学源：OpenStax / 开放教材

OpenStax 类开放大学教材 Problem-Solution 数据用于：

- 提供明确 college-level 教材题；
- 提供高质量 Gold / High Confidence 数据；
- 提供结构清晰的 problem-solution pair；
- 为 University / Problem / Analysis Gate 做 calibration。

OpenStaxQA 公开研究报告描述从 43 本大学教材得到约 18,332 个去重 Problem-Solution pairs。V1 可以使用公开数据集，也可以根据来源结构构建自己的 OpenTextbookAdapter。

### 4.3 Gold Sources

首批 Gold Seed：

- STEMQ：667 个 question-solution pairs，来自 7 所大学、27 门 STEM 课程；
- SciBench：college-level 数学、化学、物理教材题；
- CFE-Bench：真实大学作业/考试题；
- OpenStax 高置信样本。

Gold 数据不承担 500K 数量目标，承担 Gate Calibration、Regression 与人工质量基准。

### 4.4 P1 扩展公开来源

Pilot 成功后扩展：

- Electronics Stack Exchange；
- Computer Science Stack Exchange；
- Chemistry Stack Exchange；
- DSP Stack Exchange；
- Engineering Stack Exchange；
- 其他具有足够题解密度的 STEM 子站。

P1 数据仍必须经过同一质量流水线，不能根据站点名称直接 Accepted。

### 4.5 Vendor Sources

当前重点候选：

- Nexdata：约 1.5M 英文 university STEM questions，字段包含 title / answer / parse / subject / grade / question type；
- DataoceanAI：数据卡描述 200K+ university-level math / physics / chemistry / computer science problems，含 final answer 与 detailed explanation，但 Hugging Face 仓库当前无公开数据文件，需供应商验货。

Vendor 数据在 Pilot 前不作为必需依赖。

采购 Decision Gate：

```text
预计公开来源 Accepted >= 600K
→ 不采购主库，仅做 Vendor sample 质量对标

预计公开来源 Accepted 350K~600K
→ Vendor 补 100K~200K

预计公开来源 Accepted < 350K
→ Vendor 升级为主力源
```

---

## 5. 总体架构

```text
Source Adapters
   ├─ HFDatasetAdapter
   ├─ GitDatasetAdapter
   ├─ StackExchangeAdapter
   ├─ OpenTextbookAdapter
   └─ VendorDatasetAdapter
          ↓
RawSourceRecord
          ↓
Raw Store
          ↓
Source Normalization
          ↓
NormalizedQA
          ↓
Gate 0 Integrity / Provenance
          ↓
Gate 1 STEM + University
          ↓
Gate 2 Problem / Exercise
          ↓
Gate 3 Original Answer
          ↓
Gate 4 Original Analysis
          ↓
Early Dedup
          ↓
Gate 5 QA / Analysis Quality
          ↓
Independent Correctness Verification
          ↓
Final Dedup
          ↓
UniversityQuestionIR
          ↓
QuestionRecordV1 + UniversitySTEMProfileV1
          ↓
questions.jsonl + image/
```

---

## 6. Source Adapter Contract

所有来源适配器只负责：

- 获取数据；
- 保留原始格式；
- 生成稳定 source identity；
- 保存原始 provenance；
- 不做大学题判断；
- 不修改题目、答案和解析内容。

建议接口：

```python
class SourceAdapter(Protocol):
    def discover(self, config) -> Iterable[SourceDescriptor]: ...
    def acquire(self, descriptor) -> Iterable[RawSourceRecord]: ...
    def checkpoint(self) -> AdapterCheckpoint: ...
```

`RawSourceRecord` 至少包含：

```json
{
  "record_id": "raw_...",
  "source_type": "dataset",
  "source_dataset": "stackmathqa",
  "source_id": "...",
  "source_url": "...",
  "raw_question": "...",
  "raw_answer": "...",
  "raw_analysis": "...",
  "raw_payload": {},
  "metadata": {},
  "license_metadata": {},
  "raw_sha256": "..."
}
```

Raw 内容不可被后续覆盖。

---

## 7. 四层数据模型

### 7.1 Layer 1 — RawSourceRecord

忠实保存来源记录，不做语义修改。

### 7.2 Layer 2 — NormalizedQA

不同来源统一到内部格式：

```json
{
  "record_id": "norm_...",
  "source_record_id": "raw_...",
  "question": "...",
  "answer": "...",
  "analysis": "...",
  "subject_candidates": [],
  "images": [],
  "metadata": {},
  "normalization_evidence": {}
}
```

Normalization 允许：

- HTML → Markdown；
- MathML → LaTeX；
- 统一换行；
- 统一无意义空白；
- 结构化选项；
- 提取图片资产；
- 修复明确的编码问题。

Normalization 禁止：

- 改数字；
- 改公式含义；
- 改条件；
- 改选项；
- 改答案；
- 改解析结论；
- 使用 LLM 重写内容。

### 7.3 Layer 3 — UniversityQuestionCandidate / UniversityQuestionIR

通过 Gate 后的核心内部模型：

```json
{
  "candidate_id": "uq_...",
  "source_record_id": "...",
  "question": {
    "raw": "...",
    "normalized": "...",
    "assets": []
  },
  "answer": {
    "raw": "...",
    "final_answer": "...",
    "source_span": "..."
  },
  "analysis": {
    "raw": "...",
    "type": "DERIVATION"
  },
  "classification": {
    "discipline": "MATHEMATICS",
    "course": "LINEAR_ALGEBRA",
    "level": "UNDERGRADUATE",
    "problem_type": "CALCULATION"
  },
  "metadata": {
    "knowledge_points": [],
    "exam_points": [],
    "language": "en"
  },
  "quality": {},
  "provenance": {},
  "dedup": {}
}
```

### 7.4 Layer 4 — FinalQuestionRecord

仅 Accepted 数据转换为现有 exact 19-field Contract。

---

## 8. Gate 0 — Integrity / Provenance Gate

检查：

- Raw payload 可解析；
- question 不为空；
- answer source 不为空；
- source identity 稳定；
- source URL / dataset identity 至少存在一种；
- 原始 hash 已保存；
- license metadata 已记录当前已知状态；
- 图片引用若存在，对应资产可获取或显式标记 missing。

失败 Reason：

- `SOURCE_INCOMPLETE`
- `SOURCE_CORRUPTED`
- `PROVENANCE_MISSING`
- `IMAGE_SOURCE_MISSING`

许可状态允许暂时为 `UNREVIEWED`，但 provenance 本身不能缺失。

---

## 9. Gate 1 — STEM / University Gate

### 9.1 STEM 分类

内部一级学科：

```text
MATHEMATICS
STATISTICS
PHYSICS
CHEMISTRY
COMPUTER_SCIENCE
ELECTRONICS
ELECTRICAL_ENGINEERING
AUTOMATION
MECHANICAL_ENGINEERING
MATERIALS
CIVIL_ENGINEERING
BIOLOGICAL_SCIENCE
OTHER_STEM
NON_STEM
```

### 9.2 University Level

大学层级按“解题所需知识”判断，不按页面是否写有 college/university 判断。

示例：

- 特征值 / 特征向量 → university；
- Lebesgue 积分 → university；
- Maxwell 方程推导 → university；
- AVL / Red-Black Tree 复杂度证明 → university；
- 小学四则运算 → reject；
- 普通软件安装问答 → reject。

### 9.3 Pilot 阈值

```text
score >= 0.98
→ PASS

0.90 <= score < 0.98
→ Independent Verifier

score < 0.90
→ REJECT
```

Primary 与 Verifier 冲突 → REJECT。

Reason：

- `NON_STEM`
- `NOT_UNIVERSITY_LEVEL`
- `UNIVERSITY_LEVEL_UNCERTAIN`

---

## 10. Gate 2 — Problem / Exercise Gate

大学知识问答不等于大学课程试题。

允许：

```text
CALCULATION
PROOF
DERIVATION
CONCEPTUAL
MULTIPLE_CHOICE
PROGRAMMING
ALGORITHM
ENGINEERING
```

拒绝：

```text
SOFTWARE_USAGE
DEBUG_HELP
CAREER_ADVICE
OPINION
RESOURCE_REQUEST
DISCUSSION
NEWS
META
```

规则与模型共同判断：

- 明确“求 / 计算 / 证明 / 推导 / 分析 / determine / calculate / prove / derive”等结构为正证据；
- StackExchange tag、title、正文结构为辅助证据；
- 纯技术排障、工具推荐、观点讨论为负证据。

Pilot 阈值同 Gate 1。

Reason：

- `NOT_PROBLEM`
- `PROBLEM_TYPE_UNCERTAIN`

---

## 11. Gate 3 — Original Answer Gate

必须同时满足：

1. 原始来源存在答案；
2. 答案确实属于该题；
3. 可以从原始来源中可靠得到 final answer；
4. final answer 有 source span / deterministic origin；
5. 不需要 LLM 自己解题才能得到答案。

允许：

- 从来源的 `correct_answer` 字段直接取值；
- 从结构化 solution 中确定性提取 final result；
- 从来源明确的 answer section 提取结果。

禁止：

- LLM 看到解析后自行总结最终答案；
- LLM 重新求解后替换原答案；
- 因为原答案“看起来错”而修正。

建议：

```text
answer_source_exists = true
answer_extract_score >= 0.995
question_answer_alignment >= 0.995
```

失败 Reason：

- `ANSWER_MISSING`
- `ANSWER_NOT_EXTRACTABLE`
- `QUESTION_ANSWER_MISMATCH`
- `ANSWER_SOURCE_UNTRACEABLE`

---

## 12. Gate 4 — Original Analysis Gate

UniversitySTEMProfileV1 强制要求真正的原始解析。

以下全部 Reject：

- 空字符串；
- `略`；
- 只有最终答案；
- 只有链接；
- 只有参考页码；
- “同上”；
- 无法证明属于当前题的通用说明。

有效解析类型：

```text
DERIVATION
STEP_BY_STEP
PROOF
CONCEPTUAL_REASONING
ALGORITHM_EXPLANATION
CODE_EXPLANATION
ENGINEERING_REASONING
```

LLM 只返回 classification / score / issues，不改写解析。

Pilot：

```text
analysis_score >= 0.98 → PASS
0.90 <= analysis_score < 0.98 → Verifier
< 0.90 → REJECT
```

失败 Reason：

- `ANALYSIS_MISSING`
- `ANALYSIS_TOO_SHALLOW`
- `ANALYSIS_UNRELATED`
- `ANALYSIS_UNCERTAIN`

---

## 13. Early Dedup

在高成本 Verification 前进行低成本去重：

### Level 1：Exact

复用 `slim_md5_v1`，基于 normalized question canonicalization 计算。

### Level 2：Near Duplicate Candidate Recall

使用：

- canonicalized text hash；
- MinHash / SimHash；
- n-gram overlap；
- 数学表达式辅助特征。

Early Dedup 只删除确定 exact duplicate；近似重复仅形成候选关系，避免误删变量变化题、数值变化题、同模板不同条件题。

---

## 14. Gate 5 — Quality Verification

### 14.1 Deterministic Verification

优先程序化验证：

- 数学：代数、方程、导数、积分、矩阵、数值结果等可调用 SymPy / 数值方法；
- 编程：具备输入/输出时可在隔离 sandbox 执行；
- 选择题：答案选项与解析结论一致性；
- 明确数值型题：单位、结果范围等 deterministic checks。

程序只验证，不修改原始答案。

### 14.2 Semantic Alignment Verifier

独立模型仅判断：

- analysis 是否回答当前 question；
- analysis 是否支持原始 answer；
- 是否存在明显逻辑矛盾；
- 是否疑似跨题错配。

输出示意：

```json
{
  "verdict": "PASS",
  "alignment_score": 0.998,
  "issues": []
}
```

### 14.3 Independent Correctness Verifier

允许独立模型内部重新求解或推导，仅用于验证原始 answer / analysis。

输出：

```text
PASS
FAIL
UNCERTAIN
```

- PASS → 可继续；
- FAIL → REJECT；
- UNCERTAIN → REJECT。

Verifier 内部推理不进入正式训练数据。

### 14.4 综合通过条件

Pilot v1：

```text
STEM >= 0.98
University >= 0.98
Problem >= 0.98
Answer Source = TRUE
Answer Extract >= 0.995
Analysis >= 0.98
QA Alignment >= 0.995
Independent Verification = PASS
```

阈值必须配置化、版本化；Pilot 后只能通过 Gold Regression 和人工抽检调整。

---

## 15. Final Dedup

最终去重采用三层：

1. `slim_md5_v1` exact；
2. MinHash / SimHash near-duplicate recall；
3. Embedding semantic candidate recall + Duplicate Verifier。

Embedding 不能直接作为删除规则。

Duplicate Verifier 输出：

```text
SAME_PROBLEM → 去重
VARIANT → 保留
DIFFERENT → 保留
UNCERTAIN → 保留并标记，或进入人工审计样本
```

特别防止：

- 相同模板不同参数；
- 相同结论不同条件；
- 同一道题不同语言版本；
- 同题多个来源；
- StackExchange repost / quote；
- 同一题多个高票答案重复形成多条 SFT。

同一题多个高质量来源时，优先保留 provenance 更完整、analysis 更强、license 状态更清晰的一条。

---

## 16. Quality Evidence

每个 Gate 都必须保留 Evidence，而不是只保留 `accepted=true`。

示意：

```json
{
  "university": {
    "score": 0.994,
    "provider": "...",
    "model": "...",
    "prompt_version": "university_v1"
  },
  "problem": {
    "score": 0.998,
    "problem_type": "PROOF"
  },
  "answer": {
    "source_span": "...",
    "extract_score": 0.999
  },
  "analysis": {
    "type": "DERIVATION",
    "score": 0.992
  },
  "alignment": {
    "score": 0.998,
    "verdict": "PASS"
  },
  "correctness": {
    "verdict": "PASS",
    "provider": "...",
    "model": "..."
  }
}
```

任一 Prompt / Model / Threshold 变化后，应能按 stage 重跑，而不是重新采集全部数据。

---

## 17. Reject Model

所有 Reject 必须落盘。

统一 reason codes 至少包括：

```text
SOURCE_INCOMPLETE
SOURCE_CORRUPTED
PROVENANCE_MISSING
IMAGE_SOURCE_MISSING
NON_STEM
NOT_UNIVERSITY_LEVEL
UNIVERSITY_LEVEL_UNCERTAIN
NOT_PROBLEM
PROBLEM_TYPE_UNCERTAIN
ANSWER_MISSING
ANSWER_NOT_EXTRACTABLE
ANSWER_SOURCE_UNTRACEABLE
QUESTION_ANSWER_MISMATCH
ANALYSIS_MISSING
ANALYSIS_TOO_SHALLOW
ANALYSIS_UNRELATED
ANALYSIS_UNCERTAIN
ANALYSIS_ANSWER_MISMATCH
CORRECTNESS_FAILED
CORRECTNESS_UNCERTAIN
DUPLICATE
LANGUAGE_UNRESOLVED
SCHEMA_VALIDATION_FAILED
```

Rejected Record 保存：

- source_record_id；
- stage；
- reason_code；
- details；
- evidence；
- pipeline/config/prompt/model version。

---

## 18. QuestionRecordV1 + UniversitySTEMProfileV1

最终 exact 19 fields：

```text
text_question
is_pic_included
text_answer
answer_analysis
text_course
text_grade_level
text_grade
knowledge_points
exam_points
publisher
text_paper
textbook_version
static_info
language
text_year
entrance_exam_type
text_city
question_type
competition_event
```

字段数量、字段名、字段类型保持不变。

### 18.1 Profile 标识

Profile 不新增字段，写入 `static_info`：

```json
{
  "contract_version": "question_record_v1",
  "profile": "university_stem_v1"
}
```

K12 Validator 与 University Validator 必须独立，不能静默扩大原 K12 枚举。

### 18.2 `text_question`

- Markdown；
- 公式 LaTeX；
- 简单表格 Markdown；
- 复杂表格 HTML；
- 图片 `<img src="image/...">`；
- 尽量忠实于来源；
- 禁止 LLM 重写正式题目。

### 18.3 `text_answer`

必须来自来源中的明确最终答案。

### 18.4 `answer_analysis`

University Profile：

```text
non-empty
AND != "略"
AND passes Analysis Gate
```

### 18.5 `text_grade`

允许：

```text
大学
未知
```

Accepted University 数据正常应为 `大学`。

### 18.6 `text_grade_level`

允许：

```text
本科一年级
本科二年级
本科三年级
本科四年级
本科
硕士
博士
研究生
大学未知
```

禁止模型无依据猜年级；只能确定本科层级时填 `本科`。

### 18.7 `text_course`

University Profile 一级学科枚举：

```text
数学
统计学
物理
化学
计算机
电子信息
电气工程
自动化
机械工程
材料科学
土木工程
生物科学
其他理工
未知
```

课程名（如 Linear Algebra / Operating Systems）不进入 `text_course`，进入内部 metadata / `static_info.course_name`。

### 18.8 `knowledge_points`

格式：

```text
知识点1; 知识点2; 知识点3
```

来源明确值优先；允许 LLM 做 metadata inference，但必须保存 `knowledge_points_source`。

### 18.9 `exam_points`

表示主要考察能力，例如：

```text
积分计算
证明
公式推导
算法分析
程序设计
电路分析
物理建模
概念理解
```

### 18.10 `publisher / textbook_version / text_paper`

- 有明确来源才填；
- OpenStax 等可写 publisher；
- `text_paper` 可保存来源教材/课程/站点标题；
- 禁止虚构大学或试卷名。

### 18.11 `entrance_exam_type`

V1 普通大学课程题统一 `未知`。

### 18.12 `text_city / competition_event`

无可靠来源填空字符串。

### 18.13 `question_type`

最终枚举继续保持：

```text
判断题
选择题
填空题
问答题
其他题型
未知
```

内部保留更细类型：

```text
CALCULATION
PROOF
DERIVATION
PROGRAMMING
ALGORITHM
CONCEPTUAL
MULTIPLE_CHOICE
ENGINEERING_DESIGN
```

---

## 19. `static_info` 与 Provenance

`static_info` 保持 string，内容必须为合法 JSON。

建议最小内容：

```json
{
  "slim_question_md5": "...",
  "copyright": "0",
  "contract_version": "question_record_v1",
  "profile": "university_stem_v1",
  "source_dataset": "stackmathqa",
  "source_id": "...",
  "source_url": "...",
  "source_license": "...",
  "source_question_hash": "...",
  "source_answer_hash": "...",
  "course_name": "Linear Algebra",
  "pipeline_version": "university_dataset_builder_0.1.0",
  "quality_tier": "HIGH_CONFIDENCE"
}
```

内部 Provenance 独立保存更完整信息：

```json
{
  "source_type": "dataset",
  "source_dataset": "stackmathqa",
  "source_site": "math.stackexchange.com",
  "source_id": "...",
  "source_url": "...",
  "author": "...",
  "created_at": "...",
  "score": 42,
  "license": {
    "declared": "...",
    "original_source": "...",
    "status": "UNREVIEWED"
  },
  "acquired_at": "...",
  "adapter": "stackmathqa_v1",
  "raw_sha256": "...",
  "question_source": {},
  "answer_source": {}
}
```

License status：

```text
UNREVIEWED
APPROVED
RESTRICTED
UNKNOWN
EXCLUDED
```

`copyright` 字段继续沿用现有 QuestionRecord 语义，不在本项目中擅自重定义。

---

## 20. 图片与多模态

V1 优先纯文本生产：

```text
纯文本 Accepted 目标 >= 450K
多模态 Accepted 目标 <= 50K
```

多模态进入正式集必须：

- 原图可合法获取并落盘；
- 原题中图片位置可恢复；
- 图片 hash 稳定；
- 正式题干使用 `<img src="image/...">`；
- 关键图缺失 → REJECT；
- answer / analysis 仍完整。

不为了增加多模态比例而接受缺图题。

---

## 21. 工程结构

推荐新项目目录：

```text
university-stem-dataset-builder/
├── pyproject.toml
├── README.md
├── config/
├── prompts/
├── docs/
│   └── superpowers/
│       ├── specs/
│       └── plans/
├── src/university_stem_dataset_builder/
│   ├── source/
│   │   ├── huggingface/
│   │   ├── stackexchange/
│   │   ├── openstax/
│   │   ├── github/
│   │   └── vendor/
│   ├── domain/
│   ├── normalize/
│   ├── classification/
│   ├── answer/
│   ├── analysis/
│   ├── verification/
│   ├── dedup/
│   ├── metadata/
│   ├── quality/
│   ├── export/
│   ├── storage/
│   ├── cache/
│   └── pipeline/
├── tests/
└── fixtures/
```

模块必须以 Domain Contract 通信；Provider / Source DTO 不得渗透到核心 Domain。

---

## 22. 存储

V1 推荐：

```text
SQLite
+
DuckDB
+
Parquet
+
Local Filesystem
```

职责：

- Parquet：Raw / Normalized / Candidate / Accepted / Rejected 大批量记录；
- DuckDB：批量统计、筛选、数据分析、Pilot 报告；
- SQLite：Run、Stage、Cache、Provider Call、Checkpoint、Error；
- Filesystem：图片、raw snapshots、logs、reports；
- JSONL：仅正式交付。

不使用 Kafka / Spark / Flink。

---

## 23. Stage State Machine

每条记录：

```text
ACQUIRED
NORMALIZED
CLASSIFIED
ANSWER_VALIDATED
ANALYSIS_VALIDATED
EARLY_DEDUPED
VERIFIED
FINAL_DEDUPED
ACCEPTED
REJECTED
```

保存：

```text
stage
pipeline_version
config_version
prompt_version
provider
model
attempt
updated_at
```

失败或中断后可以从最后有效 Stage 恢复。

---

## 24. 缓存与幂等

所有高成本模型/Embedding 调用必须缓存。

Cache Key：

```text
content_hash
+ gate_name
+ provider
+ model
+ prompt_version
+ config_version
```

Run Fingerprint：

```text
source_snapshot_hash
+ config_hash
+ pipeline_version
```

目标：

- Prompt 未变化时不重复调用；
- 某个 Gate 调整时只重跑该 Gate 之后的数据；
- Source Acquisition 不因下游修改重复抓取。

---

## 25. 模型调用成本分层

### Tier 0 — Rules

零模型成本：

- 空值；
- 格式；
- HTML/Markdown；
- 明显 tag；
- 长度；
- 选择题结构；
- exact duplicate；
- MD5；
- 明显非 STEM；
- 明显 shallow answer/analysis。

### Tier 1 — Low-cost Classifier

处理：

- STEM；
- University；
- Problem；
- Analysis 类型；
- 课程/知识点 metadata。

### Tier 2 — Strong Verifier

只处理：

- 边界分类样本；
- Answer / Analysis Alignment；
- Correctness；
- Duplicate ambiguous candidates；
- 高价值 uncertain samples。

避免对全部 Raw 进行双强模型验证。

---

## 26. Provider Contract

LLM / Embedding Provider 必须抽象：

```text
ClassifierProvider
VerifierProvider
EmbeddingProvider
```

支持 Primary / Fallback / Independent Verifier。

Provider 需要：

- async；
- batch；
- semaphore；
- rate limit；
- retry；
- exponential backoff + jitter；
- circuit breaker；
- request id；
- usage / token / estimated cost；
- model identity；
- calibration profile。

业务层不得绑定具体厂商。

---

## 27. 并发建议

初始可配置：

```yaml
concurrency:
  acquisition: 8
  normalization: 16
  rule_gate: 32
  classifier: 16
  verifier: 8
  embedding_batch_workers: 4
```

真实值由 Provider 限流和机器资源决定。

---

## 28. Validation Pilot

### Phase 0 — Gold Seed

导入：

- STEMQ 全量；
- SciBench 全量；
- CFE-Bench 全量；
- OpenStax 高质量 subset。

目标：验证 Gate Precision 与 Gold Regression，不承担数量。

### Phase 1 — 5K Dry Run

StackMathQA 分层：

```text
Math          2,000
Physics       1,000
Statistics    1,000
MathOverflow  1,000
```

重点暴露工程问题：

- HTML；
- LaTeX；
- 超长帖子；
- 多答案；
- 引用块；
- 代码块；
- 链接；
- 图片；
- 复杂 proof；
- StackExchange 普通讨论帖；
- answer final extraction；
- duplicate。

5K Dry Run Gate 通过后才运行 50K。

### Phase 2 — 50K Pilot

建议分层：

```text
Math          20,000
MathOverflow   5,000
Statistics    10,000
Physics       10,000
Other / P1     5,000
```

输出：

```text
pilot_report.json
accepted.parquet
rejected.parquet
questions.jsonl
```

### 28.1 Pilot 必须统计

```text
University Rate
Problem Rate
Answer Source Valid Rate
Answer Extractable Rate
Analysis Complete Rate
Alignment Pass Rate
Correctness Pass Rate
Dedup Rate
Final Acceptance Rate
```

并按学科、来源拆分。

### 28.2 人工抽检

Accepted 中建议抽检：

```text
数学        500
物理        500
统计        300
CS          300
电子/其他   300
Gold        全量回归
```

合计约 1,900 + Gold。

### 28.3 Pilot Decision

以真实 Final Acceptance Rate 倒推 Raw Pool：

```text
raw_needed = 500000 / final_acceptance_rate
```

例如：

- 35% → 约 1.43M Raw；
- 25% → 2.0M Raw；
- 15% → 约 3.33M Raw。

据此决定公开数据扩容还是 Vendor 补充。

---

## 29. 正式 500K 生产

不一次跑满。

建议：

```text
Batch 1   50K Accepted
Batch 2  100K
Batch 3  100K
Batch 4  100K
Batch 5  100K
Batch 6  补足 >=500K
```

每批完成必须执行：

- Gold Regression；
- Schema Gate；
- Random Audit；
- Duplicate Analysis；
- Subject Distribution；
- Source Distribution；
- Reject Distribution；
- Model/Prompt drift check；
- Cost Report。

PASS 后才能启动下一批。

---

## 30. Release 结构

```text
release/
├── questions.jsonl
├── image/
├── dataset_card.md
├── statistics.json
├── provenance_summary.json
└── quality_report.json
```

正式训练输入仍以：

```text
questions.jsonl + image/
```

为主。

---

## 31. 质量分层

Accepted 数据建议内部保留：

```text
GOLD
HIGH_CONFIDENCE
```

Rejected 独立保存。

### GOLD

- 真实大学课程/教材/考试；
- provenance 极强；
- solution 完整；
- Gold Regression 基线。

### HIGH_CONFIDENCE

- StackMathQA / StackExchange / Vendor 等；
- 经过所有 Gate；
- QA Alignment + Correctness PASS；
- provenance 完整。

训练时可以给 GOLD 更高 sampling weight，但不改变最终 19-field Contract。

---

## 32. 报告

每次 Run 输出至少：

```text
raw_count
normalized_count
accepted_count
rejected_count
acceptance_rate
reject_reason_counts
source_distribution
subject_distribution
course_distribution
problem_type_distribution
language_distribution
quality_tier_distribution
exact_duplicate_count
near_duplicate_count
model_calls
cache_hit_rate
fallback_count
provider_errors
tokens
estimated_cost
wall_time
```

Pilot 报告必须包含完整 Funnel。

---

## 33. 测试体系

### 33.1 Unit

- normalization；
- HTML → Markdown；
- MathML → LaTeX；
- MD5；
- 19-field validator；
- University Profile enums；
- answer extraction deterministic rules；
- reject codes；
- cache keys；
- provenance serialization。

### 33.2 Source Adapter Contract Tests

每个 Adapter 验证：

- stable id；
- raw content fidelity；
- resume/checkpoint；
- provenance；
- failure behavior；
- no hidden normalization。

### 33.3 Golden Tests

固定 Gold Source → NormalizedQA / UniversityQuestionIR 结果必须版本可追踪。

### 33.4 Provider Contract Tests

Classifier / Verifier / Embedding Provider 行为统一。

### 33.5 End-to-End

Source sample → acquisition → normalization → gates → dedup → 19-field JSONL。

### 33.6 Gold Regression

任何以下变化都必须跑 Gold：

- Prompt；
- Model；
- Provider；
- Threshold；
- Answer Extractor；
- Normalizer；
- Dedup；
- University Profile Validator。

---

## 34. Prompt 管理

全部 Prompt 文件化、版本化：

```text
prompts/
├── stem_classify/v1.txt
├── university_level/v1.txt
├── problem_classify/v1.txt
├── analysis_quality/v1.txt
├── qa_alignment/v1.txt
├── correctness_verify/v1.txt
├── duplicate_verify/v1.txt
└── metadata/v1.txt
```

输出强制结构化。

不得允许 Prompt 返回可直接覆盖题目/答案/解析的字段。

---

## 35. 安全与内容完整性

- API Key 不写配置仓库、日志或 static_info；
- Raw snapshots 只读；
- Normalized 内容与 Raw 内容分层；
- 任何正式内容改动必须可解释、可追溯；
- LLM 输出不能作为正式 answer / analysis payload；
- Provider response 原始结果可存内部 audit，不导出到训练数据；
- Sandbox 执行程序题时禁止网络访问，限制 CPU / memory / timeout。

---

## 36. 版权与许可策略

当前阶段优先获取规模，但必须做到“以后可以筛”。

因此：

1. 每条 Raw 从第一天保存 source / URL / author / timestamp / license metadata；
2. dataset-level license 与 original-source license 分开保存；
3. 许可未知不静默当作可商用；
4. 正式训练发布前可按 `license.status` 进行二次筛选；
5. Vendor 数据必须保存合同/授权批次级 identity；
6. StackExchange 等来源的许可版本需要后续按来源和时间进一步确认。

V1 不在代码中硬编码“某来源永久可训练”的法律结论。

---

## 37. 数据源当前验证基线

本设计时已核验以下公开信息：

### StackMathQA

- Source: Hugging Face `math-ai/StackMathQA`
- 1,138,426 unique questions across Math / MathOverflow / Statistics / Physics
- ~1,957,006 one-question-one-answer pairs
- URL: https://huggingface.co/datasets/math-ai/StackMathQA

### STEMQ

- 667 questions and solutions
- 27 STEM courses
- 7 universities
- URL: https://github.com/idrori/stemQ

### SciBench

- college-level scientific problems sourced from instructional textbooks
- URL: https://github.com/mandyyyyii/scibench

### Nexdata University STEM

- ~1.5M English STEM test questions
- university category
- fields include title / answer / parse / subject / grade / question type
- URL: https://www.nexdata.ai/datasets/llm/1881

### DataoceanAI University STEM

- dataset card describes 200K+ university-level math / physics / chemistry / computer science problems
- each with step-by-step solution and final answer
- Hugging Face repository currently contains metadata/card but no public data files
- URL: https://huggingface.co/datasets/DataoceanAI/University-level_Mathematics_Physics_Chemistry_Computer_Science_Reasoning_Corpus

这些规模均为 Source Planning Baseline，实际生产前需要再次做 source snapshot 与可下载性验证。

---

## 38. Design Decisions Summary

正式批准的核心决策：

1. 目标为 ≥500K Final Accepted，不是 500K Raw；
2. 只接受原始来源已有答案和解析的数据；
3. 禁止 AI 生成、补写、修正正式答案/解析；
4. 新建独立 University STEM Dataset Builder，不污染现有 DOCX Question Builder；
5. 最终继续使用现有 exact 19-field QuestionRecordV1；
6. 新增 UniversitySTEMProfileV1，不破坏 K12 Profile；
7. StackMathQA 为公开大规模候选池，不直接视为大学试题；
8. OpenStax / STEMQ / SciBench / CFE 等作为 Gold / 高质量来源；
9. Vendor 数据只在 Pilot 后根据缺口采购；
10. Pipeline 使用 precision-first + abstention + independent verification；
11. Answer / Analysis / University / Problem / Quality 分 Gate；
12. 去重不能只靠 MD5 或 Embedding 相似度；
13. provenance 与 license metadata 从第一天保存；
14. V1 使用 SQLite + DuckDB + Parquet + filesystem；
15. 先 5K Dry Run，再 50K Validation Pilot，再分批生产 500K；
16. 每批都必须 Gold Regression + Random Audit；
17. 图片题首版不追求比例，纯文本优先；
18. 所有 Prompt / Model / Threshold / Provider 都版本化；
19. 所有高成本调用必须缓存；
20. 公开数据够不够由 Pilot Acceptance Rate 决定，而不是预设结论。

---

## 39. 设计结论

University STEM Dataset Builder 的核心不是数据爬虫，而是：

```text
Large-scale Source Acquisition
+ Immutable Raw Provenance
+ University / Problem Classification
+ Original Answer Extraction
+ Original Analysis Validation
+ Independent QA Verification
+ Multi-level Dedup
+ Versioned Quality Evidence
+ 19-field University Profile Export
```

只要以上边界在实现阶段不被破坏，本项目就可以先用公开数据完成可验证的 5K/50K Pilot，再以真实留存率决定是否采购商业数据，最终以分批生产方式达到不少于 500,000 条高质量大学 STEM SFT 数据，而不需要在第一版引入重型基础设施。
