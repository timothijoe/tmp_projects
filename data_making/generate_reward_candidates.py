#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


BLIND_TYPES = [
    "车辆遮挡盲区",
    "豁口盲区",
    "弯道盲区",
    "弯道遮挡盲区",
    "无灯路口盲区",
    "环岛盲区",
    "匝道盲区",
    "坡道盲区",
    "施工盲区",
]

OBJECT_TYPES = [
    "行人",
    "两轮骑行者",
    "三轮骑行者",
    "小汽车",
    "SUV",
    "货车",
    "公交车",
    "面包车",
    "出租车",
    "锥桶",
    "施工区域",
    "事故车",
    "工程车",
]

DIRECTION_REPLACEMENTS = [
    ("左前方", "右前方"),
    ("右前方", "左前方"),
    ("左侧", "右侧"),
    ("右侧", "左侧"),
    ("左后方", "右后方"),
    ("右后方", "左后方"),
    ("左方", "右方"),
    ("右方", "左方"),
]

SAME_CATEGORY_REPLACEMENTS = {
    "blind_spot": [
        ("车辆遮挡盲区", "豁口盲区"),
        ("豁口盲区", "车辆遮挡盲区"),
        ("弯道遮挡盲区", "无灯路口盲区"),
        ("弯道盲区", "无灯路口盲区"),
        ("无灯路口盲区", "弯道盲区"),
        ("环岛盲区", "匝道盲区"),
        ("匝道盲区", "环岛盲区"),
        ("坡道盲区", "豁口盲区"),
        ("施工盲区", "车辆遮挡盲区"),
    ],
    "obstacle_avoidance": [
        ("锥桶等施工区域", "事故车"),
        ("行人", "两轮骑行者"),
        ("两轮骑行者", "三轮骑行者"),
        ("三轮骑行者", "两轮骑行者"),
        ("小汽车", "货车"),
        ("SUV", "面包车"),
        ("货车", "小汽车"),
        ("公交车", "工程车"),
        ("面包车", "公交车"),
        ("锥桶", "事故车"),
        ("事故车", "施工区域"),
        ("工程车", "公交车"),
    ],
    "lane_change": [
        ("空闲", "拥堵"),
        ("正常通行", "通行效率较低"),
        ("通行效率较高", "通行效率较低"),
        ("无来车", "有快速来车"),
        ("安全距离", "距离较近"),
        ("速度较慢", "速度较快"),
        ("速度较快", "速度较慢"),
    ],
    "yielding": [
        ("小汽车", "货车"),
        ("SUV", "面包车"),
        ("货车", "小汽车"),
        ("公交车", "出租车"),
        ("两轮骑行者", "三轮骑行者"),
        ("行人", "车辆"),
        ("紧急切入", "正常行驶"),
        ("意图切入", "正常行驶"),
    ],
    "traffic_signal": [
        ("红色", "绿色"),
        ("红灯", "绿灯"),
        ("黄色", "绿色"),
        ("黄灯", "绿灯"),
        ("绿色", "红色"),
        ("绿灯", "红灯"),
        ("由红变绿", "保持红灯"),
        ("前车开始起步", "前车仍在等待"),
    ],
    "road_condition": [
        ("坑洼路面", "积水路面"),
        ("坑洼", "积水"),
        ("积水", "井盖凹陷"),
        ("井盖", "减速带"),
        ("减速带", "坑洼路面"),
        ("湿滑", "坑洼"),
    ],
    "curve": [
        ("大曲率弯道上坡", "减速带"),
        ("大曲率弯道", "坡道"),
        ("弯道上坡", "湿滑路面"),
    ],
}

CROSS_CATEGORY_REPLACEMENTS = [
    ("锥桶等施工区域", "车辆遮挡盲区"),
    ("车辆遮挡盲区", "施工区域占道"),
    ("豁口盲区", "车辆占道停车"),
    ("弯道遮挡盲区", "坑洼路面"),
    ("弯道盲区", "减速带"),
    ("无灯路口盲区", "锥桶占道"),
    ("环岛盲区", "事故车占道"),
    ("匝道盲区", "积水路面"),
    ("坡道盲区", "车辆切入自车道"),
    ("施工盲区", "红灯"),
    ("锥桶", "豁口盲区"),
    ("施工区域", "车辆遮挡盲区"),
    ("事故车", "环岛盲区"),
    ("坑洼路面", "车辆遮挡盲区"),
    ("坑洼", "车辆遮挡盲区"),
    ("积水", "锥桶占道"),
    ("井盖", "无灯路口盲区"),
    ("减速带", "车辆占道停车"),
]


def clean_sentence(text):
    text = re.sub(r"\s+", "", text.strip())
    if text and text[-1] not in "。！？；":
        text += "。"
    return text


def split_clauses(sentence):
    sentence = clean_sentence(sentence).rstrip("。！？；")
    parts = [p for p in re.split(r"[，。；]", sentence) if p]
    return parts


def extract_action(sentence):
    s = sentence
    if "紧急减速" in s:
        return "紧急减速让行"
    if "减速让行" in s or "让行" in s:
        return "减速让行"
    if "向左变更车道" in s or "向左换道" in s:
        return "向左变更车道"
    if "向右变更车道" in s or "向右换道" in s:
        return "向右变更车道"
    if "换道" in s or "变更车道" in s:
        return "换道"
    if "向左避障" in s or "左避让" in s or "向左避让" in s:
        return "向左避障"
    if "向右避障" in s or "右避让" in s or "向右避让" in s:
        return "向右避障"
    if "刹停" in s or "停车等待" in s:
        return "刹停在路口停止线前"
    if "跟随前车起步" in s or "跟车起步" in s:
        return "跟车起步"
    if "开始起步" in s or "起步通过" in s:
        return "起步通过"
    if "减速" in s:
        return "减速行驶通过"
    if any(k in s for k in ["占用自车道", "阻挡自车通行", "影响自车通行", "占据自车道", "占道"]):
        return "向左避障"
    return "谨慎通行"


def classify(sentence, action):
    if is_signal_sentence(sentence):
        return "traffic_signal", signal_subcategory(sentence)
    if "交警" in sentence:
        return "traffic_control", "police_direction"
    if "盲区" in sentence:
        count = sum(1 for t in BLIND_TYPES if t in sentence)
        return "blind_spot", "combined_blind_spots" if count >= 2 else "single_blind_spot"
    if action in ["向左变更车道", "向右变更车道", "换道"]:
        if any(k in sentence for k in ["等待", "速度较快", "通过后"]):
            return "lane_change", "must_wait"
        if any(k in sentence for k in ["锥桶", "施工", "事故", "阻挡", "占用", "未行驶"]):
            return "lane_change", "reactive_obstacle_blocking"
        return "lane_change", "proactive_slow_ahead"
    if any(k in sentence for k in ["占用自车道", "阻挡自车通行", "影响自车通行", "占据自车道", "占道"]):
        return "obstacle_avoidance", "static_obstacle_blocking_lane"
    if action in ["减速让行", "紧急减速让行"]:
        return "yielding", "emergency_evasion" if "紧急" in sentence or "突然" in sentence else "decelerate_to_yield"
    if any(k in sentence for k in ["坑洼", "积水", "井盖", "减速带", "湿滑"]):
        return "road_condition", "road_surface"
    if "大曲率弯道" in sentence or "弯道上坡" in sentence:
        return "curve", "curve_geometry"
    if action in ["向左避障", "向右避障"]:
        if "对向" in sentence or "让行后" in sentence:
            return "obstacle_avoidance", "complex_avoidance"
        if any(k in sentence for k in ["缓慢", "缓速", "通行", "行驶"]):
            return "obstacle_avoidance", "dynamic_obstacle"
        return "obstacle_avoidance", "static_obstacle_blocking_lane"
    if "前车" in sentence or "跟车" in sentence:
        return "following", "following_scenario"
    return "special", "unclassified"


def signal_subcategory(sentence):
    if any(k in sentence for k in ["红灯", "红色"]) and is_signal_sentence(sentence):
        return "red_light_stop"
    if any(k in sentence for k in ["绿灯", "绿色"]) and is_signal_sentence(sentence):
        return "green_light_go"
    if any(k in sentence for k in ["黄灯", "黄色"]) and is_signal_sentence(sentence):
        return "yellow_light"
    return "signal_combination"


def is_signal_sentence(sentence):
    return any(
        k in sentence
        for k in [
            "红灯",
            "绿灯",
            "黄灯",
            "箭头灯",
            "圆灯",
            "信号灯",
            "交通灯",
            "待转区",
            "停止线",
            "由红变绿",
        ]
    )


def action_clause(action):
    return f"自车应{action}。"


def description_clauses(sentence):
    parts = split_clauses(sentence)
    return [p for p in parts if "自车应" not in p and not p.startswith("自车应")]


def join_with_action(desc, action):
    desc = [d for d in desc if d]
    if not desc:
        return action_clause(action)
    return "，".join(desc) + "，" + action_clause(action)


def generalized_text(sentence, category, action):
    if category == "blind_spot":
        return f"前方存在盲区风险，自车应{action}。"
    if category == "lane_change":
        direction = "左侧" if "左" in action else "右侧" if "右" in action else "相邻"
        return f"前方通行受阻，{direction}车道具备换道条件，自车应{action}。"
    if category == "yielding":
        return f"前方有交通参与者影响自车通行，自车应{action}。"
    if category == "traffic_signal":
        if "刹停" in action:
            return f"当前路口信号不允许通行，自车应{action}。"
        return f"当前路口信号允许通行，自车应{action}。"
    if category == "road_condition":
        return f"前方存在异常路面，自车应{action}。"
    if category == "curve":
        return f"前方存在弯道风险，自车应{action}。"
    if category == "obstacle_avoidance":
        return f"前方存在障碍物影响通行，自车应{action}。"
    return f"前方存在通行风险，自车应{action}。"


def missing_factor_texts(sentence, action, limit=3):
    desc = description_clauses(sentence)
    candidates = []
    if len(desc) >= 2:
        candidates.append(join_with_action(desc[:-1], action))
        candidates.append(join_with_action(desc[1:], action))
    if len(desc) >= 3:
        candidates.append(join_with_action([desc[0], desc[-1]], action))
    elif desc:
        shorter = desc[0]
        shorter = re.sub(r"(白色|黑色|灰色|红色|黄色|绿色|蓝色|银色|棕色|浅棕色|银灰色|银红色)", "", shorter)
        shorter = re.sub(r"(车辆遮挡|豁口|弯道遮挡|弯道|无灯路口|环岛|匝道|坡道|施工)盲区", "盲区", shorter)
        shorter = shorter.replace("左前方", "前方").replace("右前方", "前方")
        candidates.append(join_with_action([shorter], action))
    return unique(candidates)[:limit]


def wrong_factor_records(sentence, action, category):
    records = []
    records.extend(
        make_replacement_records(
            sentence,
            action,
            DIRECTION_REPLACEMENTS,
            "direction_swap",
            2.8,
            limit=2,
        )
    )
    records.extend(
        make_replacement_records(
            sentence,
            action,
            SAME_CATEGORY_REPLACEMENTS.get(category, []),
            "same_category_subtype_swap",
            2.3,
            limit=3,
        )
    )
    records.extend(
        make_replacement_records(
            sentence,
            action,
            CROSS_CATEGORY_REPLACEMENTS,
            "cross_category_swap",
            1.7,
            limit=2,
        )
    )
    records.extend(cross_category_template_records(sentence, action, category))
    if not any(record["factor_error_subtype"] == "cross_category_swap" for record in records):
        records.append(
            {
                "text": add_extra_factor(sentence, action, "前方存在未提及的施工人员"),
                "factor_error_subtype": "cross_category_addition",
                "score": 1.7,
            }
        )
    return unique_records(records)


def make_replacement_records(sentence, action, replacements, subtype, score, limit):
    records = []
    for old, new in replacements:
        if "锥桶等施工区域" in sentence and old in ["锥桶", "施工区域"]:
            continue
        if old in sentence:
            records.append(
                {
                    "text": ensure_action(sentence.replace(old, new, 1), action),
                    "factor_error_subtype": subtype,
                    "score": score,
                }
            )
        if len(records) >= limit:
            break
    return records


def cross_category_template_records(sentence, action, category):
    if category == "blind_spot":
        text = f"前方存在锥桶占道，影响自车通行，自车应{action}。"
    elif category in ["obstacle_avoidance", "lane_change"]:
        text = f"前方存在车辆遮挡盲区，自车应{action}。"
    elif category == "yielding":
        text = f"前方存在坑洼路面，自车应{action}。"
    elif category == "traffic_signal":
        text = f"前方存在施工区域占道，自车应{action}。"
    elif category in ["road_condition", "curve"]:
        text = f"右前方有车辆意图切入自车道，自车应{action}。"
    else:
        text = f"前方存在无灯路口盲区，自车应{action}。"
    return [
        {
            "text": text,
            "factor_error_subtype": "cross_category_swap",
            "score": 1.7,
        }
    ]


def add_extra_factor(sentence, action, extra):
    sentence = clean_sentence(sentence).rstrip("。")
    if "自车应" in sentence:
        sentence = re.sub(r"，?自车应.*$", "", sentence)
    return f"{sentence}，且{extra}，自车应{action}。"


def ensure_action(sentence, action):
    sentence = clean_sentence(sentence)
    if "自车应" in sentence:
        return sentence
    return sentence.rstrip("。") + "，" + action_clause(action)


def unsafe_text(sentence, action):
    s = clean_sentence(sentence)
    if "刹停" in action or "停车" in action:
        return replace_action(s, "加速通过路口")
    if "减速" in action or "让行" in action:
        return replace_action(s, "加速通过")
    if "向左" in action:
        return replace_action(s, action.replace("向左", "向右", 1))
    if "向右" in action:
        return replace_action(s, action.replace("向右", "向左", 1))
    if "起步" in action:
        return replace_action(s, "继续停车等待")
    return replace_action(s, "加速通过")


def replace_action(sentence, new_action):
    sentence = clean_sentence(sentence).rstrip("。")
    if "自车应" in sentence:
        sentence = re.sub(r"自车应.*$", f"自车应{new_action}", sentence)
    else:
        sentence += f"，自车应{new_action}"
    return sentence + "。"


def unique(items):
    seen = set()
    out = []
    for item in items:
        item = clean_sentence(item)
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def unique_records(records):
    seen = set()
    out = []
    for record in records:
        text = clean_sentence(record["text"])
        if text not in seen:
            item = dict(record)
            item["text"] = text
            out.append(item)
            seen.add(text)
    return out


def build_records(source_id, source):
    source = clean_sentence(source)
    action = extract_action(source)
    category, subcategory = classify(source, action)
    complete_text = ensure_action(source, action)
    records = [
        {
            "source_id": source_id,
            "category": category,
            "subcategory": subcategory,
            "score": 5,
            "error_type": "complete",
            "source": source,
            "text": complete_text,
        },
        {
            "source_id": source_id,
            "category": category,
            "subcategory": subcategory,
            "score": 4,
            "error_type": "generalized",
            "source": source,
            "text": generalized_text(source, category, action),
        },
    ]
    for text in missing_factor_texts(source, action):
        records.append(
            {
                "source_id": source_id,
                "category": category,
                "subcategory": subcategory,
                "score": 3,
                "error_type": "missing_factor",
                "source": source,
                "text": text,
            }
        )
    for wrong_record in wrong_factor_records(source, action, category):
        records.append(
            {
                "source_id": source_id,
                "category": category,
                "subcategory": subcategory,
                "score": wrong_record["score"],
                "error_type": "wrong_factor",
                "factor_error_subtype": wrong_record["factor_error_subtype"],
                "source": source,
                "text": wrong_record["text"],
            }
        )
    records.append(
        {
            "source_id": source_id,
            "category": category,
            "subcategory": subcategory,
            "score": 1,
            "error_type": "unsafe_action",
            "source": source,
            "text": unsafe_text(source, action),
        }
    )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data_making/test_sentences.txt")
    parser.add_argument("--output", default="data_making/reward_candidates_sample.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    lines = [
        clean_sentence(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit > 0:
        lines = lines[: args.limit]

    records = []
    for idx, source in enumerate(lines, 1):
        records.extend(build_records(idx, source))

    out = Path(args.output)
    out.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(records)} records for {len(lines)} sources to {out}")


if __name__ == "__main__":
    main()
