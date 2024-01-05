import numpy as np
import pandas as pd
import os
import os.path as osp
import sys
from itertools import combinations, product
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy.special import comb
import shutil

LIB_DIR = osp.dirname(osp.abspath(__file__))
WORKING_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
DATA_DIR = osp.join(WORKING_DIR, "data")
if not osp.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

sys.path.append(LIB_DIR)
import kdt

COLUMNS = [
    "YEAR_DATE",
    "DEP_DELAY",
    "TAXI_OUT",
    "TAXI_IN",
    "ARR_DELAY",
    "AIR_TIME",
    "DISTANCE",
    "UNIQUE_CARRIER",
    "ORIGIN",
    "ORIGIN_STATE_ABR",
    "DEST",
    "DEST_STATE_ABR",
]
DISCRETE_COLUMNS = [
    "UNIQUE_CARRIER",
    "ORIGIN",
    "ORIGIN_STATE_ABR",
    "DEST",
    "DEST_STATE_ABR",
]
CONTINUOUS_COLUMNS = [
    "YEAR_DATE",
    "DEP_DELAY",
    "TAXI_OUT",
    "TAXI_IN",
    "ARR_DELAY",
    "AIR_TIME",
    "DISTANCE",
]
OP_MAP = {"count": 0, "sum": 1, "avg": 2}

COLUMN2INDEX = {c: i for i, c in enumerate(COLUMNS)}
COLUMN2INDEX.update({"_None_": -1})
COLUMN2INDEX.update({"*": -1})

DATA, ID2VALUE, VALUE2ID = None, None, None

modelNames = []

# 对 dataFrame 实现映射
def factorize(df, columns=DISCRETE_COLUMNS):
    id2value, value2id = {}, {}
    df = df.copy()
    for col in columns:
        value, index = pd.factorize(df[col])
        df[col] = value
        col_id = COLUMN2INDEX[col]
        id2value.update({col_id: index.values})
        value2id.update({col_id: {v: i for i, v in enumerate(index)}})
        id2value.update({col: index.values})
        value2id.update({col: {v: i for i, v in enumerate(index)}})

    return df, id2value, value2id

# 初始化 valueMap
def mapInit(data=None):
    global DATA, ID2VALUE, VALUE2ID
    DATA, ID2VALUE, VALUE2ID = factorize(data)
    maps = [VALUE2ID, ID2VALUE]

# 根据 col 读取模型名称
def getDataName(col) -> str:
    name = ""
    for c in col:
        name += str(c)
    return name

# 将询问数值化
def numerize(Q: pd.DataFrame):
    # result_col
    _result = []
    for col in Q["result_col"]:
        _result.append(
            [(OP_MAP[i[0]], COLUMN2INDEX[i[1]]) for i in col if len(i) >= 2]
        )
    Q["result_col"] = _result

    # predicate
    _predicate = []
    for predicate in Q["predicate"]:
        cur_predicate = []
        for i in predicate:
            col = COLUMN2INDEX[i["col"]]
            lb, ub = 0, 1e9
            if i["col"] in DISCRETE_COLUMNS:
                lb = ub = VALUE2ID[i["col"]][i["lb"]]
            else:
                if i["lb"] != "_None_":
                    lb = float(i["lb"])
                if i["ub"] != "_None_":
                    ub = float(i["ub"])
            cur_predicate.append((col, lb, ub))
        _predicate.append(cur_predicate)
    Q["predicate"] = _predicate

    # groupby
    _groupby = Q["groupby"].apply(COLUMN2INDEX.get)
    Q["groupby"] = _groupby
    
    return Q

# 从 data 中读取数据
def loadData(data):
    data = data[COLUMNS]
    mapInit(data)
    data = DATA.copy()
    values = data.values.astype(np.float32)
    kdt.loadData(values, data.shape[0])

def buildKDTreeProcessed(col, deltaDepth):
    _col = np.array(col, dtype=np.int32)
    kdt.buildCol(_col, len(_col), deltaDepth)

def buildKDTrees(deltaDepth=0):
    global modelNames
    if osp.exists(osp.join(DATA_DIR, "list.txt")):
        return

    # 多进程数据本地装载
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for num in [0, 1, 2, 3]:
            for col in combinations(range(7, 12), num):
                futures.append(executor.submit(
                    buildKDTreeProcessed, col, deltaDepth
                ))
                modelNames.append(getDataName(col))

        for future in tqdm(futures):
            future.result()

    with open(osp.join(DATA_DIR, "list.txt"), "w") as file:
        for name in modelNames:
            file.write(name + "\n")

def query(Q):
    res = np.array(
        [kdt.Result(op, col) for op, col in Q["result_col"]]
    )
    pre = np.array(
        [kdt.Predicate(col, lb, ub) for col, lb, ub in Q["predicate"]]
        )
    
    groupby = Q['groupby']
    ans = kdt.query(res, len(res), pre, len(pre), groupby)
    size = ans.size
    ret = []
    # 处理 groupby
    for i in range(size):
        g_ans = ans.group_ans[i]
        if g_ans.id < 0:
            ret.append([g_ans.value])
        else:
            id = g_ans.id
            ret.append([ID2VALUE[groupby][id], g_ans.value])
    return ret

isLoaded = 0

def loadTreeData():
    global isLoaded
    if isLoaded == 0:
        kdt.loadTreeData()
        isLoaded = 1
