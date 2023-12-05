import pandas as pd
import json
import os
import sys
from os import path as osp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
lib_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'codes')
sys.path.append(lib_dir)
import kdtlib as lib

def aqp_online(data: pd.DataFrame, Q: list) -> list:
    Q = pd.json_normalize([json.loads(i) for i in Q]) # 把 Q 转为 DataFrame
    Q = lib.numerize(Q) # 数值化 Q

    results = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for _, q in Q.iterrows():
            futures.append(executor.submit(lib.query, q))

        for future in tqdm(futures):
            results.append(future.result())
                
    results = [json.dumps(i, ensure_ascii=False) for i in results]

    return results

def aqp_offline(data: pd.DataFrame, Q: list) -> None:
    lib.loadData(data) # 读入数据
    lib.buildKDTrees(-3) # 预处理建树
    lib.loadTreeData() # 读入 KDTree 数据
