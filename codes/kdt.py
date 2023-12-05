import numpy as np
from enum import Enum
import math
import os.path as osp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
LIB_DIR = osp.dirname(osp.abspath(__file__))
WORKING_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
DATA_DIR = osp.join(WORKING_DIR, "data")

M = 12
D = 7
MOD = 1000000007 
COL_NUM = [1, 1, 1, 1, 1, 1, 1, 26, 363, 53, 366, 53]

class OP(Enum):
    COUNT = 0
    SUM = 1
    AVG = 2

INT_T = int
OP_T = OP
FLOAT_T = float

# KD-Tree 节点
class Node:
    def __init__(self):
        self.lc = None
        self.rc = None
        self.cnt = 0
        self.sum = [0] * D
        self.dim = [[1e9, -1e9] for _ in range(D)]

# AQP 询问参数
class Result:
    def __init__(self, op: OP_T, col: INT_T):
        self.op = op
        self.col = col

class Predicate:
    def __init__(self, col: INT_T, lb: FLOAT_T, ub: FLOAT_T):
        self.col = col
        self.lb = lb
        self.ub = ub

# AQP 询问答案
class GroupAnswer:
    def __init__(self, id=0, value=0.0):
        self.id = id
        self.value = value

class Answer:
    def __init__(self, group_ans=None, size=0):
        self.group_ans = group_ans if group_ans is not None else []
        self.size = size

rootMap = {} 
disCol = []
rootName = []

pdata = None
data = None
maxDepth = None

n = 0
sum = [0.0] * D
cnt = 0.0

def isLeaf(u: Node):
    return u.lc is None and u.rc is None

def isContinuous(c):
    return c <= 6

def isDiscrete(c):
    return c >= 7

def getId(i, j):
    return i * M + j

# 将 _data 导入为一维 pdata
def loadData(_data, _n):
    global n, pdata, data
    n = _n
    pdata = _data.reshape(-1)[:n * M]

# 更新 KD-Tree 节点区间信息
def updateNode(u, v):
    if v is None:
        return
    
    for i in range(len(u.sum)):
        u.sum[i] += v.sum[i]
        u.dim[i][0] = min(u.dim[i][0], v.dim[i][0])
        u.dim[i][1] = max(u.dim[i][1], v.dim[i][1])

# 取 data[l: r+1] 建立 KD-Tree
def buildKDTree(data, l, r, depth):
    if l > r:
        return None
    
    u = Node() 
    if l == r or depth >= maxDepth:
        u.lc = None
        u.rc = None
        u.cnt = r - l + 1
        u.sum = [0] * D
        u.dim = [[1e9, -1e9] for _ in range(D)]
        for i in range(D):
            u.sum[i] += np.sum(data[l:r + 1, i])
            u.dim[i][0] = min(u.dim[i][0], np.min(data[l:r + 1, i]))
            u.dim[i][1] = max(u.dim[i][1], np.max(data[l:r + 1, i]))

            # 优化前的实现：
            # for j in range(l, r + 1):
            #     u.sum[i] += data[j][i]
            #     u.dim[i][0] = min(u.dim[i][0], data[j][i])
            #     u.dim[i][1] = max(u.dim[i][1], data[j][i])

        return u

    k = depth % D
    mid = (l + r) >> 1

    arg = np.argsort(data[l:r+1, k]) # 代替 nth_element 的较优实现
    data[l:r+1] = data[l:r+1][arg]

    # 优化前的实现：
    # data[l:r + 1] = sorted(data[l:r + 1], key=lambda x: x[k])
    
    u.cnt = r - l + 1
    u.lc = buildKDTree(data, l, mid, depth + 1)
    u.rc = buildKDTree(data, mid + 1, r, depth + 1)
    u.sum = [0] * D
    u.dim = [[1e9, -1e9] for _ in range(D)]
    updateNode(u, u.lc)
    updateNode(u, u.rc)

    return u

# 对 columns 的组合建树
def buildCol(col, size, delta_depth):
    global maxDepth, saveCount, rootName
    
    # 多关键字排序
    d = np.arange(n)
    if len(col) > 0:
        sort_indices = np.lexsort([pdata[getId(d, i)] for i in col])
        d = d[sort_indices]

    # 优化前的实现：
    # d = list(range(n))
    # d.sort(key=lambda x: [pdata[getId(x, col[i])] for i in range(size)])

    _data = np.zeros((n, D), dtype=np.float64)
    for i in range(n):
        _data[i] = pdata[d[i] * M:d[i] * M + D]
    
    name = getName(col)
    saveCount = 0
    with open(getPath(name), "w") as file:
        l = 0
        while l < n:
            id_val, base = 0, 1
            
            # 获取对应取值所建树的根节点 Hash 值
            j = 7
            for i in col:
                while j < i:
                    base *= 13131
                    base %= MOD
                    j = j + 1
                id_val += base * (pdata[getId(d[l], i)] + 1)
                id_val %= MOD

            r = l  # 得到相同取值的区间 [l, r]
            while r < n - 1:
                mismatch = False
                for i in col:
                    if pdata[getId(d[l], i)] != pdata[getId(d[r + 1], i)]:
                        mismatch = True
                        break
                if mismatch:
                    break
                r += 1
            
            if size == 0: # 降低第一棵树的深度以减少 online 时间
                delta_depth = delta_depth - 2
            maxDepth = max(1, int(math.log2(r - l + 1) + delta_depth))
            
            if size == 3: # 限制均为离散型时不需要建树
                maxDepth = 0

            root = buildKDTree(_data, l, r, 0) 
            id_val = int(id_val)
            rootMap[id_val] = root

            # 装载到本地
            saveRoot(root, id_val, file)
                
            l = r + 1

def getName(col) -> str :
    name = ""
    for i in col:
        name += str(i)
    return name

def getPath(name) -> str :
    return osp.join(DATA_DIR, "_" + name + ".txt")

def saveKDTree(u, file):
    if not file or not u:
        return

    # 手动实现 class Node 的序列化
    if u.lc is None:
        file.write(str(None))
    else:
        file.write("1")
    file.write(" ")
    if u.rc is None:
        file.write(str(None))
    else:
        file.write("1")
    file.write(" ")
    file.write(str(u.cnt) + " ")
    for x in u.sum:
        file.write(str(x) + " ")
    for x in u.dim:
        file.write(str(x[0]) + " " + str(x[1]) + " ")
    file.write("\n")

    # 优化前的实现：
    # pickle.dump(u, file)

    saveKDTree(u.lc, file)
    saveKDTree(u.rc, file)

def saveRoot(u, id, file):
    if not file or not u:
        return
    
    file.write(str(id) + "\n")
    saveKDTree(u, file)

def loadKDTree(_data, i):
    nodeData = _data[i].split()

    # 手动实现 class Node 的反序列化
    u = Node()
    u.cnt = int(nodeData[2])
    u.sum = [float(i) for i in nodeData[3: 10]]
    u.dim = [[float(nodeData[i]), float(nodeData[i+1])] for i in range(10, 24, 2)]

    # 优化前的实现：
    # pickle.load(u, file)

    if nodeData[0] == '1':
        u.lc, i = loadKDTree(_data, i + 1)
    if nodeData[1] == '1':
        u.rc, i = loadKDTree(_data, i + 1)
    return u, i

# 读取 name 的 KD-Tree 森林
def loadname(name):
    with open(osp.join(getPath(name)), "r") as file:
        _data = [line.strip() for line in file.readlines()]
    i = 0
    while i < len(_data):
        idx = int(_data[i])
        rootMap[int(idx)], i = loadKDTree(_data, i + 1)
        i = i + 1
    return 

def loadTreeData():
    global rootName
    with open(osp.join(DATA_DIR, "list.txt")) as file:
        rootName = [name.strip() for name in file.readlines()]
    
    # IO 相关使用多线程加速，多进程无法保存
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for name in rootName:
            futures.append(executor.submit(loadname, name))
        
        for future in tqdm(futures):
            future.result()
        
# 处理得到 col 和 disCol
def extract(pre, psize, bound, col):
    global disCol
    disCol.clear()
    for i in range(psize):
        if isContinuous(pre[i].col):
            disCol.append(pre[i].col)
            if pre[i].col > 0:
                bound[pre[i].col][0] = pre[i].lb
                bound[pre[i].col][1] = pre[i].ub
            else: # YEAR_DATE 只会取整数值
                bound[pre[i].col][0] = math.ceil(pre[i].lb)
                bound[pre[i].col][1] = math.floor(pre[i].ub)
        else:
            col.append((pre[i].col, int(pre[i].lb)))

# 根据 col 求 Hash 值，得到根节点指针
def getRoot(col):
    col.sort()
    id_val, base = 0, 1
    i = 7
    for c, val in col:
        while i < c:
            base *= 13131
            base %= MOD
            i = i + 1
        id_val += base * (val + 1)
        id_val %= MOD
        
    if id_val not in rootMap:
        return None
    return rootMap[id_val]

# 近似询问区间的相交率
def crossRatio(dim, bound):
    ratio = 1.0
    for i in disCol:
        if dim[i][0] == dim[i][1]:
            ratio *= bound[i][0] <= dim[i][0] <= bound[i][1]
        else:
            l = max(dim[i][0], bound[i][0])
            r = min(dim[i][1], bound[i][1])
            t = 1 if i == 0 else 0 # YEAR_DATE
            ratio *= (r - l + t) / (dim[i][1] - dim[i][0] + t)
    return ratio

def isContain(dim, bound):
    for i in disCol:
        if dim[i][0] < bound[i][0] or dim[i][1] > bound[i][1]:
            return False
    return True

def isCross(dim, bound):
    for i in disCol:
        if dim[i][0] > bound[i][1] or dim[i][1] < bound[i][0]:
            return False
    return True

# KD-Tree 上递归查询
def queryKDTree(u, bound):
    global sum, cnt
    if u is None:
        return
    if isLeaf(u) or isContain(u.dim, bound):
        ratio = crossRatio(u.dim, bound)
        cnt += u.cnt * ratio
        for i in range(D):
            sum[i] += u.sum[i] * ratio
        return 

    if u.lc and isCross(u.lc.dim, bound):
        queryKDTree(u.lc, bound)
    if u.rc and isCross(u.rc.dim, bound):
        queryKDTree(u.rc, bound)

def queryRange(root, bound):
    global sum, cnt
    sum.clear()
    sum.extend([0.0] * D)
    cnt = 0.0
    queryKDTree(root, bound)

# AQP query
def query(res, rsize, pre, psize, groupby):
    global sum, cnt
    bound = [[0.0, 0.0] for _ in range(D)]
    col = []
    ans = Answer()

    extract(pre, psize, bound, col)

    if groupby != -1: # 存在 groupby，加入限制内
        index = -1
        in_pre = any(p[0] == groupby for p in col)
        if not in_pre:
            ans.size = COL_NUM[groupby] * rsize
            col.append((groupby, -1))
        else:
            ans.size = rsize

        ans.group_ans = [GroupAnswer() for _ in range(ans.size)]

        col.sort()
        index = next((i for i, p in enumerate(col) if p[0] == groupby), -1)
        bg, ed = 0, COL_NUM[groupby]
        if in_pre:
            bg = col[index][1]
            ed = bg + 1

        # 处理 groupby
        for i in range(bg, ed):
            col[index] = (groupby, i)
            root = getRoot(col)
            queryRange(root, bound)
            _sum, _cnt = sum, cnt

            for j in range(rsize):
                idx = j if in_pre else i * rsize + j
                ans.group_ans[idx].id = i
                if res[j].op == 0:
                    ans.group_ans[idx].value = _cnt
                elif res[j].op == 1:
                    ans.group_ans[idx].value = _sum[res[j].col]
                elif res[j].op == 2:
                    ans.group_ans[idx].value = _sum[res[j].col] / cnt if cnt else 1
    else:
        ans.size = rsize
        ans.group_ans = [GroupAnswer() for _ in range(ans.size)]
        root = getRoot(col)
        queryRange(root, bound)
        _sum, _cnt = sum, cnt

        for j in range(rsize):
            ans.group_ans[j].id = -1
            if res[j].op == 0:
                ans.group_ans[j].value = _cnt
            elif res[j].op == 1:
                ans.group_ans[j].value = _sum[res[j].col]
            elif res[j].op == 2:
                ans.group_ans[j].value = _sum[res[j].col] / cnt if cnt else 1
    return ans
