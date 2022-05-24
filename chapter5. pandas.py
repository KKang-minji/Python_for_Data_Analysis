from lib2to3.pgen2.pgen import DFAState
from this import d
from unittest import result
import pandas as pd
import numpy as np
from pandas import Series, DataFrame

# Series
obj = pd.Series([4, 7, -5, 3])
obj

obj.values
obj.index

obj2 = pd.Series([4, 7, -5, 3], index=["d", "b", "a", "c"])
obj2
obj2.index

obj2["a"]
obj2["d"] = 6
obj2[["c", "a", "d"]]

obj2[obj2 > 0]

obj2 * 2

np.exp(obj2)
# index 직접 지정하고 싶으면 원하는 순서대로 가능
sdata = {"Ohio": 35000, "Texas": 71000, "Oregon": 16000, "Utah": 5000}
obj3 = pd.Series(sdata)
obj3
states = ["California", "Ohio", "Oregon", "Texas"]
obj4 = pd.Series(sdata, index=states)
obj4

# isnull, notnull: NA 찾을 때 사용됨
pd.isnull(obj4)
pd.notnull(obj4)
obj4.isnull()  # 메서드로도 존재함

obj4.name = "population"
obj4.index.name = "state"
obj4

obj
obj.index = ["Bob", "Steve", "Jeff", "Rayan"]
obj

# 데이터프레임 만들기
data = {
    "state": ["Ohio", "Ohio", "Ohio", "Nevada", "Nevada", "Nevada"],
    "year": [2000, 2001, 2002, 2001, 2002, 2003],
    "pop": [1.5, 1.7, 3.6, 2.4, 2.9, 3.2],
}
frame = pd.DataFrame(data)
frame.head()

# 원하는 순서대로 column 지정하여 dataframe 생성
pd.DataFrame(data, columns=["year", "state", "pop"])

frame2 = pd.DataFrame(
    data,
    columns=["year", "state", "pop", "debt"],
    index=["one", "two", "three", "four", "five", "six"],
)
frame2  # 사전에 없는 값은 결측치

frame2["state"]
frame2.year
frame2.loc["three"]

# 배열 값 대입
frame2["debt"] = 16.5
frame2
frame2["debt"] = np.arange(6.0)
frame2

val = pd.Series([-1.2, -1.5, -1.7], index=["two", "four", "five"])
frame2["debt"] = val
frame2

frame2["eastern"] = frame2.state == "Ohio"
frame2
# 키워드 = 값 형식으로 키워드 인자 사용해 호출

# 컬럼 삭제
del frame2["eastern"]
frame2.columns

pop = {"Nevada": {2001: 2.4, 2002: 2.9}, "Ohio": {2000: 1.5, 2001: 1.7, 2002: 3.6}}

frame3 = pd.DataFrame(pop)
frame3
frame3.T

pd.DataFrame(pop, index=[2001, 2002, 2003])

frame3.index.name = "year"
frame3.columns.name = "state"
frame3
frame3.values
frame2.values

# 색인 객체 : 변경 불가능, 자료구조 사이에서 안전하게 공유될 수 있다.
obj = pd.Series(range(3), index=["a", "b", "c"])
index = obj.index
index
index[1:]  # 첫번째 인덱스 제외한 모든 것 추출

index[1] = "d"

labels = pd.Index(np.arange(3))
labels
obj2 = pd.Series([1.5, -2.5, 0], index=labels)
obj2
obj2.index is labels

frame3
frame3.columns

# 핵심기능
# 재색인
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=["d", "b", "a", "c"])
obj
# reindex: 재배열
obj2 = obj.reindex(["a", "b", "c", "d", "e"])
obj2
obj3 = pd.Series(["blue", "purple", "yellow"], index=[0, 2, 4])
obj3
# ffill: 누락된 값을 직전 값으로 채워넣을 수 있음
obj3.reindex(range(6), method="ffill")

df = pd.DataFrame(
    [[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]],
    index=["a", "b", "c", "d"],
    columns=["one", "two"],
)
df

df.sum()
df.sum(axis="columns")
df.mean(axis="columns", skipna=False)
df.idxmax()
df.cumsum()

### 상관관계와 공분산
import pandas_datareader.data as web

all_data = {
    ticker: web.get_data_yahoo(ticker) for ticker in ["AAPL", "IBM", "MSFT", "GOOG"]
}
price = pd.DataFrame({ticker: data["Adj Close"] for ticker, data in all_data.items()})
volume = pd.DataFrame({ticker: data["Volume"] for ticker, data in all_data.items()})

# 주식의 퍼센트 변화율
returns = price.pct_change()
returns.tail()
# corr: 상관관계
returns["MSFT"].corr(returns["IBM"])
# cov: 공분산
returns["MSFT"].cov(returns["IBM"])

returns.MSFT.corr(returns.IBM)

returns.corr()
returns.cov()


returns.corrwith(returns.IBM)

returns.corrwith(volume)

### 유일값, 값 세기, 멤버십
obj = pd.Series(["c", "a", "d", "a", "a", "b", "b", "c", "c"])

uniques = obj.unique()
uniques

# 내림차순으로 정렬
obj.value_counts()
pd.value_counts(obj.values, sort=False)

obj
# isin(메서드): 어떤 값이 series에 존재하는 지 나타내는 불리언 벡터를 반환
mask = obj.isin(["b", "c"])
mask
obj[mask]

to_mach = pd.Series(["c", "a", "b", "b", "c", "a"])
unique_vals = pd.Series(["c", "b", "a"])
pd.Index(unique_vals).get_indexer(to_mach)

data = pd.DataFrame(
    {"Qu1": [1, 3, 4, 3, 4], "Qu2": [2, 3, 1, 2, 3], "Qu3": [1, 5, 2, 4, 4]}
)
data
result = data.apply(pd.value_counts).fillna(0)
result
