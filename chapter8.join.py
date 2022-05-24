from operator import le
from re import S
import re
from unittest import result
import pandas as pd
import numpy as np

### 계층적 색인
data = pd.Series(
    np.random.randn(9),
    index=[["a", "a", "a", "b", "b", "c", "c", "d", "d"], [1, 2, 3, 1, 3, 1, 2, 2, 3]],
)
data

data.index
# multiindex를 색인으로 하는 series, 바로 위 단계의 색인을 이용해서 하위 계층을 직접 접근가능

# 부분적 색인으로 접근
data["b"]
data["b":"c"]
data.loc[["b", "d"]]
data.loc[:, 2]  # 하위 계층의 객체 선택
# unstack: 데이터 새롭게 배열
data.unstack()
# stack: unstack 반대 작업
data.unstack().stack()
data

frame = pd.DataFrame(
    np.arange(12).reshape((4, 3)),
    index=[["a", "a", "b", "b"], [1, 2, 1, 2]],
    columns=[["Ohio", "Ohio", "Colorado"], ["Green", "Red", "Green"]],
)
frame

frame.index.names = ["key1", "key2"]
frame.columns.names = ["state", "color"]
frame
frame["Ohio"]
pd.MultiIndex.from_arrays(
    [["Ohio", "Ohio", "Colorado"], ["Green", "Red", "Green"]], names=["state", "color"]
)

### 계층의 순서를 바꾸고 정렬하기
# swaplevel: 두개의 계층번호나 이름이 뒤바뀐 새로운 객체를 반환
frame.swaplevel("key1", "key2")
frame.sort_index(level=1)
frame.swaplevel(0, 1).sort_index(level=0)

### 계층별 요약 통계
frame.sum(level="key2")
frame.sum(level="color", axis=1)

### dataframe의 컬럼 사용하기
frame = pd.DataFrame(
    {
        "a": range(7),
        "b": range(7, 0, -1),
        "c": ["one", "one", "one", "two", "two", "two", "two"],
        "d": [0, 1, 2, 0, 1, 2, 3],
    }
)
frame

frame2 = frame.set_index(["c", "d"])
frame2
frame.set_index(["c", "d"], drop=False)
# reset_index함수는 set_index와 반대되는 개념인데 계층적 색인단걔가 컬럼으로 이동
frame.reset_index()


### 데이터 합치기 ------------------------------------------------------------
### 데이터베이스 스타일로 dataframe 합치기
df1 = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "a", "b"], "data1": range(7)})
df2 = pd.DataFrame({"key": ["a", "b", "d"], "data2": range(3)})

df1
df2
# merge: 처음엔 교집합으로 이루어짐
pd.merge(df1, df2)
pd.merge(df1, df2, on="key")
df3 = pd.DataFrame({"lkey": ["b", "b", "a", "c", "a", "a", "b"], "data1": range(7)})
df3
df4 = pd.DataFrame({"rkey": ["a", "b", "d"], "data2": range(3)})
df4
pd.merge(df3, df4, left_on="lkey", right_on="rkey")
pd.merge(df1, df2, how="outer")

df1 = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "b"], "data1": range(6)})
df2 = pd.DataFrame({"key": ["a", "b", "a", "b", "d"], "data2": range(5)})

df1
df2

pd.merge(df1, df2, on="key", how="left")
pd.merge(df1, df2, how="inner")
pd.merge(df1, df2)

left = pd.DataFrame(
    {"key1": ["foo", "foo", "bar"], "key2": ["one", "two", "one"], "lval": [1, 2, 3]}
)
right = pd.DataFrame(
    {
        "key1": ["foo", "foo", "bar", "bar"],
        "key2": ["one", "one", "one", "two"],
        "rval": [4, 5, 6, 7],
    }
)
pd.merge(left, right, on=["key1", "key2"], how="outer")

pd.merge(left, right, on="key1")
# suffixes 인자로 겹치는 컬럼 이름 뒤에 붙일 문자열 지정 가능
pd.merge(left, right, on="key1", suffixes=("_left", "_right"))

### 색인 병합하기
left1 = pd.DataFrame({"key": ["a", "b", "a", "a", "b", "c"], "value": range(6)})
right1 = pd.DataFrame({"group_val": [3.5, 7]}, index=["a", "b"])
left1
right1
# left_index=True, right_index=True를 지정해서 해당 색인을 병합키로 사용
pd.merge(left1, right1, left_on="key", right_index=True)
pd.merge(left1, right1, left_on="key", right_index=True, how="outer")
lefth = pd.DataFrame(
    {
        "key1": ["Ohio", "Ohio", "Ohio", "Nevada", "Nevada"],
        "key2": [2000, 2001, 2002, 2001, 2002],
        "data": np.arange(5.0),
    }
)
righth = pd.DataFrame(
    np.arange(12).reshape((6, 2)),
    index=[
        ["Nevada", "Nevada", "Ohio", "Ohio", "Ohio", "Ohio"],
        [2001, 2000, 2000, 2000, 2001, 2002],
    ],
    columns=["event1", "event2"],
)
lefth
righth

pd.merge(lefth, righth, left_on=["key1", "key2"], right_index=True)
pd.merge(lefth, righth, left_on=["key1", "key2"], right_index=True, how="outer")

left2 = pd.DataFrame(
    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    index=["a", "c", "e"],
    columns=["Ohio", "Nevada"],
)
right2 = pd.DataFrame(
    [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [13, 14]],
    index=["b", "c", "d", "e"],
    columns=["Missouri", "Alabama"],
)

left2
right2
pd.merge(left2, right2, how="outer", left_index=True, right_index=True)
left2.join(right2, how="outer")

left1.join(right1, on="key")
another = pd.DataFrame(
    [[7.0, 8.0], [9.0, 10.0], [11.0, 12], [16.0, 17.0]],
    index=["a", "c", "e", "f"],
    columns=["New York", "Oregon"],
)

another

left2.join([right2, another])
left2.join([right2, another])
left2.join([right2, another], how="outer")
left1.join(right1, on="key")
another = pd.DataFrame(
    [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [16.0, 17.0]],
    index=["a", "c", "e", "f"],
    columns=["New York", "Oregon"],
)
another
left2.join([right2, another])
left2.join([right2, another], how="outer")

### 축 따라 이어붙이기
arr = np.arange(12).reshape((3, 4))
arr
# concatenate: numpy에서 ndarray를 이어 붙이는 함수
np.concatenate([arr, arr], axis=1)

# 밑의 세개의 series객체는 색인이 겹치지 않음
s1 = pd.Series([0, 1], index=["a", "b"])
s2 = pd.Series([2, 3, 4], index=["c", "d", "e"])
s3 = pd.Series([5, 6], index=["f", "g"])

pd.concat([s1, s2, s3])
# concat 함수는 axis=0을 기본값으로 하여 series 객체 생성
# axis=1을 넘어가면 dataframe (axis=1은 컬럼 의미)
pd.concat([s1, s2, s3], axis=1)

s4 = pd.concat([s1, s3])
s4
pd.concat([s1, s4], axis=1)
pd.concat([s1, s4], axis=1, join_axes=[["a", "c", "b", "e"]])

result = pd.concat([s1, s1, s3], keys=["one", "two", "three"])
result
result.unstack()
# series를 axis=1로 병합할 경우 keys는 dataframe 컬럼 제목 됨
pd.concat([s1, s2, s3], axis=1, keys=["one", "two", "three"])

df1 = pd.DataFrame(
    np.arange(6).reshape(3, 2), index=["a", "b", "c"], columns=["one", "two"]
)
df2 = pd.DataFrame(
    5 + np.arange(4).reshape(2, 2), index=["a", "c"], columns=["three", "four"]
)
df1
df2
pd.concat([df1, df2], axis=1, keys=["level1", "level2"])
# 리스트 대신 객체의 사전으로 하면 사전의 키가 keys옵션으로 사용
pd.concat({"level1": df1, "level2": df2}, axis=1)

pd.concat([df1, df2], axis=1, keys=["level1", "level2"], names=["upper", "lower"])
# np.random.randn(함수): 표준정규분포로부터 샘플링된 난수를 반환
df1 = pd.DataFrame(np.random.randn(3, 4), columns=["a", "b", "c", "d"])
df2 = pd.DataFrame(np.random.randn(2, 3), columns=["b", "d", "a"])
df1
df2
# ignore_index=True: 기존 index를 무시하고 연결한다
# 특별하게 인덱스를 유지해야하는 경우가 아니라면 옵션 추가
pd.concat([df1, df2], ignore_index=True)

### 겹치는 데이터 합치기
a = pd.Series(
    [np.nan, 2.5, np.nan, 3.5, 4.5, np.nan], index=["f", "e", "d", "c", "b", "a"]
)
b = pd.Series(np.arange(len(a), dtype=np.float64), index=["f", "e", "d", "c", "b", "a"])
b[-1] = np.nan
a
b
np.where(pd.isnull(a), b, a)
# combine_first(): 두 객체를 포개서 한 객체에서 누락된 데이터를 다른 객체에 있는 값으로 채움
b[:-2].combine_first(a[2:])

df1 = pd.DataFrame(
    {
        "a": [1.0, np.nan, 5.0, np.nan],
        "b": [np.nan, 2.0, np.nan, 6.0],
        "c": range(2, 18, 4),
    }
)
df2 = pd.DataFrame(
    {"a": [5.0, 4.0, np.nan, 3.0, 7.0], "b": [np.nan, 3.0, 4.0, 6.0, 8.0]}
)
df1
df2
df1.combine_first(df2)

### 재형성과 피벗 ------------------------------------------------------------
### 계층적 색인으로 재형성하기
data = pd.DataFrame(
    np.arange(6).reshape(2, 3),
    index=pd.Index(["Ohio", "Colorado"], name="state"),
    columns=pd.Index(["one", "two", "three"], name="number"),
)
data
# stack: 컬럼이 로우로 피벗(또는 회전)
result = data.stack()
result
# unstack: 로우를 컬럼으로 피벗
result.unstack()
# 레벨 숫자나 이름을 전달해서 단계 지정
result.unstack(0)
result.unstack("state")

s1 = pd.Series([0, 1, 2, 3], index=["a", "b", "c", "d"])
s2 = pd.Series([4, 5, 6], index=["c", "d", "e"])
data2 = pd.concat([s1, s2], keys=["one", "two"])
data2
data2.unstack()
data2.unstack().stack()
data2.unstack().stack(dropna=False)

df = pd.DataFrame(
    {"left": result, "right": result + 5},
    columns=pd.Index(["left", "right"], name="side"),
)
df

df.unstack("state")
# 축 이름 지정 가능
df.unstack("state").stack("side")

### 긴 형식에서 넓은 형식으로 피벗하기
data = pd.read_csv("/Users/kangminji/Desktop/pandas, geo/데이터 분석/data/macrodata.csv")
data.head()
periods = pd.PeriodIndex(year=data.year, quarter=data.quarter, name="date")
columns = pd.Index(["realgdp", "infl", "unemp"], name="item")
data = data.reindex(columns=columns)
data.index = periods.to_timestamp("D", "end")
ldata = data.stack().reset_index().rename(columns={0: "value"})

ldata[:10]
pivoted = ldata.pivot("date", "item", "value")
pivoted

ldata["value2"] = np.random.randn(len(ldata))
ldata[:10]

pivoted = ldata.pivot("date", "item")
pivoted[:5]
pivoted["value"][:5]

unstacked = ldata.set_index(["date", "item"]).unstack("item")
unstacked[:7]

### 넓은 형식에서 긴 형식으로 피벗
df = pd.DataFrame(
    {"key": ["foo", "bar", "baz"], "A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}
)
df
melted = pd.melt(df, ["key"])
melted
# pivot: 원래 모양 되돌림
reshaped = melted.pivot("key", "variable", "value")
reshaped
# reset_index(): 인덱스 초기화
reshaped.reset_index()

pd.melt(df, id_vars=["key"], value_vars=["A", "B"])
pd.melt(df, value_vars=["A", "B", "C"])
pd.melt(df, value_vars=["key", "A", "B"])
