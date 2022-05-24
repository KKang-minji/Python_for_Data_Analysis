from dataclasses import replace
from nis import cat
import string
import pandas as pd
import numpy as np
from pyparsing import Regex

### 누락된 데이터 처리
string_data = pd.Series(["aardvark", "artichoke", np.nan, "avocado"])
string_data

string_data.isnull()

from numpy import nan as NA

data = pd.Series([1, NA, 3.5, NA, 7])
data.dropna()
data[data.notnull()]

data = pd.DataFrame([[1.0, 6.5, 3], [1.0, NA, NA], [NA, NA, NA], [NA, 6.5, 3.0]])
cleaned = data.dropna()
data
cleaned

# how = 'all 옵션을 넘기면 모두 NA인 로우만 제외
data.dropna(how="all")
data[4] = NA
data
data.dropna(axis=1, how="all")  # axis=1 : 컬럼을 제외시키는 방법

df = pd.DataFrame(np.random.randn(7, 3))
df.iloc[:4, 1] = NA
df.iloc[:2, 2] = NA
df
df.dropna()
df.dropna(thresh=2)

### 결측치 채우기
# fillna(메서드): 채워넣고 싶은 값 지정
df.fillna(0)
# 사전 값을 넘겨서 각 컬럼마다 다른 값 채우기
df.fillna({1: 0.5, 2: 0})
# fillna는 새로운 객체를 반환하지만 다음처럼 기존 객체 변경할 수 있음
_ = df.fillna(0, inplace=True)
df

df = pd.DataFrame(np.random.randn(6, 3))
df.iloc[2:, 1] = NA
df.iloc[4:, 2] = NA
df
df.fillna(method="ffill")
df.fillna(method="ffill", limit=2)

# 평균값이나 중간값 전달하기
data = pd.Series([1.0, NA, 3.5, NA, 7])
data.fillna(data.mean())


### 데이터 변형
# 중복 제거
data = pd.DataFrame({"k1": ["one", "two"] * 3 + ["two"], "k2": [1, 1, 2, 3, 3, 4, 4]})
data
# duplicated : 로우가 중복인지 아닌지 알려주는 불리언 series 반환
data.duplicated()

data.drop_duplicates()
data["v1"] = range(7)
data.drop_duplicates(["k1"])
# keep='last' 옵션을 넘기면 마지막으로 발견된 값을 반환한다.
data.drop_duplicates(["k1", "k2"], keep="last")

### 함수나 매핑을 이용해서 데이터 변형
data = pd.DataFrame(
    {
        "food": [
            "bacon",
            "pulled pork",
            "bacon",
            "Pastrami",
            "corned beef",
            "Bacon",
            "pastrami",
            "honey ham",
            "nova lox",
        ],
        "ounces": [4, 3, 12, 6, 7.5, 8, 3, 5, 6],
    }
)
data

meat_to_animal = {
    "bacon": "pig",
    "pulled pork": "pig",
    "pastrami": "cow",
    "corned beef": "cow",
    "honey ham": "pig",
    "nova lox": "salmon",
}

lowercased = data["food"].str.lower()
lowercased
data["animal"] = lowercased.map(meat_to_animal)
data

# map 메서드: 데이터 요소별 변환 및 데이터를 다듬는 작업 편리하게 수행가능
data["food"].map(lambda x: meat_to_animal[x.lower()])


### 값 치환
data = pd.Series([1.0, -999.0, 2.0, -999.0, -1000.0, 3.0])
data

# replce(메서드): pandas에서 인식할 수 있는 NA값으로 치환한 새로운 series 생성 가능
data.replace(-999, np.nan)
# 여러개의 값 한번에 치환하려면 치환하려는 값의 리스트를 넘기면 됨.
data.replace([-999, -1000], np.nan)
# 치환하려는 값마다 다른 값으로 치환하려면 누락된 값 대신 새로 지정할 값의 리스트를 사용
data.replace([-999, -1000], [np.nan, 0])
# 두개의 리스트 대신 사전을 이용하는 것도 가능
data.replace({-999: np.nan, -1000: 0})

### 축 색인 이름 바꾸기
data = pd.DataFrame(
    np.arange(12).reshape((3, 4)),
    index=["Ohio", "Colorado", "New York"],
    columns=["one", "two", "three", "four"],
)
data

transform = lambda x: x[:4].upper()
data.index.map(transform)
data.index = data.index.map(transform)
data
# rename: 객체 변경하지 않고 새로운 객체 생성
data.rename(index=str.title, columns=str.upper)
# 축 이름 중 일부만 변경가능
data.rename(index={"OHIO": "INDIANA"}, columns={"three": "peekaboo"})
# 원본 데이터를 바로 변경하려면 inplace=True 옵션 사용
data.rename(index={"OHIO": "INDIANA"}, inplace=True)
data

### 개별화와 양자화
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
# pd.cut: 동일 길이로 나누어서 범주 만들기
cats = pd.cut(ages, bins)
cats

cats.codes
cats.categories
pd.value_counts(cats)  # = pandas.cut 결과에 대한 그룹 수 이다.

pd.cut(ages, [18, 26, 36, 61, 100], right=False)

group_names = ["Youth", "YoungAdult", "MiddleAged", "Senior"]
pd.cut(ages, bins, labels=group_names)  # labels 옵션으로 그룹의 이름을 직접 넘겨줄 수 있음

data = np.random.rand(20)
pd.cut(data, 4, precision=2)  # precision=2: 소수점 아래 2자리까지로 제한
# cut: 데이터의 분산에 따라 각각의 그룹마다 데이터 수가 다르게 나뉘는 경우가 많음
# qcut: 표본 변위치를 기반으로 데이터를 나눠줌
# 표본 변위치를 사용하기에 적당히 같은 크기의 그룹으로 나눌 수 있음

data = np.random.randn(1000)  # 정규분포
cats = pd.qcut(data, 4)
cats

pd.value_counts(cats)
pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.0])

### 특잇값을 찾고 제외
# 한 컬럼에서 절댓값이 3을 초과하는 값 찾기
data = pd.DataFrame(np.random.randn(1000, 4))
data.describe()
col = data[2]
col[np.abs(col) > 3]
# 절댓값이 3을 초과하는 값이 들어있는 모든 로우를 선택하려면 불리언 any 메서드 사용
data[(np.abs(data) > 3).any(1)]
data[np.abs(data) > 3] = np.sign(data) * 3
data.describe()
# np.sign(data)는 data값이 양수인지 음수인지에 따라 1이나 -1이 담긴 배열 반환
np.sign(data).head()

### 치환과 임의 샘플링
df = pd.DataFrame(np.arange(5 * 4).reshape((5, 4)))
# permutation: 순서 바꾸고 싶은 만큼의 길이를 넘겨서 바뀐 순서가 담긴 정수 배열 생성
sampler = np.random.permutation(5)
sampler
# 이 배열은 iloc 기반의 색인이나 take 함수에서 사용 가능
df
df.take(sampler)
# sample: 치환없이 일부만 임의로 선택
df.sample(n=3)
# sample에 replace=true : (반복 선택을 허용하며) 표본 치환을 통해 생성
choices = pd.Series([5, 7, -1, 6, 4])
draws = choices.sample(n=10, replace=True)
draws

### 표시자/더미 변수 계산하기
df = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "b"], "data1": range(6)})
pd.get_dummies(df["key"])

dummies = pd.get_dummies(df["key"], prefix="key")
df_with_dummy = df[["data1"]].join(dummies)
df_with_dummy

### 문자열 다루기
# split을 이용해서 문자열 분리
val = "a,b, guido"
val.split(",")
pieces = [x.strip() for x in val.split(",")]
pieces
# 더하기 연산을 사용해서 문자열 더하기
first, second, third = pieces
first + "::" + second + "::" + third
"::".join(pieces)

"guido" in val
val.index(",")
val.find(":")

val.index(":")
val.count(",")
val.replace(",", "::")
val.replace(",", "")

### 정규표현식
import re

text = "foo bar\t baz \tqux"
re.split("\s+", text)

regex = re.compile("\s+")
regex.split(text)

regex.findall(text)

text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
pattern = r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}"
regex = re.compile(pattern, flags=re.IGNORECASE)

regex.findall(text)
m = regex.search(text)
m
text[m.start() : m.end()]
print(regex.match(text))

print(regex.sub("REDACTED", text))
# 컴포넌트로 나눠야 한다면 각 패턴을 괄호로 묶어준다
pattern = r"([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})"
regex = re.compile(pattern, flags=re.IGNORECASE)
m = regex.match("wesm@bright.net")
# match를 이용하면 group 메서드로 각 패턴 컴포넌트 튜플얻을 수 있음
m.groups()

regex.findall(text)

print(regex.sub(r"Username:\1, Domain:\2, Suffix:\3", text))

### pandas의 벡터화된 문자열 함수
data = {
    "Dave": "dave@google.com",
    "Steve": "steve@gmail.com",
    "Rob": "rob@gmail.com",
    "Wes": np.nan,
}
data = pd.Series(data)
data
data.isnull()

data.str.contains("gmail")
pattern
data.str.findall(pattern, flags=re.IGNORECASE)
matches = data.str.match(pattern, flags=re.IGNORECASE)
matches
# str.get, str을 사용하여 벡터화된 요소 꺼내오기
# 문자형일 때만 가능 (숫자나 불리언 값은 안됨)
matches.str.get(1)
matches.str[0]
# 문자열 자르기
data.str[:5]
