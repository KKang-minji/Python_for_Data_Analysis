template = '{0:.2f} {1:s} are worth US${2:d}'

s = '3.14159'
fval =float(s)
type (fval)
int (fval)

bool(fval)
bool(0)
# 비어있으면 false, 채워져있으면 true

a = None 
a is None

b = 5
b is not None

def add_and_maybe_multiply(a,b, c=None):
    result = a + b
    if c is not None:
        result = result * c
    return result

##### 날짜와 시간 #################################
from datetime import datetime, date, time
from email.policy import default
from pickle import EMPTY_DICT
from unittest.loader import VALID_MODULE_NAME

from numpy import empty
dt = datetime(2011, 10, 29, 20, 30, 21)
dt.day
dt.minute

dt.date()

dt.time()
#strftime는 문자열로 만들어줌
dt.strftime('%m/%d/%Y %H:%M')

datetime.strptime('20091031', '%Y%m%d')

dt.replace(minute=0, second=0)

dt2 = datetime(2011,11, 15,22,30)
delta = dt2 - dt
delta
type(delta)

dt
dt + delta

##### if, ifelse, else ###########################

if x < 0:
    print("It's negative")

if x < 0:
    print("It's negative")
elif x == 0:
    print('Equal to zero')
elif 0 < x < 5:
    print('Positive but smaller than 5')
else:
    print('Positive and larger than or equal to 5')

a = 5; b = 7
c = 8; d = 4
if a < b or c > d:
    print('Made it')
    
4 > 3 > 2 > 1

##### for문 ##########################################
# for 변수 in 리스트(또는 터플, 문자열, 딕셔너리):
   # <수행할 문장>
   # <수행할 문장>
   
for value in collection:

sequence = [1,2, None, 4, None, 5]
total = 0
for value in sequence:
    if value is None:
        continue
    total += value
    
sequence = [1, 2, 0, 4, 6, 5, 2, 1]
total_until_5 = 0
for value in sequence: #sequence의 각항목 value에 대하여
    if value == 5: #만약 value 가 5면
        break  #for문에서 나와라.
    total_until_5 += value #5가 아니면 실행   

for i in range(4):
    for j in range(4):
        if j > i:
            break
        print((i,j))
   
for a, b, c iterator:


##### while문 #######################################

x = 256
total = 0
while x > 0:
    if total > 500:
        break
    total += x
    x = x // 2
    print(x)


if x < 0:
    print('negative!')
elif x == 0:
    #TODO: 여기에 내용 채울 것
    pass
else:
    print('positive!')
    
range(10)    
list(range(10))    
    
list(range(0,20,2))    
    
list(range(5,0,-1))

seq = [1,2,3,4]
for i in range(len(seq)):
    val = seq[i]

sum = 0
for i in range(100000): # 범위: 0 ~ 99,999
    if i % 3 == 0 or i % 5 == 0: # 3의 배수, 5의 배수
        sum += i  #모두 더하기


x = 5
'Non-ndgative' if x >= 0 else 'Negative'


##### 튜플 ########################################

tup = tuple(['foo', [1,2], True])
tup[2] = False
tup[1].append(3)
tup

(4, None, 'foo') + (6,0) + ('bar',) 

('foo','bar') * 4

tup = (4,5,6)
a,b,c = tup
b

tup = 4,5,(6,7)
a, b, (c,d) = tup
d

# 리스트 순회 방법
seq = [(1,2,3),(4,5,6),(7,8,9)]
for a, b, c in seq:
    print('a={0}, b={1}, c={2}'.format(a,b,c))

values = 1,2,3,4,5
a, b, *rest = values
a,b
rest

a, b, *_ = values

# 튜플에서 많이쓰는 메서드 count
a = (1, 2, 2, 2, 3, 4, 2)
a.count(2)

##### 리스트###########################################
a_list = [2, 3, 7, None]
tup = ('foo', 'bar', 'baz')
b_list = list(tup)
b_list
b_list[1] = 'peekaboo'
b_list
# 리스트와 튜플은 객체의 1차원 순차 자료형이며 많은 함수에서 교차적으로 사용가능
gen = range(10)
gen
list(gen)

#append 메서드를 사용하여 리스트 값 추가
b_list.append('dwarf')
b_list
#insert: 특정 위치에 값 추가
b_list.insert(1, 'red')
b_list
#pop: 특정 위치의 값을 반환하고 해당 값을 리스트에서 삭제
b_list.pop(2)
b_list
# remove: 원소 삭제 (제일 앞부터)
b_list.remove('foo')
b_list

#in 예약어를 사용해서 리스트 값 검사
'dwarf' in b_list
'dwarf' not in b_list
       # 해시테이블을 이용한 파이썬의 사전이나 집합 자료구조처럼 즉각적으로 반환하지않고 많이 느림

### 리스트 이어붙이기 ########
[4, None, 'foo'] + [7, 8, (2, 3)]
# extend: 여러개의 값 추가
x = [4, None, 'foo']
x.extend([7, 8, (2, 3)])
x # 확인
      # 리스트를 이어붙이면 새로운 리스트를 생성하고 값을 복사하게 되므로 상대적으로 연산비용 높음
      #큰 리스트일수록 extend로 기존 리스트에 값을 추가하는 것이 좋음

##### 정렬 ####################################
# sort(함수): 새로운 리스트를 생성하지 않고 있는 그대로 리스트 정렬 가능 
a = [7,2, 5, 1, 3]
a.sort()
a

b = ['saw', 'small', 'He', 'foxes', 'six']
b.sort(key=len)
b

### 이진 탐색과 정렬된 리스트 유지하기
# bisect 모듈
# bisect.bisect(메서드): 새로운 숫자 추가 명령 했을 경우 추가될 숫자 인덱스 반환
# bisect.insort(메서드): 정렬된 상태 유지한채 값만 추가 (출력 안됨)
        # 인덱스(index, 색인): 위치 값을 뜻하고 0부터 시작
import bisect
c = [1, 2, 2, 2, 3, 4, 7]
bisect.bisect(c,2)
bisect.bisect(c,5)
bisect.insort(c,6)
c

### 슬라이싱
seq = [7,2, 3, 7, 5, 6, 0, 1]
seq[1:5]

seq[3:4] = [6, 3]
seq

seq[:5] #0~4까지
seq[3:] #4부터
seq[-4:]#뒤에서 5번째앞부터
seq[-6:-2] #뒤에서 6번째 앞부터 뒤에서 세번째까지(?)

seq[::2] # 하나씩 건너서 원소 선택
seq[::-1] # 역순으로 반환

### 내장 순차 자료형 함수

# enumerate(함수): 순차 자료형에서의 값과 그 위치(인덱스)를 dict에 넘겨줌
    # 사용전 예제 ------------------------------------
i = 0
for value in collection:
    #value를 사용하는 코드 작성
    i += 1
    # 사용후 예제 ------------------------------------
for i, value in enumerate(collection):
    #value를 사용하는 코드 작성
# -------------------------------------------------
some_list = ['foo', 'bar', 'baz']
mapping = {}
for i, v in enumerate(some_list):
    mapping[v] = i #value를 사용하는 코드 작성
    
mapping  # {값: 인덱스, ...} 출력                                                                                                                                                                                                                          

# sorted(함수): 정렬된 새로운 순차 자료형 반환, 리스트의 sort 매서드와 같은 인자 취함
        #(인자: 함수를 정의할 때 변수의 이름을 칭함/ 인수: 함수를 호출할대 전달하는 값)
sorted([7, 1, 2, 6, 0, 3, 2])
sorted('horse race')

# zip(함수): 여러 개의 리스트나 튜플 또는 다른 순차 자료형을 서로 짝지어서 튜플의 리스트를 생성
seq1 = ['foo', 'bar', 'baz']
seq2 = ['one', 'two', 'three']
zipped = zip(seq1, seq2)
list(zipped)
    # 반환되는 리스트의 크기는 넘겨받은 순차 자료형 중 가장 짧은 크기로 정해짐
seq3 = [False, True]

list(zip(seq1, seq2, seq3))

    # enumerate와 함께 사용: 여러 개의 순차 자료형을 동시에 순회하는 경우
for i, (a, b) in enumerate(zip(seq1,seq2)):
    print('{0}: {1},{2}'.format(i, a, b))
            # i : 인덱스, (a,b): zip(seq1,seq2)튜플의 리스트 변수   
            #format(함수): 문자열을 예쁘게 만듦
            #'인덱스: 값' 형식으로 문자열 출력

    #zip(*list): 리스트 로우를 컬럼으로 변환, 짝지어진 순차 자료형 풀어내기
pitchers = [('Nolan', 'Ryan'), ('Roger','Clemens'),
            ('Schilling', 'Curt')]
first_names, last_names = zip(*pitchers)
first_names
last_names

# reversed : 순차 자료형을 역순으로 순회
list(reversed(range(10)))
        #Generator이므로 list()나 for문으로 모든 값은 다 받아오기 전에는 순차 자료형 생성하지 않음
        #Generator: iterator(반복자)를 생성해주는 함수 
        #           함수 안에 yield(값(변수)) 키워드 사용
        

##### 사전 ########################################################
# dict(사전)는 파이썬 내장 자료구조 중 가장 중요
#dict = 해시맵 = 연관배열
# 키-값으로 이루어져 있음(모두 파이썬 객체)
empty_dict = {}
d1 = {'a' : 'some value', 'b': [1,2,3,4]}
d1

d1[7] = 'an integer' # 원래 사전에 키-값 추가
d1
d1['b'] # b키의 값 출력
    # 사전에 어떤 키가 있는지 리스트나 튜플과 같은 문법으로 확인
'b' in d1 # d1안에 b가 존재 하는가

d1[5] = 'some value'
d1
d1['dummy'] = 'another value'
d1
del d1[5]  # del 예약어나 pop메서드를 사용하여 사전 값 삭제
d1
ret = d1.pop('dummy')
ret
d1

    # keys(메서드): 키 담긴 iterator 반환
list(d1.keys())
    # values(메서드): 값 담긴 iterator 반환
list(d1.values())
    # update(메서드): 하나의 사전을 다른 사전과 합칠 수 있음
d1.update({'b' : 'foo', 'c' : 12})
d1
        # 이미 존재하는 키에 대해 update하면 이전 값은 사라짐.

### 순차 자료형에서 사전 생성하기 
    # 예제 ---------------------------------------------
    mapping = {}
    for key, value in zip(key_list, value_list):
        mapping[key] = value
    #--------------------------------------------------
mapping = dict(zip(range(5), reversed(range(5))))
mapping
            # reversed: 뒤집기 기능

### 기본값
    # 예제 ---------------------------------------------
    if key in some_dict:
        value = some_dict[key]
    else:
        value = default_value
        
    value = some_dict.get(key, default_value)
    #--------------------------------------------------
    #여러 단어를 시작 글자에 따라 사전에 리스트로 저장
words = ['apple', 'bat', 'bar', 'atom', 'book']
by_letter = {}
for word in words:
    letter = word[0]
    if letter not in by_letter:
        by_letter[letter] = [word]
    else:
        by_letter[letter].append(word)
        
by_letter

        #이해를 위해 풀어놓기
words = ['apple', 'bat', 'bar', 'atom', 'book']
by_letter = {}    # dictionary
for word in words:  
    letter = word[0]  # 문자열들의 첫번째 글자들만 letter로 지정
    print(letter)  # for문 처음돌았을때 apple 부터 있으므로 a부터 출력
    if letter not in by_letter: 
        print(by_letter) # 첫번째 돌릴때는 아무것도 없는 dictionary, 두번째는 {'a': ['apple']}가 들어있음
        by_letter[letter] = [word]
        print('if문 실행한 후 ', by_letter) # if문 실행한 후  {'a': ['apple'], 'b': ['bat']}
    else: # a,b 둘다 dictionary 안 에 있을 때 출력
        by_letter[letter].append(word)
        # if 에서는 apple, bat 출력되고 그러면 키가 a,b 둘다 dictionary에 잇으므로
        # else 실행됨
        #else부분이 충족될때까지 if 문 돌을때 for문부터 해서 돌아가짐
        
        
### 유효한 사전 키
#hash: 리스트를 사용할 수 없을 때, 빠른 접근 필요, 집계 필요
from enum import unique
from gc import collect
from multiprocessing import Condition
from unittest import result


hash('string')
hash((1,2,(2,3)))
hash((1,2,[2,3])) # 리스트는 변경이 가능한 값이므로 해시 불가능
    # 리스트를 튜플로 변경 (리스트를 키로 사용하기 위해)
d = {}
d[tuple([1,2,3])] = 5
d