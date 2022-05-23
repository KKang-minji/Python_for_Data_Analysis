##### 집합 ###########################################
# 집합: 유일한 원소만 담는 정렬되지 않은 자료형
    #  값은 없고 키만 존재
set([2, 2, 2, 1, 3, 3])
{2, 2, 2, 1, 3, 3}

# 집합은 합집합, 교집합, 차집합, 대칭차집합 같은 산술 집합 연산 제공
a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7, 8}
# 합집합: union(메서드) or | 이항 연산자
a.union(b)
a | b
# 교집합: intersection(메서드) or & 이항 연산자
a.intersection(b)
a & b

    # 모든 논리 집합 연산은 연산 결과를 좌항에 대입하는 함수도 따로 제공
c = a.copy()
c |= b
c
d =a.copy()
d &= b
d
    # 리스트 같은 원소를 담으려면 튜플로 변경
my_data = [1, 2, 3, 4]
my_set = {tuple(my_data)}
my_set
    #어떤 집합이 부분집합인지 확대집합인지 검사
a_set = {1,2,3,4,5}
{1,2,3}.issubset(a_set) # 부분집합 인가
a_set.issuperset({1,2,3}) # 확대집합 인가
    # 집합 내용 같으면 두 집합 동일
{1,2,3} == {3,2,1}

##### 리스트, 집합, 사전 표기법 #######################################
    # 예제 -------------------------------------------------------
    [expr for val in collection if Condition]
    # 이를 반복문으로 하면
    result = []
    for val in collection:
        if condition:
            result.append(expr)
    #------------------------------------------------------------
# string(바꾸고자하는 문자열 객체).upper: 문자열 대문자로 변환
    # 문자열 길이가 2이하인 문자열은 제외하고 나머지 대문자 변환
strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
[x.upper() for x in strings if len(x) > 2]

    # 문자열 길이
unique_lengths = {len(x) for x in strings}
unique_lengths
# map(함수)
set(map(len, strings))
# 리스트에서 문자열의 위치를 담고있는 사전을 생성
loc_mapping = {val : index for index, val in enumerate(strings)}
loc_mapping

### 중첩된 리스트 표기법
        # 리스트 표기법에서 for 부분은 중첩의 순서에 따라 나열, 필터 조건은 끝에 위치
all_data = [['John', 'Emily', 'Michael', 'Mary', 'Steven'],
            ['Maria','Juan','Javier','Natalia','Pilar']]
    # 각 이름에서 알파뱃 e 가 2개 이상 포함된 이름의 목록
result = [name for names in all_data for name in names
          if name.count('e') >= 2]
result
    # 숫자 튜플이 담긴 리스트를 그냥 단순한 리스트로 변환
some_tuples = [(1,2,3), (4,5,6), (7,8,9)]
flattened = [x for tup in some_tuples for x in tup]
flattened
    # 2단계 이상의 중첩
[[x for x in tup] for tup in some_tuples]

##### 함수 #########################################################
    # 예제 -----------------------------------------------------
    def my_function(x,y,z=1.5):
        if z > 1:
            return z * (x, y)
        else:
            return z / (x + y)
    #----------------------------------------------------------

### 네임스페이스, 스코프 지역 함수
        # 함수 = 전역과 지역(스코프)에서 변수를 참조
        # 네임스페이스 : 변수의 스코프, 함수 내에서 선언된 변수
        # 지역 네임스페이스는 함수가 호출될 때 생성되며 함수의 인자를 통해 즉시 생성
        # 함수 실행 끝나면 지역 네임스페이스 사라짐
    # 예제 ------------------------------------------------------
    # 함수안에 변수에 값 대입
    def func():
        a = []
        for i in range(5):
            a.append(i)
            #func()함수를 호출하면 비어있는 리스트 a 생성되고
            #다섯개의 원소가 리스트에 추가 되며 함수가 끝나면 이 리스트 a는 사라짐
    #-----------------------------------------------------------
# 위와 달리 함수의 스코프 밖에서 변수에 값을 대입하려면 
# global 예약어를  사용하여 전역변수로 선언해야함
a = None
def bind_a_vaiable():
    global a
    a = []
    bind_a_vaiable()

print(a)    
        # global예약어는 전역변수로 선언하는데 사용됨
        # 일반적으로 전역변수는 시스템 전체의 상태를 저장하기 위한 용도로 사용되므로
        #전역변수를 많이 사용하면 클래스를 사용한 객체지향 프로그래밍이 적잘한 상황이라는 반증
        
### 여러 값 반환하기
    # 하나의 함수에서 여러 개의 값을 반환
    # 예제--------------------------------------------------------------------
    def f():
        a = 5
        b = 6
        c = 7
        return a, b, c
    a, b, c = f()
    # -----------------------------------------------------------------------
    
### 함수도 객체다.
states = ['Alabama', 'Georgia!', 'Georgia', 'georgia', 'FIOrIda', 'southcarolina##',
          'West virginia?']
    # 공백 문자나 필요 없는 문장 부호를 제거하거나 대소문자를 맞추는 등의 작업이 필요
    # 내장 문자열 메서드와 정규 표현식을 위한 re 표준 라이브러리를 이용하여 해결
import re
def clean_strings(strings):
    result = []
    for value in strings:
        value = value.strip()
        value = re.sub('[!#?]', '', value)
        result.append(value)
    return result

clean_strings(states)

def remove_punctuation(value):
    return re.sub('[!#?]','',value)

clean_ops = [str.strip, remove_punctuation, str.title]

def clean_strings(strings, ops):
    result = []
    for value in strings:
        for function in ops:
            value = function(value)
        result.append(value)
    return result

clean_strings(states, clean_ops)

    # 순차적 자료형에 대해 함수 적용
    #map 함수 이용하여 함수를 인자로 사용
for x in map(remove_punctuation, states):
    print(x)

### 익명 함수
# lambda: 예약어로 정의하고 익명함수를 선언한다는 의미
# 람다함수를 사용하면 실제 함수를 선언하거나 람다함수를 지역변수에 대입하는 것보다
# 코드를 적게 쓰고 더 간결해지기 때문
def short_function(x):
    return x*2

equiv_anon = lambda x: x * 2

    # 예제 ---------------------------------------------------------
    def apply_to_list(some_list, f):
        return [f(x) for x in some_list]
    
    ints = [4, 0, 1, 5, 6]
    apply_to_list(ints, lambda x: x * 2)
    # [x * 2 for x in ints]라고 해도 되지만 이렇게 하면 apply_to_list함수에
    # 사용자 연산 간결하게 전달 가능
    #--------------------------------------------------------------
    # 문자열에서 다양한 문자가 포함된 순서
strings = ['foo', 'card', 'bar', 'aaaa', 'abab']
strings.sort(key=lambda x: len(set(list(x))))
strings #set이라는 집합으로 만들면서 동일한 원소(같은 문자)가 하나의 원소로 변함


### 커링: 커링은 인자를 여러개 받는 함수를 분리하여, 인자를 하나씩만 받는 함수의 체인으로 만드는 방법이다.
        #함수형 프로그래밍 기법 중 하나로 함수를 재사용하는데 유용하게 쓰일 수 있는 기법이다.
        # 인자: 함수를 호출하며 값으로 전달되는 것
    # 예제 ----------------------------------------------------------
    def add_numbers(x,y):
        return x + y     # 2개의 숫자를 더하는 함수가 있다고 가정
    add_five = lambda y: add_numbers(5, y) 
    #하나의 변수만 인자로 받아 5를 더해주는 새로운 함수 add_five를 생성
    from functools import partial
    add_five = partial(add_numbers, 5)
    # --------------------------------------------------------------

### 제너레이터: 순회 가능한 객체를 생성하는 간단한 방법
some_dict = {'a':1, 'b':2, 'c':3}
for key in some_dict: #some_dict에서 이터레이터를 생성
    print(key)

dict_iterator = iter(some_dict)

dict_iterator
    #min,max,sum같은 내장 메서드와 list,tuple 과 같은 자료구조를 생성하는 메서드 포함
list(dict_iterator)

def squares(n=10):
    print('Generating squares from 1 to {0}'.format(n**2))
    for i in range(1, n+1):
        yield i ** 2
      
gen = squares()  

for x in gen:
    print(x, end=' ')













