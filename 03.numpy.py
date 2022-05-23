import numpy as np
my_arr = np.arange(1000000)
my_list = list(range(1000000))
# 각 배열과 리스트 원소에 곱하기 2
%time for _ in range(10): my_arr2 = my_arr * 2
%time for _ in range(10): my_list2 = [x * 2 for x in my_list]

### numpy ndarray: 다차원 배열 객체
import numpy as np
data = np.random.randn(2,3)
data
data * 10
data + data
data.shape          # shape: 각 차원의 크기를 알려줌
data.dtype          # dtype: 튜플과 배열에 저장된 자료형을 알려줌

### ndarray 생성하기
# array 함수를 사용하여 배열 생성하기
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
arr1
    # 다차원 배열도 가능
data2 = [[1,2,3,4], [5,6,7,8]]
arr2 = np.array(data2)
arr2

arr2.ndim           # ndim:
arr2.shape

arr1.dtype

np.zeros(10)

np.zeros((3,6))

np.zeros((2,3,2))

np.arange(15)

arr1 = np.array([1,2,3], dtype= np.float64)
arr2 = np.array([1,2,3], dtype=np.int32)
arr1.dtype
arr2.dtype

arr = np.array([3.7,-1.2, -2.6, 0.5, 12.9, 10.1])
arr
arr.astype(np.int32)

numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
numeric_strings.astype(float)

arr2d = np.array([[1,2,3], [4,5,6],[7,8,9]])
arr2d
arr2d[:2]
arr2d[:2, 1:]
arr2d[:2,2]
arr2d[:, :1]
arr2d[:2, 1:] = 0
arr2d

### 불리언 값으로 선택하기
# randn 을 사용 하여 임의의 표준 정규분포 데이터 생성
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will','Joe', 'Joe'])
data = np.random.randn(7,4)
names
data

names == 'Bob'
data[names == 'Bob']
data[names == 'Bob', 2:]
data[names == 'Bob', 3]
# 'Bob'이 아닌 요소들을 선택하려면 != 연산자를 사용하거나 ~ 를 사용해서 조건절 부인
names != 'Bob'
data[~(names == 'Bob')]

#세 가지 이름 중에서 두 가지 이름 선택하려면 &이나 |같은 논리 연산자를 사용한 여러개의 불리언 조건 사용
mask = (names == 'Bob')|(names == 'Will')
mask
data[mask]

data[data<0] = 0
data
data[names != 'Joe'] = 7
data

### 팬시 색인
arr = np.empty((8,4))
for i in range(8):
    arr[i] = i
arr
# 특정한 순서로 로우 추출
arr[[4,3,0,6]]
# 밑에서 부터 로우 추출
arr[[-3, -5, -7]]
# 다차원 색인 배열 넘기는 것
arr = np.arange(32).reshape((8,4))
arr             # 0~31 범위로 8행 4열 만들기
arr.T       # 배열 전치






