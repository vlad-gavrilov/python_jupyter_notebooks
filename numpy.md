## **Введение в библиотеку NumPy**
![Image](https://numpy.org/doc/stable/_static/numpy_logo.png)  
*NumPy* это библиотека Python для векторизованных вычислений. Написана на языке C.  
Читается как */ˈnʌmpaɪ/* (нам-пай).  
Главные преимущества NumPy перед стандартными функциями Python это большая **скорость** и меньшее количество используемой **памяти**.  

- [Официальный сайт](https://numpy.org/)  
    - [Быстрый старт](https://numpy.org/doc/stable/user/quickstart.html)
    - [Справочник](https://numpy.org/doc/stable/reference/index.html)    
- [GitHub](https://github.com/numpy/numpy)
- [Wikipedia](https://en.wikipedia.org/wiki/NumPy)
- [Курс "Введение в анализ данных" (2019)](https://www.youtube.com/playlist?list=PLrCZzMib1e9p6lpNv-yt6uvHGyBxQncEh)
- [100 задач NumPy](https://github.com/rougier/numpy-100)
- [Data Science Notebooks](https://github.com/donnemartin/data-science-ipython-notebooks)



---
### **Подключение библиотеки**


```python
import numpy as np

np.__version__
```




    '1.17.3'




```python
a = list(range(1000000))
```


```python
%%timeit

[element * element for element in a]
```

    143 ms ± 16.2 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
a = np.arange(1000000)
```


```python
%%timeit

a * a
```

    5.23 ms ± 394 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


### **Создание массивов**

![image](https://numpy.org/doc/stable/_images/np_array.png)


```python
a = np.array([1, 2, 3, 4, 5], dtype=np.float64)

a
```




    array([1., 2., 3., 4., 5.])



![Image](https://numpy.org/doc/stable/_images/np_create_matrix.png)


```python
a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [-1, -2, -3]
])

a
```




    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 7,  8,  9],
           [-1, -2, -3]])




```python
a.shape
```




    (4, 3)




```python
type(a.shape)
```




    tuple




```python
a.ndim
```




    2




```python
a.dtype
```




    dtype('int64')



### **Типы данных**
Основные типы данных в NumPy:  
`np.int8`: Byte (-128 to 127)  
`np.int16`: Integer (-32768 to 32767)  
`np.int32`: Integer (-2147483648 to 2147483647)  
`np.int64`: Integer (-9223372036854775808 to 9223372036854775807)  

`np.uint8`: Unsigned integer (0 to 255)  
`np.uint16`: Unsigned integer (0 to 65535)  
`np.uint32`: Unsigned integer (0 to 4294967295)  
`np.uint64`: Unsigned integer (0 to 18446744073709551615)  

`np.float32`: Note that this matches the precision of the builtin python float  
`np.float64`: Note that this matches the precision of the builtin python float  

`np.complex64`: Complex number, represented by two 32-bit floats  
`np.complex128`: Note that this matches the precision of the builtin python complex  

[И другие...](https://numpy.org/doc/stable/user/basics.types.html#array-types-and-conversions-between-types)



```python
a
```




    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 7,  8,  9],
           [-1, -2, -3]])




```python
a.dtype
```




    dtype('int64')




```python
a.astype(np.complex128)
```




    array([[ 1.+0.j,  2.+0.j,  3.+0.j],
           [ 4.+0.j,  5.+0.j,  6.+0.j],
           [ 7.+0.j,  8.+0.j,  9.+0.j],
           [-1.+0.j, -2.+0.j, -3.+0.j]])



### **Размерность массивов**
Массив в numpy хранится как единый последовательный набор данных. У этого набора есть характеристики:  
- `dtype` - тип данных
- `shape` - форма
- `strides` - количество байт для доступа к следующему элементу

![Image](https://miro.medium.com/max/1000/1*Ikn1J6siiiCSk4ivYUhdgw.png)


```python
a
```




    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 7,  8,  9],
           [-1, -2, -3]])




```python
a.dtype
```




    dtype('int64')




```python
a.dtype.itemsize
```




    8




```python
a.shape
```




    (4, 3)




```python
a.strides
```




    (24, 8)



![Image](https://numpy.org/doc/stable/_images/np_reshape.png)


```python
b = a.reshape(2, 6)

b
```




    array([[ 1,  2,  3,  4,  5,  6],
           [ 7,  8,  9, -1, -2, -3]])




```python
b.strides
```




    (48, 8)




```python
a.reshape(3, -1)
```




    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, -1, -2, -3]])



### **Хранение массивов в памяти**
При присваивании массива другой переменной новый массив будет ссылаться на ту же область памяти, что и первоначальный.


```python
a
```




    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 7,  8,  9],
           [-1, -2, -3]])




```python
b = a.reshape(2, 6)

b
```




    array([[ 1,  2,  3,  4,  5,  6],
           [ 7,  8,  9, -1, -2, -3]])




```python
a[2, 1] = 100
```


```python
a
```




    array([[  1,   2,   3],
           [  4,   5,   6],
           [  7, 100,   9],
           [ -1,  -2,  -3]])




```python
b
```




    array([[  1,   2,   3,   4,   5,   6],
           [  7, 100,   9,  -1,  -2,  -3]])



Метод `flatten` "вытягивает" массив в одну линию и создает его полную копию.


```python
b = a.flatten()
a[3, 2] = 111

a
```




    array([[  1,   2,   3],
           [  4,   5,   6],
           [  7, 100,   9],
           [ -1,  -2, 111]])




```python
b
```




    array([  1,   2,   3,   4,   5,   6,   7, 100,   9,  -1,  -2,  -3])



Метод `ravel` "вытягивает" массив в одну линию, но полной копии не создает.


```python
b = a.ravel()
a[3, 2] = 111

b
```




    array([  1,   2,   3,   4,   5,   6,   7, 100,   9,  -1,  -2, 111])



Атрибут `T` транспонирует исходный массив, но полной копии не создает.
![Image](https://numpy.org/doc/stable/_images/np_transposing_reshaping.png)

Транспонирование матрицы A:  
$A = \begin{bmatrix}
1 & 2\\
3 & 4\\
5 & 6
\end{bmatrix}$

$A ^ \tau = \begin{bmatrix}
1 & 3 & 5\\
2 & 4 & 6
\end{bmatrix}$


```python
a
```




    array([[  1,   2,   3],
           [  4,   5,   6],
           [  7, 100,   9],
           [ -1,  -2, 111]])




```python
c = a.T

c
```




    array([[  1,   4,   7,  -1],
           [  2,   5, 100,  -2],
           [  3,   6,   9, 111]])




```python
a[0, 0] = 222

c
```




    array([[222,   4,   7,  -1],
           [  2,   5, 100,  -2],
           [  3,   6,   9, 111]])



### **Фиктивные оси (оси с размерностью 1)**
Используется для того, чтобы библиотека могла проводить операции над массивами с одинаковой размерностью.


```python
a = np.arange(15)

a
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])




```python
a.shape
```




    (15,)




```python
a.ndim
```




    1




```python
b = a[np.newaxis, :]
```


```python
b
```




    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]])




```python
b.shape
```




    (1, 15)




```python
b.ndim
```




    2




```python
c = a[:, np.newaxis]
```


```python
c
```




    array([[ 0],
           [ 1],
           [ 2],
           [ 3],
           [ 4],
           [ 5],
           [ 6],
           [ 7],
           [ 8],
           [ 9],
           [10],
           [11],
           [12],
           [13],
           [14]])




```python
a[np.newaxis, np.newaxis, :]
```




    array([[[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]]])




```python
a[np.newaxis, np.newaxis, :].shape
```




    (1, 1, 15)



### **Создание особенных массивов**
![Image](https://numpy.org/doc/stable/_images/np_ones_zeros_matrix.png)


```python
np.zeros((3, 2))
```




    array([[0., 0.],
           [0., 0.],
           [0., 0.]])




```python
np.zeros_like(a)
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])




```python
np.ones((3, 2))
```




    array([[1., 1.],
           [1., 1.],
           [1., 1.]])




```python
np.eye(5)
```




    array([[1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1.]])




```python
np.arange(10, 20)
```




    array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])




```python
np.arange(25, 50, 5)
```




    array([25, 30, 35, 40, 45])




```python
np.arange(1, 15, 0.5)
```




    array([ 1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. ,  5.5,  6. ,
            6.5,  7. ,  7.5,  8. ,  8.5,  9. ,  9.5, 10. , 10.5, 11. , 11.5,
           12. , 12.5, 13. , 13.5, 14. , 14.5])




```python
np.linspace(0, 10, 11)
```




    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])



### **Генератор случайных чисел**


```python
np.random.seed(12321)
```


```python
np.random.rand(15)
```




    array([0.18325525, 0.7969142 , 0.05063654, 0.08596682, 0.43223338,
           0.70304694, 0.43190833, 0.27597651, 0.58900897, 0.43063381,
           0.97426208, 0.72385849, 0.45790999, 0.09685338, 0.85620942])




```python
np.random.randint(-9, 10, 15)
```




    array([ 0, -2,  9,  2, -1, -7,  9,  4,  4, -1,  9, -9, -6, -2,  0])




```python
np.random.permutation(5)
```




    array([0, 4, 1, 2, 3])




```python
np.random.choice(4, size=20)
```




    array([1, 2, 2, 2, 0, 1, 3, 3, 0, 3, 3, 2, 1, 1, 0, 0, 2, 1, 1, 0])



### **Создание массивов из существующих массивов**
Метод `array` создает новый массив, который занимает отдельную область памяти.  
Метод `asarray` создает новый массив, указывающий на ту же память, что и исходный массив.  


```python
a = np.array(range(10))
b = np.array(a)
c = np.asarray(a)

a, b, c
```




    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
     array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
     array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))




```python
a[5] = 0

a, b, c
```




    (array([0, 1, 2, 3, 4, 0, 6, 7, 8, 9]),
     array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
     array([0, 1, 2, 3, 4, 0, 6, 7, 8, 9]))



### **Поэлементные операции над массивами**
![Image](https://numpy.org/doc/stable/_images/np_multiply_broadcasting.png)


```python
a = np.array(range(36)).reshape(6, 6)

a
```




    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17],
           [18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29],
           [30, 31, 32, 33, 34, 35]])




```python
a + 2
```




    array([[ 2,  3,  4,  5,  6,  7],
           [ 8,  9, 10, 11, 12, 13],
           [14, 15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24, 25],
           [26, 27, 28, 29, 30, 31],
           [32, 33, 34, 35, 36, 37]])




```python
a - 2
```




    array([[-2, -1,  0,  1,  2,  3],
           [ 4,  5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14, 15],
           [16, 17, 18, 19, 20, 21],
           [22, 23, 24, 25, 26, 27],
           [28, 29, 30, 31, 32, 33]])




```python
a * 10
```




    array([[  0,  10,  20,  30,  40,  50],
           [ 60,  70,  80,  90, 100, 110],
           [120, 130, 140, 150, 160, 170],
           [180, 190, 200, 210, 220, 230],
           [240, 250, 260, 270, 280, 290],
           [300, 310, 320, 330, 340, 350]])




```python
a / 10
```




    array([[0. , 0.1, 0.2, 0.3, 0.4, 0.5],
           [0.6, 0.7, 0.8, 0.9, 1. , 1.1],
           [1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
           [1.8, 1.9, 2. , 2.1, 2.2, 2.3],
           [2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
           [3. , 3.1, 3.2, 3.3, 3.4, 3.5]])




```python
a ** 2
```




    array([[   0,    1,    4,    9,   16,   25],
           [  36,   49,   64,   81,  100,  121],
           [ 144,  169,  196,  225,  256,  289],
           [ 324,  361,  400,  441,  484,  529],
           [ 576,  625,  676,  729,  784,  841],
           [ 900,  961, 1024, 1089, 1156, 1225]])




```python
np.power(a, 2)
```




    array([[   0,    1,    4,    9,   16,   25],
           [  36,   49,   64,   81,  100,  121],
           [ 144,  169,  196,  225,  256,  289],
           [ 324,  361,  400,  441,  484,  529],
           [ 576,  625,  676,  729,  784,  841],
           [ 900,  961, 1024, 1089, 1156, 1225]])




```python
np.exp(a)
```




    array([[1.00000000e+00, 2.71828183e+00, 7.38905610e+00, 2.00855369e+01,
            5.45981500e+01, 1.48413159e+02],
           [4.03428793e+02, 1.09663316e+03, 2.98095799e+03, 8.10308393e+03,
            2.20264658e+04, 5.98741417e+04],
           [1.62754791e+05, 4.42413392e+05, 1.20260428e+06, 3.26901737e+06,
            8.88611052e+06, 2.41549528e+07],
           [6.56599691e+07, 1.78482301e+08, 4.85165195e+08, 1.31881573e+09,
            3.58491285e+09, 9.74480345e+09],
           [2.64891221e+10, 7.20048993e+10, 1.95729609e+11, 5.32048241e+11,
            1.44625706e+12, 3.93133430e+12],
           [1.06864746e+13, 2.90488497e+13, 7.89629602e+13, 2.14643580e+14,
            5.83461743e+14, 1.58601345e+15]])




```python
np.cos(a)
```




    array([[ 1.        ,  0.54030231, -0.41614684, -0.9899925 , -0.65364362,
             0.28366219],
           [ 0.96017029,  0.75390225, -0.14550003, -0.91113026, -0.83907153,
             0.0044257 ],
           [ 0.84385396,  0.90744678,  0.13673722, -0.75968791, -0.95765948,
            -0.27516334],
           [ 0.66031671,  0.98870462,  0.40808206, -0.54772926, -0.99996083,
            -0.53283302],
           [ 0.42417901,  0.99120281,  0.64691932, -0.29213881, -0.96260587,
            -0.74805753],
           [ 0.15425145,  0.91474236,  0.83422336, -0.01327675, -0.84857027,
            -0.90369221]])




```python
a > 20
```




    array([[False, False, False, False, False, False],
           [False, False, False, False, False, False],
           [False, False, False, False, False, False],
           [False, False, False,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True]])



### **Вставка и удаление**


```python
a = np.random.randint(10, size=5)

a
```




    array([1, 3, 6, 2, 1])




```python
a = np.insert(a, 3, 100)
```


```python
a
```




    array([  1,   3,   6, 100,   2,   1])




```python
a = np.delete(a, 2)
```


```python
a
```




    array([  1,   3, 100,   2,   1])



### **Агрегирующие функции**

![Image](https://numpy.org/doc/stable/_images/np_aggregation.png)


```python
a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [-1, -2, -3]
])

a
```




    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 7,  8,  9],
           [-1, -2, -3]])




```python
#np.max(a)
a.max()
```




    9




```python
#np.min(a)
a.min()
```




    -3




```python
#np.sum(a)
a.sum()
```




    39




```python
#np.argmax(a)
a.argmax()
```




    8




```python
#np.prod(a)
a.prod()
```




    -2177280




```python
a = np.random.randint(-9, 10, size=(4, 8))

a
```




    array([[ 2,  9, -3,  8,  3,  5,  7, -2],
           [-4, -8, -6, -5,  3,  2, -5, -1],
           [-8, -3, -1, -6, -3,  1, -3, -7],
           [-1,  4, -9, -5,  6, -1, -4,  6]])



![Image](https://numpy.org/doc/stable/_images/np_matrix_aggregation_row.png)


```python
a.max(axis=0)
```




    array([ 2,  9, -1,  8,  6,  5,  7,  6])




```python
a.max(axis=1)
```




    array([9, 3, 1, 6])



### **Операции над булевыми массивами**


```python
a = np.array([False, True, True, False, True])

a
```




    array([False,  True,  True, False,  True])




```python
#np.any(a)
a.any()
```




    True




```python
#np.all(a)
a.all()
```




    False




```python
a = np.array([
    [False, True, True, False, True],
    [True, True, False, False, True]
])

a
```




    array([[False,  True,  True, False,  True],
           [ True,  True, False, False,  True]])




```python
a.any(axis=0)
```




    array([ True,  True,  True, False,  True])




```python
a.all(axis=0)
```




    array([False,  True, False, False,  True])




```python
a.any(axis=1)
```




    array([ True,  True])




```python
a.all(axis=1)
```




    array([False, False])



### **Бинарные операции**
![Image](https://numpy.org/doc/stable/_images/np_sub_mult_divide.png)


```python
a = np.random.randint(-9, 10, size=(3, 4))

a
```




    array([[-1, -9, -2, -5],
           [ 9,  2,  2, -2],
           [ 3, -7,  0,  3]])




```python
b = np.random.randint(-9, 10, size=(3, 4))

b
```




    array([[-3, -1,  5,  6],
           [-2, -3,  8, -3],
           [ 5,  8,  6, -3]])




```python
a + b
```




    array([[ -4, -10,   3,   1],
           [  7,  -1,  10,  -5],
           [  8,   1,   6,   0]])




```python
a * b
```




    array([[  3,   9, -10, -30],
           [-18,  -6,  16,   6],
           [ 15, -56,   0,  -9]])




```python
a > b
```




    array([[ True, False, False, False],
           [ True,  True, False,  True],
           [False, False, False,  True]])




```python
a == b
```




    array([[False, False, False, False],
           [False, False, False, False],
           [False, False, False, False]])



### **Бинарные операции над булевыми массивами**


```python
a = np.array([
    [False, True, True, False, True],
    [True, True, False, False, True]
])

a
```




    array([[False,  True,  True, False,  True],
           [ True,  True, False, False,  True]])




```python
b = np.array([
    [False, False, True, False, False],
    [True, False, False, True, True]
])

b
```




    array([[False, False,  True, False, False],
           [ True, False, False,  True,  True]])




```python
np.logical_not(a)
```




    array([[ True, False, False,  True, False],
           [False, False,  True,  True, False]])




```python
np.logical_and(a, b)
```




    array([[False, False,  True, False, False],
           [ True, False, False, False,  True]])




```python
np.logical_or(a, b)
```




    array([[False,  True,  True, False,  True],
           [ True,  True, False,  True,  True]])




```python
np.logical_xor(a, b)
```




    array([[False,  True, False, False,  True],
           [False,  True, False,  True, False]])



### **Матричные операции**


```python
a = np.arange(5, 17).reshape(3, 4)

a
```




    array([[ 5,  6,  7,  8],
           [ 9, 10, 11, 12],
           [13, 14, 15, 16]])




```python
b = np.arange(-5, 3).reshape(4, 2)

b
```




    array([[-5, -4],
           [-3, -2],
           [-1,  0],
           [ 1,  2]])




```python
# Ошибка!

# a * b
```


```python
a.dot(b)
```




    array([[ -42,  -16],
           [ -74,  -32],
           [-106,  -48]])




```python
np.matmul(a, b)
```




    array([[ -42,  -16],
           [ -74,  -32],
           [-106,  -48]])




```python
# m1 = np.asmatrix(a)
m1 = np.matrix(a)

m1
```




    matrix([[ 5,  6,  7,  8],
            [ 9, 10, 11, 12],
            [13, 14, 15, 16]])




```python
# m2 = np.asmatrix(b)
m2 = np.matrix(b)

m2
```




    matrix([[-5, -4],
            [-3, -2],
            [-1,  0],
            [ 1,  2]])




```python
m1 * m2
```




    matrix([[ -42,  -16],
            [ -74,  -32],
            [-106,  -48]])




```python
m1.dot(m2)
```




    matrix([[ -42,  -16],
            [ -74,  -32],
            [-106,  -48]])




```python
np.matmul(m1, m2)
```




    matrix([[ -42,  -16],
            [ -74,  -32],
            [-106,  -48]])



### **Срезы**

![Image](https://numpy.org/doc/stable/_images/np_indexing.png)

массив[*первый индекс* : *последний индекс* : *шаг*]

Если одно из этих значений пропущено, то вместо него принимается значение по умолчанию:  
- первый индекс = **0**
- последний индекс = **длина массива**
- шаг = **1**  



```python
a = np.arange(15)

a
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])




```python
a[0], a[4], a[len(a) - 2]
```




    (0, 4, 13)




```python
a[-1]
```




    14




```python
a[::2]
```




    array([ 0,  2,  4,  6,  8, 10, 12, 14])




```python
a[1::2]
```




    array([ 1,  3,  5,  7,  9, 11, 13])



### **Использование логических масок**


```python
a = np.random.randint(-9, 10, 15)

a
```




    array([-8,  1, -2,  5,  0,  1, -9,  9, -3,  4, -8,  4, -2,  8, -5])




```python
a > 3
```




    array([False, False, False,  True, False, False, False,  True, False,
            True, False,  True, False,  True, False])




```python
a[a > 3]
```




    array([5, 9, 4, 4, 8])




```python
a % 2 == 0
```




    array([ True, False,  True, False,  True, False, False, False, False,
            True,  True,  True,  True,  True, False])




```python
a[a % 2 == 0]
```




    array([-8, -2,  0,  4, -8,  4, -2,  8])




```python
a[np.logical_or(a % 2 == 0, a % 3 == 0)]
```




    array([-8, -2,  0, -9,  9, -3,  4, -8,  4, -2,  8])




```python
a[np.logical_and(a % 2 == 0, a % 3 == 0)]
```




    array([0])



### **Индексация в многомерных массивах**
![Image](https://numpy.org/doc/stable/_images/np_matrix_indexing.png)


```python
a = np.random.randint(-9, 10, (5, 5))

a
```




    array([[ 9,  8,  3, -3, -3],
           [-7,  7, -7, -5, -8],
           [ 2, -1, -7,  8, -3],
           [-5, -8,  8,  1, -7],
           [ 3, -6, -9,  4,  6]])




```python
# Способ хорош для стандартный Python-списков, но не для NumPy-массивов

a[2][0]
```




    2




```python
a[2, 0]
```




    2




```python
a[3, :]
```




    array([-5, -8,  8,  1, -7])




```python
a[:, 2]
```




    array([ 3, -7, -7,  8, -9])




```python
a[:, ::2]
```




    array([[ 9,  3, -3],
           [-7, -7, -8],
           [ 2, -7, -3],
           [-5,  8, -7],
           [ 3, -9,  6]])




```python
a
```




    array([[ 9,  8,  3, -3, -3],
           [-7,  7, -7, -5, -8],
           [ 2, -1, -7,  8, -3],
           [-5, -8,  8,  1, -7],
           [ 3, -6, -9,  4,  6]])




```python
a < 0
```




    array([[False, False, False,  True,  True],
           [ True, False,  True,  True,  True],
           [False,  True,  True, False,  True],
           [ True,  True, False, False,  True],
           [False,  True,  True, False, False]])




```python
a[a < 0]
```




    array([-3, -3, -7, -7, -5, -8, -1, -7, -3, -5, -8, -7, -6, -9])




```python
np.where(a < 0)
```




    (array([0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4]),
     array([3, 4, 0, 2, 3, 4, 1, 2, 4, 0, 1, 4, 1, 2]))




```python
a[np.where(a < 0)]
```




    array([-3, -3, -7, -7, -5, -8, -1, -7, -3, -5, -8, -7, -6, -9])




```python
a
```




    array([[ 9,  8,  3, -3, -3],
           [-7,  7, -7, -5, -8],
           [ 2, -1, -7,  8, -3],
           [-5, -8,  8,  1, -7],
           [ 3, -6, -9,  4,  6]])




```python
a[[3, 4, 1]]
```




    array([[-5, -8,  8,  1, -7],
           [ 3, -6, -9,  4,  6],
           [-7,  7, -7, -5, -8]])




```python
a[:, [2, 4, 0]]
```




    array([[ 3, -3,  9],
           [-7, -8, -7],
           [-7, -3,  2],
           [ 8, -7, -5],
           [-9,  6,  3]])




```python
a = np.arange(24).reshape(2, 3, 4)

a
```




    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])




```python
a[..., 0]
```




    array([[ 0,  4,  8],
           [12, 16, 20]])




```python
a[0, ...]
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])




```python
a[..., 0, :]
```




    array([[ 0,  1,  2,  3],
           [12, 13, 14, 15]])



### **Сортировка массивов**


```python
a = np.random.randint(-99, 100, size=(25, 4))

a
```




    array([[-68, -73,  30,  34],
           [-16,  86,  41,  42],
           [ 80,  39, -87, -97],
           [ 76,  17, -37,  81],
           [ 25, -72,  10,  56],
           [-23, -81,  53,  11],
           [ 89, -92, -26, -31],
           [ 35, -51,  -5,  80],
           [-99,   2, -57, -62],
           [ 38, -68, -47, -32],
           [ 84,  49,  92,  47],
           [-21, -93, -32, -13],
           [ 44, -44,  37,  30],
           [ 96,  56,  92, -63],
           [ 14,  28, -48, -21],
           [ 65,  16,  38,  -7],
           [ 32, -95, -22, -56],
           [ 28,  -8, -17,  59],
           [-34, -11,  42, -66],
           [  6,  12, -52,  -9],
           [ 53, -82, -53, -44],
           [ 84, -84, -65,   8],
           [-46, -56, -27,  16],
           [-85, -13,  99,  26],
           [-74, -26,  -4,  36]])




```python
np.sort(a)
```




    array([[-73, -68,  30,  34],
           [-16,  41,  42,  86],
           [-97, -87,  39,  80],
           [-37,  17,  76,  81],
           [-72,  10,  25,  56],
           [-81, -23,  11,  53],
           [-92, -31, -26,  89],
           [-51,  -5,  35,  80],
           [-99, -62, -57,   2],
           [-68, -47, -32,  38],
           [ 47,  49,  84,  92],
           [-93, -32, -21, -13],
           [-44,  30,  37,  44],
           [-63,  56,  92,  96],
           [-48, -21,  14,  28],
           [ -7,  16,  38,  65],
           [-95, -56, -22,  32],
           [-17,  -8,  28,  59],
           [-66, -34, -11,  42],
           [-52,  -9,   6,  12],
           [-82, -53, -44,  53],
           [-84, -65,   8,  84],
           [-56, -46, -27,  16],
           [-85, -13,  26,  99],
           [-74, -26,  -4,  36]])




```python
# Возвращает копию

np.sort(a.ravel())
```




    array([-99, -97, -95, -93, -92, -87, -85, -84, -82, -81, -74, -73, -72,
           -68, -68, -66, -65, -63, -62, -57, -56, -56, -53, -52, -51, -48,
           -47, -46, -44, -44, -37, -34, -32, -32, -31, -27, -26, -26, -23,
           -22, -21, -21, -17, -16, -13, -13, -11,  -9,  -8,  -7,  -5,  -4,
             2,   6,   8,  10,  11,  12,  14,  16,  16,  17,  25,  26,  28,
            28,  30,  30,  32,  34,  35,  36,  37,  38,  38,  39,  41,  42,
            42,  44,  47,  49,  53,  53,  56,  56,  59,  65,  76,  80,  80,
            81,  84,  84,  86,  89,  92,  92,  96,  99])




```python
# Сортировка "на месте"

a.sort(axis=0)

a
```




    array([[-99, -95, -87, -97],
           [-85, -93, -65, -66],
           [-74, -92, -57, -63],
           [-68, -84, -53, -62],
           [-46, -82, -52, -56],
           [-34, -81, -48, -44],
           [-23, -73, -47, -32],
           [-21, -72, -37, -31],
           [-16, -68, -32, -21],
           [  6, -56, -27, -13],
           [ 14, -51, -26,  -9],
           [ 25, -44, -22,  -7],
           [ 28, -26, -17,   8],
           [ 32, -13,  -5,  11],
           [ 35, -11,  -4,  16],
           [ 38,  -8,  10,  26],
           [ 44,   2,  30,  30],
           [ 53,  12,  37,  34],
           [ 65,  16,  38,  36],
           [ 76,  17,  41,  42],
           [ 80,  28,  42,  47],
           [ 84,  39,  53,  56],
           [ 84,  49,  92,  59],
           [ 89,  56,  92,  80],
           [ 96,  86,  99,  81]])




```python
a.sort(axis=1)

a
```




    array([[-99, -97, -95, -87],
           [-93, -85, -66, -65],
           [-92, -74, -63, -57],
           [-84, -68, -62, -53],
           [-82, -56, -52, -46],
           [-81, -48, -44, -34],
           [-73, -47, -32, -23],
           [-72, -37, -31, -21],
           [-68, -32, -21, -16],
           [-56, -27, -13,   6],
           [-51, -26,  -9,  14],
           [-44, -22,  -7,  25],
           [-26, -17,   8,  28],
           [-13,  -5,  11,  32],
           [-11,  -4,  16,  35],
           [ -8,  10,  26,  38],
           [  2,  30,  30,  44],
           [ 12,  34,  37,  53],
           [ 16,  36,  38,  65],
           [ 17,  41,  42,  76],
           [ 28,  42,  47,  80],
           [ 39,  53,  56,  84],
           [ 49,  59,  84,  92],
           [ 56,  80,  89,  92],
           [ 81,  86,  96,  99]])



### **Уникальные элементы**


```python
a = np.random.randint(10, size=15)

a
```




    array([8, 3, 3, 6, 6, 7, 6, 8, 4, 3, 8, 2, 9, 9, 7])




```python
a.size
```




    15




```python
np.unique(a)
```




    array([2, 3, 4, 6, 7, 8, 9])




```python
np.unique(a).size
```




    7



### **Объединение массивов**


```python
a = np.random.randint(10, size=(3, 4))
b = np.random.randint(10, size=(3, 4))
```


```python
a
```




    array([[3, 9, 3, 6],
           [9, 2, 1, 2],
           [6, 4, 1, 5]])




```python
b
```




    array([[5, 6, 0, 1],
           [4, 1, 5, 4],
           [8, 9, 2, 4]])




```python
np.vstack((a, b))
```




    array([[3, 9, 3, 6],
           [9, 2, 1, 2],
           [6, 4, 1, 5],
           [5, 6, 0, 1],
           [4, 1, 5, 4],
           [8, 9, 2, 4]])




```python
np.concatenate((a, b), axis=0)
```




    array([[3, 9, 3, 6],
           [9, 2, 1, 2],
           [6, 4, 1, 5],
           [5, 6, 0, 1],
           [4, 1, 5, 4],
           [8, 9, 2, 4]])




```python
np.hstack((a, b))
```




    array([[3, 9, 3, 6, 5, 6, 0, 1],
           [9, 2, 1, 2, 4, 1, 5, 4],
           [6, 4, 1, 5, 8, 9, 2, 4]])




```python
np.concatenate((a, b), axis=1)
```




    array([[3, 9, 3, 6, 5, 6, 0, 1],
           [9, 2, 1, 2, 4, 1, 5, 4],
           [6, 4, 1, 5, 8, 9, 2, 4]])



### **Разбиение массива**


```python
a = np.random.randint(10, size=(5, 4))

a
```




    array([[3, 3, 6, 8],
           [2, 0, 0, 5],
           [7, 1, 8, 0],
           [0, 4, 4, 5],
           [1, 1, 2, 6]])




```python
np.vsplit(a, 5)
```




    [array([[3, 3, 6, 8]]),
     array([[2, 0, 0, 5]]),
     array([[7, 1, 8, 0]]),
     array([[0, 4, 4, 5]]),
     array([[1, 1, 2, 6]])]




```python
np.hsplit(a, 4)
```




    [array([[3],
            [2],
            [7],
            [0],
            [1]]),
     array([[3],
            [0],
            [1],
            [4],
            [1]]),
     array([[6],
            [0],
            [8],
            [4],
            [2]]),
     array([[8],
            [5],
            [0],
            [5],
            [6]])]




```python

```
