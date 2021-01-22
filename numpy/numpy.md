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

    147 ms ± 17.5 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
a = np.arange(1000000)
```


```python
%%timeit

a * a
```

    5.05 ms ± 644 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


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




```python
np.resize(a, (2, 2))
```




    array([[1, 2],
           [3, 4]])



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
np.full((5, 3), 7)
```




    array([[7, 7, 7],
           [7, 7, 7],
           [7, 7, 7],
           [7, 7, 7],
           [7, 7, 7]])




```python
np.empty((3, 2))
```




    array([[1., 1.],
           [1., 1.],
           [1., 1.]])




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

a
```




    array([  1,   3,   6, 100,   2,   1])




```python
a = np.delete(a, 2)

a
```




    array([  1,   3, 100,   2,   1])




```python
a = np.random.randint(12, size=(4, 3))

a
```




    array([[11,  2,  3],
           [ 6,  1,  9],
           [ 7,  6,  3],
           [ 0,  7,  5]])




```python
a = np.insert(a, 2, [100, 100, 100], axis=0)

a
```




    array([[ 11,   2,   3],
           [  6,   1,   9],
           [100, 100, 100],
           [  7,   6,   3],
           [  0,   7,   5]])




```python
a = np.insert(a, 0, [200, 200, 200, 200, 200], axis=1)

a
```




    array([[200,  11,   2,   3],
           [200,   6,   1,   9],
           [200, 100, 100, 100],
           [200,   7,   6,   3],
           [200,   0,   7,   5]])




```python
a = np.random.randint(12, size=(4, 3))

a
```




    array([[ 1,  3,  4],
           [11,  5,  4],
           [ 8,  1,  6],
           [ 8,  3,  6]])




```python
a = np.delete(a, 3, axis=0)

a
```




    array([[ 1,  3,  4],
           [11,  5,  4],
           [ 8,  1,  6]])




```python
a = np.delete(a, 1, axis=1)

a
```




    array([[ 1,  4],
           [11,  4],
           [ 8,  6]])



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




    array([[ 1, -3, -7, -1,  4, -9, -5,  6],
           [-1, -4,  6, -1, -9, -2, -5,  9],
           [ 2,  2, -2,  3, -7,  0,  3, -3],
           [-1,  5,  6, -2, -3,  8, -3,  5]])



![Image](https://numpy.org/doc/stable/_images/np_matrix_aggregation_row.png)


```python
a.max(axis=0)
```




    array([2, 5, 6, 3, 4, 8, 3, 9])




```python
a.max(axis=1)
```




    array([6, 9, 3, 8])



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




    array([[ 8,  6, -3, -8],
           [ 1, -2,  5,  0],
           [ 1, -9,  9, -3]])




```python
b = np.random.randint(-9, 10, size=(3, 4))

b
```




    array([[ 4, -8,  4, -2],
           [ 8, -5,  9,  8],
           [ 3, -3, -3, -7]])




```python
a + b
```




    array([[ 12,  -2,   1, -10],
           [  9,  -7,  14,   8],
           [  4, -12,   6, -10]])




```python
a * b
```




    array([[ 32, -48, -12,  16],
           [  8,  10,  45,   0],
           [  3,  27, -27,  21]])




```python
a > b
```




    array([[ True,  True, False, False],
           [False,  True, False, False],
           [False, False,  True,  True]])




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



### **Линейная алгебра**


```python
a = np.random.randint(10, size=(3, 3))

a
```




    array([[0, 4, 4],
           [9, 2, 4],
           [1, 5, 8]])




```python
b = np.linalg.inv(a)

b
```




    array([[ 0.04,  0.12, -0.08],
           [ 0.68,  0.04, -0.36],
           [-0.43, -0.04,  0.36]])




```python
a.dot(b)
```




    array([[ 1.00000000e+00, -2.77555756e-17,  2.22044605e-16],
           [ 0.00000000e+00,  1.00000000e+00,  2.22044605e-16],
           [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])




```python
np.trace(a)
```




    10




```python
np.linalg.det(a)
```




    -100.00000000000004




```python
np.linalg.matrix_rank(a)
```




    3




```python
np.linalg.eig(a)[0]
```




    array([-4.54061975,  1.71747453, 12.82314523])




```python
np.linalg.eig(a)[1]
```




    array([[ 0.47323046, -0.25538821,  0.40249756],
           [-0.83062984, -0.73628389,  0.59257077],
           [ 0.29343994,  0.6266282 ,  0.69775038]])



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




    array([-7,  8, -3, -5, -8,  8,  1, -7,  3, -6, -9,  4,  6, -8, -4])




```python
a > 3
```




    array([False,  True, False, False, False,  True, False, False, False,
           False, False,  True,  True, False, False])




```python
a[a > 3]
```




    array([8, 8, 4, 6])




```python
a % 2 == 0
```




    array([False,  True, False, False,  True,  True, False, False, False,
            True, False,  True,  True,  True,  True])




```python
a[a % 2 == 0]
```




    array([ 8, -8,  8, -6,  4,  6, -8, -4])




```python
a[np.logical_or(a % 2 == 0, a % 3 == 0)]
```




    array([ 8, -3, -8,  8,  3, -6, -9,  4,  6, -8, -4])




```python
a[np.logical_and(a % 2 == 0, a % 3 == 0)]
```




    array([-6,  6])



### **Индексация в многомерных массивах**
![Image](https://numpy.org/doc/stable/_images/np_matrix_indexing.png)


```python
a = np.random.randint(-9, 10, (5, 5))

a
```




    array([[ 3,  4,  1,  3, -7],
           [ 6,  4,  1,  3,  9],
           [ 5, -2,  0, -5, -3],
           [ 7, -9, -4,  1, -4],
           [ 0, -6,  9,  5, -3]])




```python
# Способ хорош для стандартный Python-списков, но не для NumPy-массивов

a[2][0]
```




    5




```python
a[2, 0]
```




    5




```python
a[3, :]
```




    array([ 7, -9, -4,  1, -4])




```python
a[:, 2]
```




    array([ 1,  1,  0, -4,  9])




```python
a[:, ::2]
```




    array([[ 3,  1, -7],
           [ 6,  1,  9],
           [ 5,  0, -3],
           [ 7, -4, -4],
           [ 0,  9, -3]])




```python
a
```




    array([[ 3,  4,  1,  3, -7],
           [ 6,  4,  1,  3,  9],
           [ 5, -2,  0, -5, -3],
           [ 7, -9, -4,  1, -4],
           [ 0, -6,  9,  5, -3]])




```python
a < 0
```




    array([[False, False, False, False,  True],
           [False, False, False, False, False],
           [False,  True, False,  True,  True],
           [False,  True,  True, False,  True],
           [False,  True, False, False,  True]])




```python
a[a < 0]
```




    array([-7, -2, -5, -3, -9, -4, -4, -6, -3])




```python
np.where(a < 0)
```




    (array([0, 2, 2, 2, 3, 3, 3, 4, 4]), array([4, 1, 3, 4, 1, 2, 4, 1, 4]))




```python
a[np.where(a < 0)]
```




    array([-7, -2, -5, -3, -9, -4, -4, -6, -3])




```python
a
```




    array([[ 3,  4,  1,  3, -7],
           [ 6,  4,  1,  3,  9],
           [ 5, -2,  0, -5, -3],
           [ 7, -9, -4,  1, -4],
           [ 0, -6,  9,  5, -3]])




```python
a[[3, 4, 1]]
```




    array([[ 7, -9, -4,  1, -4],
           [ 0, -6,  9,  5, -3],
           [ 6,  4,  1,  3,  9]])




```python
a[:, [2, 4, 0]]
```




    array([[ 1, -7,  3],
           [ 1,  9,  6],
           [ 0, -3,  5],
           [-4, -4,  7],
           [ 9, -3,  0]])




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




    array([[-32, -13,  44, -44],
           [ 37,  30,  96,  56],
           [ 92, -63,  14,  28],
           [-48, -21,  65,  16],
           [ 38,  -7,  32, -95],
           [-22, -56,  28,  -8],
           [-17,  59, -34, -11],
           [ 42, -66,   6,  12],
           [-52,  -9,  53, -82],
           [-53, -44,  84, -84],
           [-65,   8, -46, -56],
           [-27,  16, -85, -13],
           [ 99,  26, -74, -26],
           [ -4,  36, -70, -27],
           [-80, -96,  83, -45],
           [-60,   9, -21,  37],
           [-20, -16,  53,  -6],
           [-39, -24,  86,  38],
           [-76, -71, -64, -70],
           [-29,  70, -36,  -1],
           [ 14, -61,  33, -34],
           [ 18, -94,  83, -50],
           [-57,  33, -95,  37],
           [ 86,  -9, -71,  63],
           [-79,  25, -55,  48]])




```python
np.sort(a)
```




    array([[-44, -32, -13,  44],
           [ 30,  37,  56,  96],
           [-63,  14,  28,  92],
           [-48, -21,  16,  65],
           [-95,  -7,  32,  38],
           [-56, -22,  -8,  28],
           [-34, -17, -11,  59],
           [-66,   6,  12,  42],
           [-82, -52,  -9,  53],
           [-84, -53, -44,  84],
           [-65, -56, -46,   8],
           [-85, -27, -13,  16],
           [-74, -26,  26,  99],
           [-70, -27,  -4,  36],
           [-96, -80, -45,  83],
           [-60, -21,   9,  37],
           [-20, -16,  -6,  53],
           [-39, -24,  38,  86],
           [-76, -71, -70, -64],
           [-36, -29,  -1,  70],
           [-61, -34,  14,  33],
           [-94, -50,  18,  83],
           [-95, -57,  33,  37],
           [-71,  -9,  63,  86],
           [-79, -55,  25,  48]])




```python
# Возвращает копию

np.sort(a.ravel())
```




    array([-96, -95, -95, -94, -85, -84, -82, -80, -79, -76, -74, -71, -71,
           -70, -70, -66, -65, -64, -63, -61, -60, -57, -56, -56, -55, -53,
           -52, -50, -48, -46, -45, -44, -44, -39, -36, -34, -34, -32, -29,
           -27, -27, -26, -24, -22, -21, -21, -20, -17, -16, -13, -13, -11,
            -9,  -9,  -8,  -7,  -6,  -4,  -1,   6,   8,   9,  12,  14,  14,
            16,  16,  18,  25,  26,  28,  28,  30,  32,  33,  33,  36,  37,
            37,  37,  38,  38,  42,  44,  48,  53,  53,  56,  59,  63,  65,
            70,  83,  83,  84,  86,  86,  92,  96,  99])




```python
# Сортировка "на месте"

a.sort(axis=0)

a
```




    array([[-80, -96, -95, -95],
           [-79, -94, -85, -84],
           [-76, -71, -74, -82],
           [-65, -66, -71, -70],
           [-60, -63, -70, -56],
           [-57, -61, -64, -50],
           [-53, -56, -55, -45],
           [-52, -44, -46, -44],
           [-48, -24, -36, -34],
           [-39, -21, -34, -27],
           [-32, -16, -21, -26],
           [-29, -13,   6, -13],
           [-27,  -9,  14, -11],
           [-22,  -9,  28,  -8],
           [-20,  -7,  32,  -6],
           [-17,   8,  33,  -1],
           [ -4,   9,  44,  12],
           [ 14,  16,  53,  16],
           [ 18,  25,  53,  28],
           [ 37,  26,  65,  37],
           [ 38,  30,  83,  37],
           [ 42,  33,  83,  38],
           [ 86,  36,  84,  48],
           [ 92,  59,  86,  56],
           [ 99,  70,  96,  63]])




```python
a.sort(axis=1)

a
```




    array([[-96, -95, -95, -80],
           [-94, -85, -84, -79],
           [-82, -76, -74, -71],
           [-71, -70, -66, -65],
           [-70, -63, -60, -56],
           [-64, -61, -57, -50],
           [-56, -55, -53, -45],
           [-52, -46, -44, -44],
           [-48, -36, -34, -24],
           [-39, -34, -27, -21],
           [-32, -26, -21, -16],
           [-29, -13, -13,   6],
           [-27, -11,  -9,  14],
           [-22,  -9,  -8,  28],
           [-20,  -7,  -6,  32],
           [-17,  -1,   8,  33],
           [ -4,   9,  12,  44],
           [ 14,  16,  16,  53],
           [ 18,  25,  28,  53],
           [ 26,  37,  37,  65],
           [ 30,  37,  38,  83],
           [ 33,  38,  42,  83],
           [ 36,  48,  84,  86],
           [ 56,  59,  86,  92],
           [ 63,  70,  96,  99]])



### **Уникальные элементы**


```python
a = np.random.randint(10, size=15)

a
```




    array([3, 6, 8, 2, 0, 0, 5, 7, 1, 8, 0, 0, 4, 4, 5])




```python
a.size
```




    15




```python
np.unique(a)
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8])




```python
np.unique(a).size
```




    9



### **Объединение массивов**


```python
a = np.random.randint(10, size=(3, 4))
b = np.random.randint(10, size=(3, 4))
```


```python
a
```




    array([[1, 1, 2, 6],
           [0, 3, 4, 3],
           [9, 4, 4, 0]])




```python
b
```




    array([[3, 6, 2, 7],
           [5, 2, 1, 0],
           [0, 2, 8, 0]])




```python
np.vstack((a, b))
```




    array([[1, 1, 2, 6],
           [0, 3, 4, 3],
           [9, 4, 4, 0],
           [3, 6, 2, 7],
           [5, 2, 1, 0],
           [0, 2, 8, 0]])




```python
np.concatenate((a, b), axis=0)
```




    array([[1, 1, 2, 6],
           [0, 3, 4, 3],
           [9, 4, 4, 0],
           [3, 6, 2, 7],
           [5, 2, 1, 0],
           [0, 2, 8, 0]])




```python
np.hstack((a, b))
```




    array([[1, 1, 2, 6, 3, 6, 2, 7],
           [0, 3, 4, 3, 5, 2, 1, 0],
           [9, 4, 4, 0, 0, 2, 8, 0]])




```python
np.concatenate((a, b), axis=1)
```




    array([[1, 1, 2, 6, 3, 6, 2, 7],
           [0, 3, 4, 3, 5, 2, 1, 0],
           [9, 4, 4, 0, 0, 2, 8, 0]])



### **Разбиение массива**


```python
a = np.random.randint(10, size=(5, 4))

a
```




    array([[2, 6, 3, 6],
           [5, 4, 8, 0],
           [6, 9, 0, 6],
           [6, 2, 4, 3],
           [6, 3, 5, 5]])




```python
np.vsplit(a, 5)
```




    [array([[2, 6, 3, 6]]),
     array([[5, 4, 8, 0]]),
     array([[6, 9, 0, 6]]),
     array([[6, 2, 4, 3]]),
     array([[6, 3, 5, 5]])]




```python
np.hsplit(a, 4)
```




    [array([[2],
            [5],
            [6],
            [6],
            [6]]),
     array([[6],
            [4],
            [9],
            [2],
            [3]]),
     array([[3],
            [8],
            [0],
            [4],
            [5]]),
     array([[6],
            [0],
            [6],
            [3],
            [5]])]


