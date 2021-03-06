# [⬅Глубокое обучение 2022.1](../index.html)

## Модель нейронной сети

[TOC]

![](.\images\network.png)

Под термином **сеть** понимается искусственная нейронная сеть, представляющая собой **стек взаимосвязанных слоёв**, где структурно-функциональной **единицей является узел или нейрон**. Исходные данные подаются на вход сети, а на её выходе мы получаем преобразованные данные. Каждый слой выполняет определенную математическую операцию над пропускаемыми через него данными и характеризуется набором переменных, изменение которых определяет точное поведение слоя. Здесь термин данные означает тензор — вектор, имеющий несколько измерений. Так тензоры нулевого ранга — скаляры, тензоры первого ранга — векторы, тензоры второго ранга — матрицы, тензоры третьего — матрицы в массиве, так называемый числовой куб и так далее для тензоров высших рангов.

> В глубоком обучении чаще всего используются тензоры от нулевого ранга до четырехмерных, но иногда, например при обработке видеоданных, дело может дойти  до пятимерных тензоров.

![](.\images\EOWNQ_iXsAEQq5x.jpg)

С точки зрения структурно-функциональной единицы слоя, т.е. **нейрона**, он **принимает один или несколько входов**, **умножает каждый вход на** параметр, так называемый **вес** (**weight**), **суммирует** взвешенные входные значения между собой с некоторыми значением **смещения** (***bias***), а затем **передает значение в активационную функцию**. Этот результат затем направляется дальше, к другим нейронам в следующих слоях, расположенным глубже в нейронной сети, если они существуют.

![](.\images\perceptron.png)

Как видно из схемы одной из первых моделей нейрона (перцептрон) — это линейная модель бинарной классификации.



### Перцептрон

В конце 1950-х годов американский нейробиолог Фрэнк Розенблатт (*Frank Rosenblatt*) опубликовал статью о алгоритме моделирующем, в его понимании биологические нейроны, названных в последствии  **линейным перцептроном** или просто **перцептроном** (в некоторых русских источниках, можно также встретить вариант: **персептрон**), что можно считать одной из первых реализаций искусственного нейрона. 

> Предшественником перцептрона был блок пороговой логики (БПЛ), разработанный Маккалоком и Питтсом в 1943 году и способный обучаться логическим функциям И и ИЛИ. Алгоритм обучения перцептрона относится к алгоритмам обучения с учителем. В основе идеи как БПЛ, так и перцептрона лежал биологический нейрон. 
>
> Также Маккалок и Питтс ввели в обиход важную идею анализа нейронной активности, основанную на пороговых значениях и взвешенных суммах. Эти идеи легли в основу разработки модели для более поздних вариантов, в т. ч. перцептрона.

Подобно своему биологическому аналогу, перцептрон может:

- *Принимать информацию* от множества других нейронов.
- *Агрегировать эту информацию* с использованием простой арифметической операции, которая называется взвешенной суммой.
- *Генерировать выходной сигнал*, если взвешенная сумма превысит пороговое значение, который затем может быть отправлен многим другим нейронам в сети.

По сути своей перцептрон Розенблатта — это линейная модель классификации, в которой он принимает и передаёт только двоичную информацию,  т.е. выполняет самую простую бинарную классификацию, когда все объекты в тренировочной выборке помечены одной из двух меток, и задача состоит в том, чтобы научиться расставлять эти метки у новых, ранее не неизвестных объектов. А «линейная модель» означает, что в результате обучения модель разделит все пространство входов на две части **гиперплоскостью**: правило принятия решения о том, какую метку ставить, будет **линейной функцией от входящих признаков**.

> В геометрии гиперплоскостью называется подпространство, размерность которого на единицу меньше размерности объемлющего пространства. В трехмерном случае размерность гиперплоскости равна 2, а в двумерном гиперплоскостью считается одномерная прямая.
>
> Гиперплоскость делит n-мерное пространство на две части и потому имеет полезные применения в приложениях классификации. Оптимизация параметров гиперплоскости – и есть ключевая идея линейного моделирования.

Если рассматривать перцептрон как «черный ящик» и говорить строгим математическим языком, то — это линейный бинарный классификатор со связью между входом и выходом. Внутри же можно выделить 2 этапа: в начале **вычисляется взвешенная сумму** из $n$ входов и далее **сумма передаётся ступенчатой функции** Хевисайда с заданным пороговым значением (***threshold***). Эта функция возвращает число $0$ или $1$ в зависимости от входного аргумента.

> Сразу оговоримся, что решающая граница и результат классификации функцией Хевисайда описывается следующим образом:
> $$
> \theta = 
> \left\{\begin{matrix}
> 1 :& \sum > threshold \\ 
> 0 :& \sum \leqslant threshold
> \end{matrix}\right.
> $$
>
> - Если взвешенная сумма входных значений перцептрона превышает пороговое значение, выводится 1.
> - Если взвешенная сумма входных значений перцептрона меньше или
>   равна пороговому значению, выводится 0.
>
> Ниже мы рассмотрим её подробнее.

Обозначим входные значения (каждый вход) как вектор вещественных чисел $\mathbf{x}=\vec{x}=(x_1,...,x_n) \in \mathbb{R}^n$, где $n$ — общее количество входов в перцептроне. Вектор весов $\mathbf{w}=\vec{w}=(w_1,...,w_n) \in \mathbb{R}^n$.

На этапе вычисления взвешенной сумму, перцептрон независимо взвешивает каждое из входных значений и суммирует. В общем виде **сумма всех взвешенных входных значений** или скалярное произведение (*dot product*) двух векторов длинной $n$ записывается как:
$$
\mathbf{x} \cdot \mathbf{w}=\vec{x} \cdot \vec{w}=\sum_{i=1}^{n} x_i w_i
$$
Роль сумматора в данном случае очевидна: он преобразует все входные сигналы (которых может быть очень много) в одно число - взвешенную сумму, которая характеризует поступивший на нейрон сигнал в целом.

Далее, этап **сравнения взвешенной суммы входов с порогом** перцептрона при помощи ступенчатой функции Хевисайда (**функции активации**) является выходом всего перцептрона и определяет классификацию входных значений. 

Можно упростить формулу для вычисления всего перцептрона следую­щим образом:
$$
P(\mathbf{x}) = 
\left\{\begin{matrix}
1 :& \mathbf{x} \cdot \mathbf{w} > threshold \\ 
0 :& \text{else}
\end{matrix}\right.
$$

Обратите внимание, что значение вектора весов $\mathbf{w}$ и пороговое значение функции активации (*threshold*) уже заданы в перцептроне. Они могут иметь как случайные значение, так и просто один скаляр. Далее именно на процессе обучения, эти значения и будут изменяться (обновляться), отвечая на запросы решаемой задачи и входных данных.

> Рассмотренный перцептрон чаще всего называют однослойным, чтобы отличить его от «многослойного перцептрона Румельхарта», изобретенного позже.

Рассмотрим **программную реализация перцептрона** на примере, типичной задачи, в которой нужно про­анализировать сочетание множества факторов и на основе этого анализа принять итоговое решение. Для простоты рассмотрим всего четыре фактора, влияющих на активность рыб, а значит, на успех рыбалки. Так на входы в перцептрон подадаются следующие исходные данные:

$x_1$ - скорость ветра
$x_2$ - атмосферное давление
$x_3$ - яркость солнца
$x_4$ - перепад температуры воды

Если у перцептрона есть четыре входа, то должны быть и четыре весовых коэффициен­та. В нашем примере весовые коэффициенты можно представить как показатели важности каждого входа, влияющие на общее решение нейрона. Чем больше значение коэффициента, тем больше важность входного параметра. Веса входов распре­делим следующим образом:

$w_1 = 5$
$w_2 = 4$
$w_3 = 1$
$w_4 = 1$

Пусть на входы нашего перцептрона подаются следующие сигналы (ветер- умерен­ный, атмосферное давление - высокое, яркость солнца - пасмурно, температура воды - стабильная):

$x_1 = 1$ - ветер (умеренный)
$x_2 = 0$ - атмосферное давление (высокое)
$x_3 = 0$ - яркость солнца (солнечно)
$x_4 = 1$ - перепад температуры воды (нет)

В результате поступлении этой информации в сумматор он выдаст следующую итоговую сумму:
$$
S = x_1 w_1 + x_2 w_2 + x_3 w_3 + x_4 w_4 = 
\\
=1 \cdot 5 + 0 \cdot 4 + 0 \cdot 1 + 1 \cdot 1 = 6
$$

```python
import numpy as np

def activation(x):
    b = 5
    if x >= b:
        return 1
    else:
        return 0
        
class Perceptron:
    def __init__(self, w):
    	self.w = w
    
    # Взвешивание
    def weighing(self, х) :
        # умножаем веса на входы и суммируем их
    	s = np.dot(self.w, х)
    	return s

Xi = np.array([1, 0, 0, 1]) # Задание значений входам
Wi = np.array([5, 4, 1, 1]) # Веса входных сенсоров

# создадим и инициализируем весами объект перцептрона
p = Perceptron(Wi)

print("Sum =  ", p.weighing(Xi)) # 6
print("Идти на рыдалку? (да - 1, нет - 0): ", activation(p.weighing(Xi)))
```

ДОБАВИТЬ ПРО ЛИНЕЙНОСТЬ ПЕРЦЕПТРОНА И КРИТИКУ

Стоит отметить, что один перцептрон Розенблатта может обучиться разделять только те множества точек, между которыми можно провести гиперплоскость (такие множества логично называются линейно разделимыми). Например, одинокий линейный перцептрон никогда не обучится реализовывать функцию XOR: множество ее нулей и множество ее единиц, линейно неразделимы. Что является серьезным ограничением для использования  в системах глубокого обучения данной модели искусственного нейрона — перцептрона.



### Функции активации

В общем смысле **функция активации** (**activation function**) — управляет поведением нейрона, преобразуя линейные комбинации сумм входов и весов, и порога (смещения). Многие (но не все) нелинейные преобразования, применяемые в нейронных сетях, приводят данные к  диапазону значений, например от 0 до 1 или от –1 до 1. Если нейрон передает ненулевое значение следующему нейрону, то говорят, что он активируется, или возбуждается.

> Значения, передаваемые от предыдущего слоя следующему. Это результаты, вычисленные функциями активации всех нейронов предыдущего слоя.

Общую запись функции активации, можно представить следующим образом:
$$
output = 
\left\{\begin{matrix}
min :& \sum_i x_i w_i < threshold \\ 
max :& \sum_i x_i w_i \geqslant threshold
\end{matrix}\right.
$$
Для упрощения, внесем небольшую корректировку в приведенную выше формулу. Переместим порог на другую сторону неравенства и заменим его на то, что известно как смещение нейрона. В результате, *смещение = - порог.*

> Смещение (***bias***) – это скалярная величина, прибавляемая к входному сигналу, чтобы хотя бы несколько нейронов в каждом слое активировались вне зависимости от силы сигнала. Смещения позволяют продолжить обучение, заставляя сеть реагировать даже на слабый сигнал. Благодаря им сеть пробует новые интерпретации, или виды поведения. Обычно смещения обозначаются буквой $b$ и, как и веса, модифицируются в процессе обучения.
>
> Элементы, которые вычисляют $\mathbf{x} \cdot \mathbf{w} + b$, называют линейными блоками (*linear unit*).

$$
output = 
\left\{\begin{matrix}
min :& \sum_i x_i w_i + bias < 0 \\ 
max :& \sum_i x_i w_i + bias \geqslant 0
\end{matrix}\right.
$$
Поскольку значения весов и порога изначально расставляются случайным образом, то данное преобразование делается для того, чтобы при подборе «правильных» весов и порога для нашей сети, вносить изменения приходилось только в левую часть уравнения, в то время как правая часть остаётся постоянной и равняется нулю.

Ниже приведены только основные функции активации. На деле их достаточно много и все остальные на практике подгоняются под очень тонкие предметные области и задачи. Одной из самых популярных функций активации является сигмоида и линейная ректификация (*ReLU*), в то время как функции Хевисайда и Гистерезиса больше применяются в обучающих целях.

![](.\images\f_act.png)

![](.\images\6f_act.png)

Влияние смещения и параметра наклона на функции активации: https://www.geogebra.org/classic/utrd9nnc

> Что же выбрать и как оценить, какая функция активации лучше подходит для конкретной задачи?
>
> Во время разработки реальной системы глубокого обучения большинство сетей будет использовать одну из двух функций активации: либо логистический сигмоид $\sigma$ (для за­дач аппроксимации и классификации), либо линейная ректификация *ReLU*. Из них, особенно *ReLU*, и рекомендуется начинать разработку, а потом уже, можно попробовать параметризованные *ReLU* и другие современные идеи: они могут дать некоторое улучшение качества, но оно, вероятнее всего, будет маргинальным, и эту оптимизацию лучше отложить на потом.
>
> Подбор функции активации это далеко не первый и не главный вопрос в разработке систем глубокого обучения, куда важнее выбрать правильную  архитектуру системы и алгоритмы оптимизации.

![](.\images\uxrxmtrbzdniqbytle0ps2_jhbs.png)

Далее в процессе обучения нейронной сети, потребуется вычисление производных функций активации и поэтому это также является одной из характеристик функций активации (свойство дифференцируемости функции активации).



#### Пороговая функция или функция Хевисайда

![](.\images\0-92154-168992.png)

**Пороговая (ступенчатая) функци**я или **функция Хевисайда** или **функции единичного скачка** работает по принципу: если входное значение меньше порогового, то значение функции активации равно минимальному допустимому, иначе — максимально допустимому.

Математическая модель: 
$$
step(x) = 
\left\{\begin{matrix}
0 :& x < threshold \\ 
1 :& x \geqslant threshold
\end{matrix}\right.
$$
или
$$
step(x) = \left\{x + bias < 0\ :\ 0,\ x + bias \ge0\ :\ 1\right\}
$$


```python
def Heaviside(x):
    if x >= 0.5:
        return 1
    else:
    	return 0
```



#### Функция Гистерезис или линейный порог

![](.\images\jRF1UC2-T6I.jpg)

**Кусочно-линейная функция** имеет два линейных участка, где функция активации тождественно равна минимально допустимому и максимально допустимому значению и есть участок, на котором функция строго монотонно возрастает.
$$
linear(x) =
\left\{\begin{matrix}
0 :& x < threshold - n \\ 
x :& x \in [threshold - n; threshold + n)\\
1 :& x \geqslant threshold + n
\end{matrix}\right.
$$

```python
def Linear_foo(x):
	f = math.pow(x, 2) / 2
    return f
```



#### Сигмоид или логистическая функция

![](.\images\sigmoid.png)

Монотонно возрастающая всюду дифференцируемая *S*-образная нелинейная функция с насыщением. Сигмоид позволяет усиливать слабые сигналы и не насыщаться от сильных сигналов (решает проблему шумового насыщения).

> Усиление слабых сигналов и предотвращение насыще­ние от больших сигналов, так как они соответствуют областям ар­гументов, где сигмоид имеет пологий наклон. Таким образом одна и таже сеть может обрабатывать как слабые, так и сильные сигналы.

Примером сигмоидальной функции активации может служить **логистическая функция**, задаваемая в общем виде следующим выражением:
$$
sigmoid(x) = \sigma(x) = \frac{1}{1 + e^{-x}}
$$

или с учетом отдельного гиперпараметра $\alpha$ - параметра наклона сигмоидальной функции активации:
$$
\frac{1}{1 + e^{-x \cdot \alpha}}
$$


```python
def Sigmoid_foo(x):
    if x >= 0.5:
        z = np.exp(-x)
        return 1 / (1 + z)
    else: 
        z = np.exp(x)
        return z / (1 + z)
```



#### Гиперболический тангенс (*tanh*)

![](.\images\tanh.png)

Еще одним примером сигмоидальной функции активации является **гиперболический тангенс**, задаваемый следующим выражением:
$$
\tanh(x) = \frac{\sinh(x)}{\cosh(x)} = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{e^{2x} - 1}{e^{2x} + 1}
$$

или с параметром наклона:
$$
\tanh(\frac{x}{\alpha})
$$

Гиперболическая касательная функция достигает своего максимума, когда $x → +∞$, что соответствует $+1$, и минимума, когда $x → -∞$, что соответствует $-1$.

```python
def Hyperbolistic_Th(x):
    Tn = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
    return Th
```

> Гиперболический тангенс работает лучше, чем сигмовидная функция, когда применяется как функция активации в скрытом слое, потому что среднее значение активаций будет 0, в отличие от ситуации, когда мы используем сигмовидную функцию, где среднее значение будет 0,5. Обоснование здесь аналогично - точно так же, как мы центрируем входные данные вокруг среднего значения, чтобы градиентный спуск легко сходился, если у активаций также есть среднее по нулю значение, тогда градиентный спуск, в общем, работает хорошо. 
>
> Единственное место, где сигмоид может быть полезен, это когда мы выполняем задачу двоичной классификации. В этой ситуации выходной ярлык должен быть 0 или 1. Если бы мы использовали $tanh$ функцию для выходного слоя, тогда прогноз будет лежать между -1 и 1 вместо 0 и 1. Поэтому в этом сценарии лучше использовать сигмовидную функцию только в последнем слое.



#### Линейная ректификация (*ReLU*)

![](.\images\relu.png)

**Линейная ректификация** (*rectified linear units - **ReLU***) или **спрямленная линейная единичная функция** задаётся следующим образом:
$$
relu(x) =
\left\{\begin{matrix}
0 :& x < 0 \\ 
x :& x \geqslant 0
\end{matrix}\right.
$$

```python
def ReLU(x):
    return x * (x > 0)
```

Время работы разных вариантов реализаций *ReLU*:

```python
import numpy as np

x = np.random.random((5000, 5000)) - 0.5

print("max method:")
%timeit -n10 np.maximum(x, 0) # 10 loops, best of 3: 239 ms per loop

print("multiplication method:") 
%timeit -n10 x * (x > 0) # 10 loops, best of 3: 145 ms per loop

print("abs method:")
%timeit -n10 (abs(x) + x) / 2 # 10 loops, best of 3: 288 ms per loop
```

Как видно из результатов, реализация построенная на умножении, даёт самый быстрый результат.

> Преимущество функции *ReLU* заключается в том, что по большей части  для нескольких значений производная далека от нуля, по сравнению с сигмоидной или $tanh$ функцией, где они страдают от исчезающих градиентов.
>
> Следовательно, алгоритм градиентного спуска сходится быстрее, так как градиенты далеки от нуля (поэтому обучение не замедляется, как у сигмоидной или $tanh$ функцией).




### Общая модель нейрона и нейронной сети

![](.\images\neuron.png)

Модель (искусственного) нейрона в общем виде можно представить как:
$$
neuron_i = activation\_function \left( \sum_{j=1}^{n} w_{ij} x_j + b_i \right)
$$
В качестве функции активации может выступать одна из представленных выше. 





Вектор входных значений $\mathbf{x}$ умноженный на матрицу весов $\mathrm{W}$ даёт вектор выходных значений $\mathbf{y}$:
$$
\mathbf{y} = \mathbf{x} \cdot \mathrm{W}
$$
Поскольку умножение матриц — линейная операция, соответственно сеть, содержащая лишь операции матричного умножения, может обучаться только линейным отображением. Для придания сети большей гибкости, результат матричного умножения подвергается нелинейному преобразованию, при помощи функций активации. Её может выступать любая дифференцируемая функция.



Для создания нейронной сети, нужно выбрать тип соединения нейронов, опреде­лить вид передаточных функций элементов и подобрать весовые коэффициенты межнейронных связей.



### Типы нейронных сетей

Нейронная сеть представляет собой совокупность узлов (нейроподобных элементов), определенным образом соединенных друг с другом и с внешней средой с помощью связей, определяемых весовыми коэффициентами. В зависимости от функций, выполняемых нейронами в сети, можно выделить три **типа нейронов**: входные нейроны, выходные нейроны и промежуточные нейроны.

С точки зрения топологии можно выделить три основных **типа нейронных сетей:** полносвязные сети, многослойные сети с последовательными связями, и слабосвязные сети.





#### Полносвязные сети

Полносвязные сети (многослойный перцептрон) были первыми из исследованных типов сетей и оставались актуальными вплоть до конца 1980-х годов. В полносвязной сети выход каждого нейрона рассчитывается как взвешенная сумма всех его входов. Именно этим поведением слоя обусловлено происхождение термина полносвязный (*fully connected*): в полносвязном слое каждый из его нейронов соединен со всеми элементами предыдущего слоя. 





#### Сверточные сети

![](.\images\CNN.png)





#### Рекуррентные сети

#### Состязательные сети и автокодировщики





### Выводы

Существует множество способов построения сетей. Конкретный выбор, определяется стоящими перед сетью задачами. Проектирование новых типов сетей относится к области научных исследований, и даже реализация сети, архитектура которой описана в литературе, — тяжелая задача. На практике проще всего взять готовый образец, делающий нечто концептуально схожее, и постепенно, шаг за шагом, вносить в него изменения, пока не получится то, что вам нужно.



### Тест





---

### Источники





### Дополнительные материалы

