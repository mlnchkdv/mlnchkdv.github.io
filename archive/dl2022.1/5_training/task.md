# [⬅Глубокое обучение 2022.1](../index.html)

## Создание модели нейронной сети

[TOC]

В этой разделе мы создадим и обучим нейронную сеть. Для начала, создадим нейронную сеть, это можно сделать всего в несколько строк с помощью библиотеки **Keras**. Подключите основу – класс создания последовательной модели **Sequential**.

> **Важно!** Для ускорения обучения модели стоит переключиться на **GPU** в верхнем меню Colab:
>
> `Среда выполнения --> Сменить среду выполнения --> Аппаратный ускоритель`

```python
from tensorflow.keras.models import Sequential
```

С помощью него создайте экземпляр вашей модели:

```python
model = Sequential()
```

Это и есть ваша модель! Сейчас она больше похожа на пустую коробку. Чтобы она что-то делала, нужно поместить в нее какой-нибудь механизм. Это не механизм в обычном смысле слова, потому что вы будете оперировать не предметами, а информацией – главным ресурсом XXI века. Механизм будет принимать на вход и выдавать на выходе какие-то данные.

**Объекты**

Так из чего же вы можете создать механизм? Для начала определитесь, сколько информации вы будете давать нейросети на вход. Один экземпляр такой информации называется **объектом**, который **всегда состоит из чисел**.

Допустим, вы решили, что каждый ваш входной объект состоит из **10** чисел. Настройте сеть на вход из **10** чисел:

```python
from tensorflow.keras.layers import Dense
model.add(Dense(32, input_dim=10))
```

Внутри нейросеть состоит из слоев нейронов, и только что вы создали один из них. Этот первый слой называется **Dense**-слоем (линейным или полносвязным слоем). Здесь же вы указали с помощью параметра **input_dim**, что ваша сеть принимает на вход последовательность из **10** чисел.

Полносвязный слой чаще других используется в нейросетях. Как механизм делится по слоям, так и некоторые слои тоже делятся на составляющие элементы. В разных слоях они имеют разные функции и названия. Например, в линейном слое этими элементами выступают **полносвязные нейроны**.

Их количество задается самым первым аргументом (в примере: **32**).

Нужно ли создавать выход сети? На самом деле нет. Результат, который выдает последний слой, и есть выход сети.

Значит, сеть готова к работе? Еще нет, потому что для работы нужно ее еще скомпилировать (собрать, подготовить к обучению) и обучить.

Для подготовки к обучению вам понадобятся еще две вещи – **оптимизатор** и **функция потерь** (или **функция ошибки**). Они задаются с помощью метода модели `.compile()`:

```python
model.compile(loss='categorical_crossentropy', 
              optimizer='adam')
```

> При обучении нейронной сети обязательно указывают оптимизатор и функцию ошибки.



## Обучение нейронной сети

При обучении нейронные сети самостоятельно подбирают веса.

Рассмотрим простой пример: НС, которая на основании веса, роста и длины тела животного будет определять, кошка это или собака. На первом этапе веса НС задаются случайным образом. Каждый нейрон отвечает за свой признак (вес меньше четверти длины, рост меньше длины,...) и, найдя его, может дальше передавать информацию. На основании всех вычислений НС выдает результат, например: 91%, что это кошка.

![](.\images\обучение1.jpg)

Чтобы НС обучалась, ей подают на вход массив данных – обучающую выборку. Это как учебник для школьника, в котором много примеров.

Допустим, что у вас есть выборка из 20 тысяч примеров с правильными ответами. В ней есть данные о весе, росте и длине животного, информация, кошка это или собака. Соответственно, числа подаются на вход НС, а на выходе вы будете ждать ответа – кошка это или собака. Нейросеть подстроит все свои веса автоматически, чтобы идеально распознавать обучающую выборку. Но не каждая архитектура так может. Вашей задачей как раз и будет **подбор необходимой архитектуры** для достижения результата. Нейронная сеть же обучится сама.

Если веса уникальны, то она может их заучить, например, если задача начинается со слов «Было 2 землекопа», то ответ «17». Во время теста школьник увидит, что задача начинается с этих же слов, напишет в ответе «17», рискуя в итоге ошибиться. Ведь он не разобрался в вопросе, а просто заучил результаты. Такая же проблема существует и у НС. Чтобы ее проверить, создают тестовую выборку, которую нейронная сеть никогда не видела.

> Процесс обучения в любом алгоритме на основе весов заключается в пересчете весов и смещений, так что одни становятся больше, а другие меньше. В результате значимость одних частей информации увеличивается, а других – уменьшается. Это позволяет модели узнать, какие предикторы (признаки) с какими выходами связаны, и соответствующим образом подстроить веса и смещения.

![](.\images\обучение2.jpg)

Вы разобрались, из чего состоит НС и какие данные ей нужны; осталось разобраться с тем, как происходят обучение и подбор весов.



### Градиентный спуск

Вначале у НС **все веса назначаются случайно**: она, как маленький ребенок, болтает ножками и пока не понимает, что с ними делать. Нейронная сеть изначально ~~глупа~~ не понимает что вообще от неё хотят и будет невпопад выдавать случайные ответы. В нашем примере с кошкой и собакой процесс будет выглядеть как подбрасывание монеты.

Соответственно, на выходе сети вы получите некоторую долю неверных ответов - ошибку нейронной сети или функции стоимости или функции потерь. Ее можно считать множеством способов, например, в процентах.

> *Квадратичная функция стоимости* — одна из простейших с вычислительной точки зрения. Также ее называют среднеквадратичной ошибкой, что объясняется следующей формулой:
> $$
> C = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y_i})^2
> $$
>
> Для каждого $i$-го экземпляра вычисляется разность (ошибка) между
> истинной меткой $y_i$ и оценкой сети $i$, и затем эта разность возводится
> в квадрат, потому что:
>
> 1. Возведение в квадрат гарантирует, что, если $y$ больше, чем $\hat{y}$, или наоборот, разность всегда будет иметь положительное значение.
> 2. Возведение в квадрат увеличивает штраф за большую разность между $y$ и $\hat{y}$.


> *Перекрестная энтропия* — уменьшает отрицательное влияние насыщенных нейронов на скорость обучения.
> $$
> C = -\frac{1}{n} \sum_{i=1}^n [y_i \ln \hat{y_i} + (1-y_i) \ln (1 - \hat{y_i})]
> $$

Представим, что у НС заморозили все нейроны, кроме одного, и посмотрим, как изменяется ошибка сети при изменении весов одного незамороженного нейрона.

![](.\images\обучение3.jpg)

При изменении веса вы получите различную ошибку НС. Для понимания, как проходит обучение, рассмотрим график ошибки при изменении веса от -1000 до 1000.

![](.\images\обучение4.png)

Представим, что вы начинаете поиск с красной точки, в которой вес равен 0, а ошибка 62%. Сдвинулись влево (вес равен -1), и ошибка увеличилась до 62,5%. Это плохо, так как увеличение ошибки показывает, что НС стала хуже работать. Поэтому сдвиньтесь вправо (вес равен 1), и ошибка уменьшится до 61,5%. Похоже, обучение НС идет в верном направлении.

Двигайтесь вправо маленькими шажками (измененяя вес на 1), пока не найдете минимум графика – вес 120, ошибка 42%. Но, как видно из графика, первичное случайное распределение весов может и не позволить спуститься до минималього значения ошибки (старт с левого пика графика, вес -800). Так выглядит движение одного веса. А как будет выглядеть движение большего количества весов?

![](.\images\обучение5.jpg)

График ошибки в таком случае выглядит как пустыня с барханами. И веса, как шарики, движутся под действием силы тяжести, стремясь к минимуму ошибки.

> В хорошо обученной нейронной сети веса подобраны так, чтобы сигнал усиливался, а шум ослаблялся. Чем больше вес, тем сильнее корреляция между сигналом и выходом сети. Входы с большими весами влияют на интерпретацию данных сетью сильнее, чем входы с малыми весами.



В этом нейронная сеть помогает специальный **алгоритм обратного распространения ошибки**. Это измененный **метод градиентного спуска** - математические формулы, которые вычисляют, как изменить веса, чтобы быстрее прийти к минимальной ошибке (а тем самым - к наилучшей точности). К счастью, алгоритм уже заложен в библиотеке, и его не нужно придумывать и писать.

> Обучение методом обратного распространения похоже на обучение перцептрона. Мы хотим вычислить отклик на входной сигнал путем прямого распространения по сети. Если отклик совпадает с меткой, то не надо делать ничего. Если же не совпадает, то нужно подправить веса связей в сети.
>
> Идея в том, чтобы распределить штраф за ошибку между весами, внесшими вклад в выход. В случае алгоритма обучения перцептрона это легко, потому что на выходное значение влияет только один вес для каждого входа. Но в многослойных сетях прямого распространения ситуация сложнее, поскольку на пути от входов к выходам находится много весов. Каждый вес вносит вклад в несколько выходов, поэтому алгоритм обучения должен быть более изощренным.
>
> Обратное распространение – это прагматичный подход к распределению вклада в ошибку между отдельными весами. Здесь есть сходство с алгоритмом обучения перцептрона. Мы пытаемся минимизировать расхождение между меткой (истинным выходным значением), ассоциированной с данным входом, и значением, сгенерированным сетью.

Даже если в нейронной сети очень много весов, алгоритм обратного распространения ошибки регулирует их все – увеличивает или уменьшает на каждом шаге обучения.



### Оптимизаторы (Light-версия)

Оптимизатор — это метод достижения лучших результатов, помощь в ускорении обучения.

Это алгоритм, используемый для незначительного изменения параметров, таких как веса и скорость обучения, чтобы модель работала правильно и быстро.

Рассмотрим такие оптимизаторы, как:

- SGD (стохастический градиентный спуск)
- Adagrad (среднеквадратичное распространение корня)
- RMSprop (экспоненциально затухающее среднее)
- Adam



#### Импульсный оптимизатор

##### Стохастический градиентный спуск

Стохастический градиентный спуск (Stochasticу Gradient descent - SGD).

Берем случайные объекты, подаем их в модель и получаем предсказание, считаем функцию потерь, обновляем веса, повторяем снова до тех пор пока функция ошибки не окажется в точке минимума. Есть три варианта подачи объектов: по одному, весь набор данных и партиями (батчами, степенью числа 2 (32, 64, 128, ...), это наиболее распространенная реализация).

![](.\images\SGD.png)

Основная проблема - попадание функции в локальные минимумы вместо нахождения глобального. Для ее решения используются модификации стахастического градиента, использующие скользящее среднее градиентов:

- оптимизатор **Momentum**, учитывает прошлые градиенты для сглаживания обновления
- Оптимизатор **Nesterov Momentum**, его особенностью является вычисление градиента при обновлении

Пример использования оптимизатора **SGD**:

```python
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(64, input_shape=(10,), activation='softmax'))
sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```

Документация:

- SGD в Керас https://keras.io/api/optimizers/sgd/

Документация по оптимизаторам на русском:

- https://ru-keras.com/optimizer/



#### Адаптивные оптимизаторы

Цель адаптивных алгоритмов - отдельный learning rate для каждого из параметров. Чем чаще и сильнее меняется параметр, тем меньше его следующие изменения.



##### Среднеквадратичное распространение корня Adagrad

Например, какой-то из нейронов в каждой итерации немного изменяет свои значения(0,0.1,0,0.1,0.2) и есть другой, который колеблется от 0 до 10, к примеру. Эти скачки нужно сгладить. Для этого мы по каждому нейрону храним его историю, это дает нам возможность рассчитать размер следующий шаг как корень из **произведения квадратов его двух предыдущих шагов**.

Таким образом, чем больше были предыдущие шаги изменений для нейрона, тем меньше буду следующие, это помогает "успокоить" особо активные нейроны.

Но существует вероятность застрять в локальном минимуме, потому что скорость постоянно падает.

Пример использования оптимизатора **Adagrad**:

```python
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(64, input_shape=(10,), activation='softmax'))
adagrad = optimizers.Adagrad(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=adagrad)
```

Документация:

- Adagrad в Keras https://keras.io/api/optimizers/adagrad/

Документация по оптимизаторам на русском:

- https://ru-keras.com/optimizer/



##### Экспоненциально затухающее среднее значение (RMSprop, root mean square)

Существенным свойством RMSprop является то, что вы не ограничены только суммой прошлых градиентов, но вы более ограничены градиентами последних временных шагов. RMSprop вносит свой вклад в экспоненциально затухающее среднее значение прошлых «квадратичных градиентов». В RMSProp мы пытаемся уменьшить вертикальное движение, используя среднее значение.

Появляется новая зависимость размера шага от предыдущих. Уходит обязательность того, если 10 шагов назад был большой шаг, то сейчас должен быть меньший, и это позволяет выбраться из ямы. То есть он немного буксует в локальном минимуме, а потом снова начинает увеличиваться при необходимости, потому что зависимость от предыдущих шагов экспоненциальная.

Чаще всего используется в генеративных алгоритмах.

![](.\images\RMSprop.png)

Пример использования оптимизатора **RMSprop**:

```python
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(64, input_shape=(10,), activation='softmax'))
rmsprop = optimizers.RMSprop(learning_rate=0.001, rho=0.9)
model.compile(loss='mean_squared_error', optimizer=rmsprop)
```

Документация:

- RMSprop в Keras https://keras.io/api/optimizers/rmsprop/

Документация по оптимизаторам на русском:

- https://ru-keras.com/optimizer/



#### Оптимизатор Adam

Adam — один из самых эффективных алгоритмов оптимизации в обучении нейронных сетей. Он сочетает в себе идеи RMSProp и оптимизатора импульса (momentum).

Чем больше движемся в одну сторону, тем больше шаг и адаптация каждого параметра.

Вместо того чтобы адаптировать скорость обучения параметров на основе среднего первого момента (среднего значения), как в RMSProp, Adam также использует среднее значение вторых моментов градиентов. В частности, алгоритм вычисляет экспоненциальное скользящее среднее градиента и квадратичный градиент

Пример использования оптимизатора **Adam**:

```python
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(64, input_shape=(10,), activation='softmax'))
adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=adam)
```

Документация:

- Adam в Keras https://keras.io/api/optimizers/adam/

Документация по оптимизаторам на русском:

- https://ru-keras.com/optimizer/

![](.\images\Adam.gif)



## После обучения

Посмотрим, как теперь выглядит сеть, вызывая метод `.summary()`:

```python
model.summary()
```

```bash
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 32)                352       
=================================================================
Total params: 352
Trainable params: 352
Non-trainable params: 0
_________________________________________________________________
```

Внимательно посмотрите на получившееся количество параметров.

При входе сети из **10** чисел и Dense-слое с **32** нейронами количество весовых коэффициентов будет **10∗32=320**. Но, как видите, оно получается на **32** больше. Это происходит из-за наличия в слое **нейрона смещения**.

Для чего он нужен? Бывают ситуации, в которых нейросеть просто не сможет найти верное решение из-за того, что нужная точка будет находиться вне пределов досягаемости. Именно для этого и нужны такие нейроны, чтобы иметь возможность сместить область определения.

Схематически нейроны смещения обычно не обозначаются, их вес учитывается по умолчанию при работе нейрона.



Вы можете заметить, что у сети есть название **"sequential"**. Оно автоматически присваивается при создании.

У слоев также есть названия. Они указаны в левой колонке.

Колонка **"Output Shape"** показывает форму данных на выходе нейронного слоя.

В данном случае у вас получается:

- на вход нейронной сети подается последовательность из **10** элементов (вы указали это с помощью параметра **input_dim**)
- нейронная сеть состоит из одного слоя (**Dense**), который состоит из **32** нейронов (количество нейронов вы указали при создании слоя)
- на выходе нейронной сети будет последовательность из **32** элементов (выход нейронной сети равен выходу последнего слоя)

В выведенной информации вы можете увидеть строку **"Total params: 352"**. В ней указано общее количество параметров модели.

**Параметры**

Параметры модели – это все веса, внутренние настройки сети, которые определяют, как будет преобразован объект, подаваемый в сеть, прежде чем оказаться на выходе. Они автоматически изменяются при обучении.

Вы можете добавлять неограниченное количество слоев к сети:

```python
model = Sequential()
model.add(Dense(32, input_dim=10))
model.add(Dense(5))
model.add(Dense(1))

model.summary()
```

```bash
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 32)                352       
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 165       
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 6         
=================================================================
Total params: 523
Trainable params: 523
Non-trainable params: 0
_________________________________________________________________
```

Выше добавлено еще два линейных слоя. Теперь выход изменился: сеть на выходе выдает одно число. Как видите, также изменилось и общее число параметров. Чем больше слоев и нейронов в слоях, тем больше параметров сети.

Далее вы узнаете, как обучить модель нейронной сети.



## Распознавание рукописных цифр **MNIST**

### Создание нейронной сети

Для начала создайте объект нейронной сети с помощью класса **Sequential**:

```python
model = Sequential()
```

Сейчас это пустая нейронная сеть, не содержащая в себе никаких слоев и нейронов. Добавьте в нее несколько слоев нейронов, идущих друг за другом, последовательно, по образцу:

```python
model.add(Dense(400, activation='relu'))
```

В данном случае **400** – это количество нейронов в слое, а **'relu'** – функция активации, которая будет применяться после умножения значений входов нейрона на его веса.

```python
# Создание последовательной модели
model = Sequential()

# Добавление полносвязного слоя на 800 нейронов с relu-активацией
model.add(Dense(800, input_dim=784, activation='relu')) 

# Добавление полносвязного слоя на 400 нейронов с relu-активацией
model.add(Dense(400, activation='relu')) 

# Добавление полносвязного слоя с количеством нейронов по числу классов с softmax-активацией
model.add(Dense(CLASS_COUNT, activation='softmax'))
```

Теперь нужно подготовить НС к обучению (скомпилировать) и запустить само обучение.

Следующей строкой кода вы скомпилируете модель:

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

В методе `.compile()` вы назначаете функцию ошибки (**'categorical_crossentropy'**), оптимизатор нейронной сети (**'adam'**) и метрики, которые будут подсчитываться в процессе обучения нейросети (**['accuracy']**).

Метод `.summary()` выведет на экран структуру вашей нейронной сети в виде таблицы:

```python
# Компиляция модели
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Вывод структуры модели
print(model.summary())
```

```bash
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_4 (Dense)              (None, 800)               628000    
_________________________________________________________________
dense_5 (Dense)              (None, 400)               320400    
_________________________________________________________________
dense_6 (Dense)              (None, 10)                4010      
=================================================================
Total params: 952,410
Trainable params: 952,410
Non-trainable params: 0
_________________________________________________________________
None
```

Функция `plot_model()` модуля `utils` нарисует наглядную схему (граф) нейронной сети, она удобна для понимания и более сложных моделей.

Эта функция принимает следующие аргументы:

- **model** - модель, схему которой вы хотите построить (обязательный параметр);
- **to_file** - имя файла или путь к файлу, в который сохраняется схема (обязательный параметр);
- **show_shapes** - Показывать или нет формы входных/выходных данных каждого слоя (необязательный параметр, по умолчанию **False**);
- **show_layer_names** - показывать или нет название каждого слоя (необязательный параметр, по умолчанию **True**).

```python
utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=False)
```

![img](data:image/png;base64,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 )

Здесь вы впервые сталкиваетесь с сохранением данных.

Сейчас сохранение графа модели произойдёт в хранилище виртуальной машины ноутбука. Найти сохраненный файл можно, нажав на иконку **"Файлы"** в левой части рабочего пространства Google Colab.

В данном случае при завершении сеанса все файлы будут удалены вместе с виртуальной машиной.

Если необходимо воспользоваться файлами в дальнейшем, можно сохранить их на свой Google-диск - это постоянное хранилище. Для этого необходимо подключить его:

```python
from google.colab import drive
drive.mount('/content/drive/')
```

Необходимо перейти по ссылке и разрешить доступ.

Теперь данные могут быть загружены на диск, но для этого необходимо указывать полный путь для сохранения.



### Обучение нейронной сети

Одной строчкой кода вы запустите обучение нейронной сети и сможете наблюдать за процессом.

Для этого вызовите метод модели `.fit()` и передайте ему данные для обучения - **x_train, y_train**:

```python
model.fit(x_train, y_train, batch_size=128, epochs=15, verbose=1)
```

**batch_size** – размер батча, который указывает нейросети на то, сколько картинок она будет обрабатывать за один раз.

**epochs** – количество циклов обучения, то есть сколько раз нейронная сеть повторит просмотр и обучение на всех ваших данных.

```python
model.fit(x_train,        # обучающая выборка, входные данные
          y_train,        # обучающая выборка, выходные данные
          batch_size=128, # кол-во примеров, которое обрабатывает нейронка перед одним изменением весов
          epochs=15,      # количество эпох, когда нейронка обучается на всех примерах выборки
          verbose=1)      # 0 - не визуализировать ход обучения, 1 - визуализировать
```

```python
Epoch 1/15
469/469 [==============================] - 8s 16ms/step - loss: 0.2036 - accuracy: 0.9389
Epoch 2/15
469/469 [==============================] - 8s 16ms/step - loss: 0.0756 - accuracy: 0.9769
Epoch 3/15
469/469 [==============================] - 8s 16ms/step - loss: 0.0480 - accuracy: 0.9847
Epoch 4/15
469/469 [==============================] - 8s 16ms/step - loss: 0.0342 - accuracy: 0.9890
Epoch 5/15
469/469 [==============================] - 8s 17ms/step - loss: 0.0260 - accuracy: 0.9915
Epoch 6/15
469/469 [==============================] - 8s 17ms/step - loss: 0.0211 - accuracy: 0.9928
Epoch 7/15
469/469 [==============================] - 8s 17ms/step - loss: 0.0174 - accuracy: 0.9941
Epoch 8/15
469/469 [==============================] - 8s 17ms/step - loss: 0.0148 - accuracy: 0.9952
Epoch 9/15
469/469 [==============================] - 8s 17ms/step - loss: 0.0132 - accuracy: 0.9956
Epoch 10/15
469/469 [==============================] - 8s 16ms/step - loss: 0.0142 - accuracy: 0.9951
Epoch 11/15
469/469 [==============================] - 8s 16ms/step - loss: 0.0098 - accuracy: 0.9967
Epoch 12/15
469/469 [==============================] - 8s 17ms/step - loss: 0.0127 - accuracy: 0.9959
Epoch 13/15
469/469 [==============================] - 8s 17ms/step - loss: 0.0114 - accuracy: 0.9962
Epoch 14/15
469/469 [==============================] - 8s 16ms/step - loss: 0.0087 - accuracy: 0.9974
Epoch 15/15
469/469 [==============================] - 8s 16ms/step - loss: 0.0063 - accuracy: 0.9981
```

Только что вы наблюдали процесс обучения нейронной сети на 15 эпохах. После каждого цикла обучения вы можете видеть среднее значение ошибки. Обратите внимание, что практически каждую эпоху значение метрики точности (**accuracy**) увеличивается. Это означает, что ваша нейронная сеть с каждым разом делает все более точное распознавание!



### Подведем итоги: вспомним изученный материал

В первой части вы познакомились с ИИ, узнали, что он бывает сильным и слабым, и то, что к 2050 развитие технологии достигнет пика. Для ИИ практически нет неразрешимых задач, и область его использования безгранична. Еще вы узнали, что нейронная сеть неспроста так называется, и у нее есть прототип – человеческий мозг и нейроны, из которых он состоит.

Далее вы перешли к рассмотрению различных моделей нейрона (биологической и математической). В НС нейроны отвечают за выделение определенного признака, при этом некоторое количество нейронов организовано в слои. Несколько слоев нейронов могут называться полноценной НС.

Сама по себе сеть без обучения не может хорошо решать задачи. Для этого ей требуются две выборки – обучающая и проверочная. Используя их, НС узнает, как правильно выполнить задачу. В этом ей помогает функция ошибки, которая указывает, в правильную ли сторону НС меняет свои веса. После этого вы познакомились с полносвязным слоем, нейроны которого связаны со всеми входными нейронами.

Во второй части вы попробовали создать свою первую модель нейронной сети. Подгрузили в ноутбук основу **Sequential**, которая отвечает за построение модели, и создали начальную модель: `model = Sequential()`. Но она выглядела как пустая коробка, в которую необходимо что-то положить. И первым стал слой **Dense** (полносвязный слой). Так появилась модель, но ее еще нужно было обучить, и для этого вы указали оптимизатор и функцию потерь при помощи метода `.compile()`. Завершили все это вызовом метода `.summary()`, посмотрев, как выглядит структура НС.

Затем вы перешли к практике и решили первую задачу по распознаванию рукописных цифр. Для этого импортировали все необходимые инструменты и загрузили данные **MNIST**. Определили форму массива данных и представление данных в виде картинки. Преобразовали данные для модели НС, превратив картинку **28x28** пикселов в последовательность из **784** чисел. Не оставили без внимания и метки классов. Чтобы сеть лучше классифицировала, перевели метки в формат **one hot encoding**. По изученной схеме создали НС и приступили к ее обучению. В завершение проверили, как обученная НС распознает отдельные изображения рукописных цифр из набора, на котором сеть не обучалась.



