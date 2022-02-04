## История машинного обучения

[TOC]

### Машинное обучение

![ml hype curve](./images/hype.png)

Термин «искусственный интеллект» был введен еще в 50-е годы прошлого века. К нему относится любая машина или программа, выполняющая задачи, «обычно требующие интеллекта человека». Со временем компьютеры справлялись все с новыми и новыми задачами, которые прежде требовали интеллекта человека, то есть то, что прежде считалось «искусственным интеллектом» постепенно перестало с ним ассоциироваться.

Машинное обучение — один из методов реализации приложений искусственного интеллекта, и с его помощью искусственный интеллект значительно продвинулся вперед. Но, хотя этот метод действительно очень важен, это далеко не первый значительный шаг в истории искусственного интеллекта: когда-то не менее важными казались экспертные системы, логический вывод и многое другое.

За последнее десятилетие термин «машинное обучение» широко цитируется в массмедиа и научных работах. Стал использоваться в отношении большого спектра разнообразных классов задач.

> Актуальные [данные](https://trends.google.ru/trends/explore?date=all&q=Machine%20Learning) на Google Trends.

Для решения каждой задачи создается **модель**, теоретически способная приблизиться к человеческому уровню решения данной задачи при правильных значениях **параметров**. Обучение этой модели – это постоянное изменение ее параметров, чтобы модель выдавала все лучшие и лучшие результаты.

Разумеется, это лишь общее описание. Как правило, вы не придумываете модель с нуля, а пользуетесь результатами многолетних исследований в этой области, поскольку создание новой модели, превосходящей существующие хотя бы на одном виде задач – это настоящее научное достижение. Методы задания целевой функции, определяющей, насколько хороши выдаваемые моделью результаты (**функции потерь**), также занимают целые тома исследований. То же самое относится к методам изменения параметров модели, ускорения обучения и многим другим. Даже начальная инициализация этих параметров может иметь большое значение!

В процессе обучения модель усваивает **признаки**, которые могут оказаться важными для решения задачи. Выделение таких признаков зачастую не менее, а иногда намного более ценно, чем решение основной задачи.






![](.\images\mashinnoe-obuchenie-dlya-nespecialistov.jpg)



![](.\images\uf0t6gyvgn4ooh_14jda1locmmq.jpeg)



![](.\images\ai-ml-ds.png)



### Вехи развития машинного обучения

![](.\images\AI History.png)

> [Диаграмма](./images/AI History.svg) в высоком качестве.

Знаменательные открытия(цвет - алгоритмы и теоремы), события(цвет - конференции,)  и внедрения технологий(цвет - MLOps):

💡 - открытие, 📺 - событие, 👷‍♂️- внедрение. 

|      |            |                                                              |
| :--: | ---------- | ------------------------------------------------------------ |
|  💡   | 1763, 1812 | [Bayes Theorem](https://wikipedia.org/wiki/Bayes%27_theorem) and its predecessors. This theorem and its applications underlie inference, describing the probability of an event occurring based on prior knowledge. |
|  💡   | 1805       | [Least Square Theory](https://wikipedia.org/wiki/Least_squares) by French mathematician Adrien-Marie Legendre. This theory, which you will learn about  in our Regression unit, helps in data fitting. |
|  💡   | 1913       | [Markov Chains](https://wikipedia.org/wiki/Markov_chain) named after Russian mathematician Andrey Markov is used to describe a sequence of possible events based on a previous state. |
|      | 1943       |                                                              |
|      | 1950       |                                                              |
|      | 1952       |                                                              |
|  📺   | 1956       | Dartmouth Summer Research Project                            |
|      | 1957       |                                                              |
|  👷‍♂️  | 1958       | LISP                                                         |
|      | 1959       |                                                              |
|      | 1960       |                                                              |
|  👷‍♂️  | 1964       | ELIZA                                                        |
|      | 1964       |                                                              |
|      | 1965       |                                                              |
|      | 1965       |                                                              |
|      | 1966       |                                                              |
|      | 1966       |                                                              |
|  💡   | 1967       | [Nearest Neighbor](https://wikipedia.org/wiki/Nearest_neighbor) is an algorithm originally designed to map routes. In an ML context it is used to  detect patterns. |
|      | 1968       |                                                              |
|      | 1969       |                                                              |
|      | 1970       |                                                              |
|      | 1970       |                                                              |
|      | 1972       |                                                              |
|      | 1972       |                                                              |
|      | 1973       |                                                              |
|      | 1974       |                                                              |
|      | 1979       |                                                              |
|      | 1982       |                                                              |
|      | 1983       |                                                              |
|      | 1986       |                                                              |
|      | 1987       |                                                              |
|      | 1989       |                                                              |
|      | 1992       |                                                              |
|      | 1994       |                                                              |
|      | 1995       |                                                              |
|  💡   | 1995       | SVM                                                          |
|      | 1996       |                                                              |
|  💡   | 1997       | LSTM                                                         |
|      | 2006       |                                                              |
|      | 2009       |                                                              |
|  💡📺  | 2012       |                                                              |
|  💡   | 2013       | Word2Vec                                                     |
|  💡   | 2014       | GAN                                                          |
|  👷‍♂️  | 2015       | MLOps                                                        |
|      | 2016       |                                                              |
|      | 2017       |                                                              |
|      | 2018       |                                                              |
|      | 2019       |                                                              |
|      | 2020       |                                                              |
|      | 2021       |                                                              |



- 💡1763, 1812 [Bayes Theorem](https://wikipedia.org/wiki/Bayes%27_theorem) and its predecessors. This theorem and its applications underlie inference, describing the probability of an event occurring based on prior knowledge.
- 💡1805 [Least Square Theory](https://wikipedia.org/wiki/Least_squares) by French mathematician Adrien-Marie Legendre. This theory, which you will learn about  in our Regression unit, helps in data fitting.
- 💡1913 - [Markov Chains](https://wikipedia.org/wiki/Markov_chain) named after Russian mathematician Andrey Markov is used to describe a sequence of possible events based on a previous state.
- 1943
- 1950
- 1952
- 1956 Dartmouth Summer Research Project
- 1957 [Perceptron](https://wikipedia.org/wiki/Perceptron) is a type of linear classifier invented by American psychologist Frank Rosenblatt that underlies advances in deep learning.
- 1958 LISP
- 1959
- 1960
- 1964 ELIZA
- 1964
- 1965
- 1965
- 1966
- 1966
- 1967 [Nearest Neighbor](https://wikipedia.org/wiki/Nearest_neighbor) is an algorithm originally designed to map routes. In an ML context it is used to  detect patterns.
- 1968
- 1969
- 1970 INTERNIST
- 1970 [Backpropagation](https://wikipedia.org/wiki/Backpropagation) is used to train [feedforward neural networks](https://wikipedia.org/wiki/Feedforward_neural_network).
- 1972 PROLOG
- 1972 Shakey 
- 1973
- 1974
- 1979 CNN
- 1982 Bayesian Network
- 1982 [Recurrent Neural Networks](https://wikipedia.org/wiki/Recurrent_neural_network) are artificial neural networks derived from feedforward neural networks that create temporal graphs.
- 1983 SOAP
- 1986
- 1987
- 1989 LeNet
- 1992
- 1994
- 1995 SVM
- 1995
- 1996
- 1997 LSTM
- 2006
- 2009
- 2012
- 2013 Word2Vec
- 2014 GAN
- 2016
- 2015 MLOps
- 2017
- 2018
- 2019
- 2020
- 2021
- 2022?



### Технологический стек

#### Фреймворки

#### MLOps

### Вызовы



---

### Источники

Пример организации MLOps цикла у Neoflex: https://www.neoflex.ru/solutions/neoflex-mlops-center

7 questions for DL: https://jameskle.com/writes/deep-learning-infrastructure-tooling



### Дополнительные материалы

