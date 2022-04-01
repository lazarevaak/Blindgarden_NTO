# Blindgarden_NTO

## Используемые методы решения

### Распознавание и детектирование QR-кодов:

    Для распознавания QR-кодов используется библиотека pyzbar
    Часто при взлете квадрокоптера бывает такое, что он сильно отлетел от места старта (QR-код не видно),
    поэтому если не удалось найти QR-код при взлете, то квадрокоптер начинает двигаться по квадрату (со стороной 2м),
    с целью найти QR-код

### Распознавание линии, повреждений и разливов:

    Для распознавания использовалась библиотека технического зрения OpenCV.

    Для выделения объектов используется цветовое пространство HSV.

    Чтобы избежать неправильного детектирования линии, используется 
    несколько фильтров (по площади контура и по соотношению его сторон).

    Для нахождения повреждений используется тот факт, что при его появлении линия разделяется на два сегмента.
    Поэтому при разделении линии на две части, начинается поиск повреждений между этими двумя сегментами.
    Также применяются фильтры по площади контура и соотношению сторон.

    Если было найдено повреждение то, начинается поиск разливов.
    Поиск разливов ведется на небольшом расстоянии от центра повреждения.

### Распознавание посадочного места:

    Для распознавания посадочного места изображение переводится в оттенки серого.
    После чего, выделяются светлые объекты.

    Также применяется фильтр по соотношению сторон посадочного места,
    площади контура и проверка контура на выпуклость.
    
## Ход работы              

### Программная часть:

#### День № 1 (28.03.22)

        1) Взлет
        2) Распознавание полки
        3) Посадка на полку
        4) Тренировка полетов
        
Код: https://github.com/lazarevaak/Blindgarden_NTO/blob/main/blindgarden_day1.py

#### День № 2 (29.03.22)

    (Работа над 1-ым заданием)

    1) Взлет
    2) Распознавание Qr-code
    3) Получение информации о начале линии из Qr-code
    4) Вывод в терминал сообщения с результатами распознавания
    5) Полет по линии
    6) Посадка в районе старта 
    7) Тренировка полетов

Код: https://github.com/lazarevaak/Blindgarden_NTO/blob/main/blindgarden_day2.py

#### День № 3 (30.03.22)

    (Работа над 1-ым заданием)

    1) Обнаружение повреждений нефтепровода или их отсутствие
    2) Вывод в терминал информации о нахождении повреждений нефтепровода
    3) Выделение розовым контуром повреждений нефтепровода
    4) Запись в топик изображения с повреждениями
    5) Обнаружение разлива нефти
    6) Расчет площади разлива
    7) Выделение контуром синего цвета разлива нефти
    8) Запись в топик
    9) Тренировка полетов
    10) Зачетная попытка, сдача решения первого задания

    На зачете столкнулись с такой проблемой, что для коптера линия была перевернута,
    и ему требовался поворот для дальнейшего движения по линии, что в нашей первой 
    зачетной программе не предусматривалось.

Код: https://github.com/lazarevaak/Blindgarden_NTO/blob/main/blindgarden_day3.py

#### День № 4 (31.03.22)

    (Работа над 2-ым заданием)

    Попытка решить проблему, которая повлияла на предыдущий засчет (30 марта).

    1) Добавить точность в распознавание повреждений и разливов нефти
    2) Мониторинг сразу двух разветвлений желтой линии 
    3) Добавили поворот коптера при условии, что полоса перевернута
    4) Тренировка полетов
    5) Посадка в точке старта

Код:

### Инженерная часть:

#### День № 1 (29.03.22)

    1) Разработка идеи 
    2) Реализация 3D модели в собранном и разобранном виде

#### День № 2 (30.03.22)

    1) Написание спецификации
    2) Создание сборочного чертежа
    3) Написание монтажной инструкции

#### День № 3 (31.03.22)

    1) Написание инструкции по эксплуатации
    2) Видео-презентация модели в 3D с демонстрацией
