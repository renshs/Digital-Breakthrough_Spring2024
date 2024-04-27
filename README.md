# Digital-Breakthrough_Spring2024

# Описание проекта

Этот проект посвящен разработке автоматизированной системы для анализа видеоуроков с целью оценки их качества, эффективности и вовлеченности студентов.

Проблема: Ручной анализ видеоуроков отнимает много времени и ресурсов, что делает его неэффективным для больших объемов данных.
Решение: Разработка автоматизированной системы, которая использует обработку данных и машинное обучение для анализа видеоуроков.

# Идея проекта

1. Группировка данных: Датасет с информацией о видеоуроках группируется по ID урока и дате начала.
2. Обработка данных: Проводится предварительная обработка данных, включая очистку и преобразование в нужный формат.
3. Анализ активности: Анализируется активность студентов на уроке по времени отправки сообщений.
4. Извлечение признаков: Извлекаются ключевые признаки, такие как интерактивность, события и средняя активность в минуту.
5. Кластеризация: Используется алгоритм кластеризации fastText для группировки видеоуроков по схожим характеристикам.
6. Анализ кластеров: Проводится анализ полученных кластеров для поиска закономерностей и выявления гипотез о факторах, влияющих на качество и эффективность обучения.

# Технологии

* FastText: Для кластеризации текста и анализа признаков.
* Python: Для обработки данных и реализации алгоритмов машинного обучения.
* Веб-сайт: Для предоставления интерфейса пользователю и визуализации результатов анализа.
* Серверная инфраструктура: Два сервера для обеспечения мультизадачности и балансировки нагрузки.

# Вызовы

* Недостаток данных: Ограниченное количество данных может снизить точность анализа.
* Битые данные: Необходимость обработки и очистки данных от ошибок и несоответствий.
* Качество кластеров: Поиск оптимальных параметров кластеризации для получения информативных и полезных * кластеров.
* Определение пола пользователя: Разработка методов для определения пола пользователя по его активности и сообщениям.

# Результаты

* Графики активности: Визуализация активности студентов на уроках по времени отправки сообщений.
* Кластеры видеоуроков: Группировка видеоуроков по схожим характеристикам для дальнейшего анализа.
* Гипотезы: Формулирование гипотез о факторах, влияющих на качество и эффективность обучения.
