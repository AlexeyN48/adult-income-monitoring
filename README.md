# Adult Income: стенд мониторинга ML

Учебный проект подготовки CatBoost на настоящем датасете UCI Adult с дальнейшим мониторингом. Реализованы подготовка данных, модель, сценарии, отчёты Evidently и Prometheus stack.

## Подготовка окружения

Нужен CPython 3.12 (проверенная команда в этой среде — `python`; launcher `py -3.12` не работает). Зависимости устанавливаются только в виртуальное окружение проекта:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

Если `.venv` уже создана, используйте её, не пересоздавая. После обновления зависимостей повторите `.\.venv\Scripts\python.exe -m pip install -e .`. Прямые зависимости указаны в `pyproject.toml`: CatBoost 1.2.10, Evidently 0.7.23, pandas 2.3.3, prometheus-client 0.26.0 и PyYAML 6.0.3.

## Получить данные и обучить модель

Из корня репозитория:

```powershell
.\.venv\Scripts\python.exe -m adult_monitoring prepare --config configs/example.yaml
.\.venv\Scripts\python.exe -m adult_monitoring train --config configs/example.yaml
```

Или запустить обе команды как единый pipeline:

```powershell
.\.venv\Scripts\python.exe -m adult_monitoring pipeline --config configs/example.yaml
```

Первая команда загружает официальный архив UCI Adult в `data/raw/adult.zip`, сохраняет исходные файлы и строит четыре CSV в `data/processed/`. Архив кешируется; checksum и атрибуция фиксируются в `data/processed/data_manifest.json`. Если архив некорректен, программа пытается получить его заново. Нужен доступ к `archive.ics.uci.edu` при первом запуске.

Источник: Barry Becker и Ronny Kohavi (1996), [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult), DOI [10.24432/C5XW20](https://doi.org/10.24432/C5XW20), лицензия CC BY 4.0. Данные преобразуются для учебного проекта: очистка, удаление точных дублей и разбиение; исходная лицензия сохраняется. Атрибуция также записана в манифест.

В подготовке пробелы удаляются, точка после меток из `adult.test` убирается, `?` в категориальных полях заменяется явным `__MISSING__`. Неизвестная целевая метка или некорректное числовое поле приводят к ошибке. Точные повторы проверяются по всем 15 очищенным исходным колонкам; если запись есть в обоих источниках, сохраняется запись из официального train файла. Удалённые повторы и исходный `row_id` формата `файл:номер_строки` отражены в manifest.

Официальный `adult.data` разбивается стратифицированно на train (80%) и validation (20%). Официальный `adult.test` делится пополам на reference и current. Разбиение воспроизводится seed из конфигурации; все четыре части не пересекаются ни по `row_id`, ни по полным очищенным записям. Исходные тестовые строки не участвуют в обучении.

CatBoost обучается на CPU с фиксированным seed. `sex` остаётся в данных для аудита, но исключён из признаков; `row_id` и `income` также исключены. Фиксированная модель и список категориальных признаков записаны в сводку.

## Артефакты и пороги

- `data/processed/{train,validation,reference,current}.csv` — очищенные части с `row_id`, `sex` и истинным `income`.
- `data/processed/{reference,current}_predictions.csv` — исходные поля, две вероятности (`probability_le50k`, `probability_gt50k`) и бинарный `prediction` при пороге 0.5.
- `models/catboost.cbm` — модель CatBoost.
- `reports/training_summary.json` — F1 положительного класса `>50K`, macro-F1 валидации, размеры/баланс, feature schema, параметры и версии.
- `data/processed/data_manifest.json` — источник, checksums, разбиение, дедупликация, баланс, параметры и версии.

Порог бинарного решения модели равен 0.5. Будущий порог алерта низкого F1 выбирается один раз по validation: `max(0, validation_F1_positive - 0.05)`. Он сохраняется в manifest и сводке; current не используется для его подбора. Это учебный запас относительно baseline validation F1, а не универсальная производственная норма.

## Сценарии мониторинга

После подготовки данных и обучения запускайте генератор отдельно, не переобучая модель:

```powershell
.\.venv\Scripts\python.exe -m adult_monitoring scenarios --config configs/example.yaml
```

Генератор проверяет SHA256 `train`, `validation`, `reference`, `current`, оба CSV прогнозов и CBM по `data_manifest.json`/`training_summary.json`, проверяет схему и непересечение исходных частей. Затем создаёт в `data/processed/scenarios/` CSV и `manifest.json`. Исходные данные, модель, манифест блока 2 и training summary не изменяются. Seed и интенсивности задаются в секции `scenarios`.

Сценарии: `baseline` — признаки и настоящие метки current с заново рассчитанными CatBoost прогнозами; `feature_drift` — `age +5`; `feature_drift_strong` — `age +15`; `target_drift` — выборка с целевыми 60% меток `>50K` с возвращением внутри классов, без переписывания меток; `fairness_shift` — у Female целевая доля настоящих `>50K` 3%, Male сохраняется полностью; `recovery` — побайтная копия baseline; `fairness_defect` — явно синтетический постпроцессинг: вероятность Female умножается на 0.05 и решение считается повторно. `fairness_defect` не является предсказанием CatBoost: в каждой строке сохраняются оригинальные `model_probability_gt50k`/`model_prediction` и эффективные `probability_gt50k`/`probability_le50k`/`prediction`.

В каждом CSV `row_id` указывает исходную запись, а уникальный `sample_id` различает повторные выборки (`#draw=N`). `synthetic_postprocessing_applied` отмечает Female в синтетическом дефекте и равен false во всех остальных случаях. Manifest раскрывает seed, параметры трансформаций, контрольные суммы, повторы, размеры групп/классов и диагностические F1/selection rate/TPR. Диагностика генератора предназначена для сверки, но не используется как источник значений Evidently. Недостаточный размер положительной подгруппы делает fairness shift ошибкой генерации, а неопределённые знаменатели представлены `null` вместе с признаками валидности. Для оценки fairness требуется минимум 30 строк и 30 положительных меток на группу; повторные выборки увеличивают число строк, но не число независимых исходных наблюдений. Это учебная демонстрация, а не обоснование статистической значимости повторов.

Baseline Adult уже имеет заметное различие selection rate по полу (на current и модели этого проекта: Female 0.0831, Male 0.2579; disparate impact около 0.322). Поэтому восстановление baseline не означает снятие постоянного fairness-сигнала. Целевые метки в сценариях не меняются вручную: target/fairness shift достигаются только отбором или повторной выборкой настоящих строк.

Параметры в `configs/example.yaml` применяются напрямую; для быстрого CPU-прогона по умолчанию выбраны 400 итераций и глубина 6. Генерируемые данные, модели и отчёты исключены из Git.

Поля `source`, `target_column`, `positive_label`, `missing_category_marker`, `duplicate_policy`, имя модели, `task_type` и `decision_threshold` описывают фиксированный контракт этого этапа: неподдерживаемое значение вызывает ошибку. `reference_split` и `current_split` зарезервированы для следующего блока, имена выходных частей пока фиксированы.

## Evidently monitoring

После подготовки блоков 2–3 запускайте мониторинг всех семи сценариев или одного по alias из manifest:

```powershell
.\.venv\Scripts\python.exe -m adult_monitoring monitor --config configs/example.yaml --scenario all
.\.venv\Scripts\python.exe -m adult_monitoring monitor --config configs/example.yaml --scenario fairness_defect
```

Для каждого успешного расчёта создаются `reports/evidently/<scenario>/report.html`, `female_target_report.html`, `snapshot.json`, `monitoring.json` и `status.json`. HTML содержит графики распределений, встроенные Evidently drift tests, custom group metrics и pass/fail проверки. `monitoring.json` предназначен для будущего экспортера: current/reference скаляры индексируются стабильными ключами; в нём также есть sample counts, источник, SHA256, model/schema lineage, версия Evidently, время UTC, формулы и `null` для неопределённых отношений. `status.json` становится `success` только после записи файлов; при начале расчёта он `calculating`, при ошибке — `failed`, ранее сохранённый result помечается `stale`.

Data drift охватывает ровно 13 модельных признаков из `training_summary.json`: числовые сравниваются Evidently normalized Wasserstein (threshold 0.10), категориальные — Jensen–Shannon distance (0.10). Целевой `income` оценивается отдельно как Jensen–Shannon drift; отдельный отчёт повторяет сравнение только внутри Female. Пол, prediction и income не входят в знаменатель feature drift. В JSON присутствуют drift score/test по каждому из 13 столбцов и aggregate drift share (50%); отдельный `age_drift_status` сохраняет сигнал возраста, даже если агрегат ниже долевого порога.

Положительный класс — `>50K`, бинарный label 1. F1 alert threshold берётся из validation training summary (0.663487); macro-F1 усредняет F1 обоих классов. Для каждой группы показываются размер, число настоящих положительных меток, selection rate и TPR. DPD = Female selection rate − Male selection rate; DI = Female / Male selection rate; TPR gap = Female TPR − Male TPR. Учебные проверки: `abs(DPD) <= 0.10`, `DI >= 0.80`, `abs(TPR gap) <= 0.10`; fairness ratios требуют минимум 30 строк и 30 положительных меток на каждую группу. Неопределённый знаменатель даёт null и проваленную validity проверку, а HTML выводит N/A. Это наблюдаемые различия, не доказательство причинной дискриминации.

Baseline уже не проходит абсолютные DPD/DI учебные проверки (DPD около -0.175, DI около 0.322); `recovery` возвращает те же сигналы, не устраняя их. Synthetic `fairness_defect` специально снижает F1 ниже validation порога; это отдельный постпроцессинг, не естественный прогноз CatBoost.

## Фактический результат блока 2

Проверено 24 ноября 2026 года, Python 3.12.10 и зависимости из `pyproject.toml`. Из исходных 48 842 записей удалено 52 полных дубля. Размер train — 26 029; пересечения четырёх частей по исходным идентификаторам и полному содержимому равны нулю.

| Часть | Записей | F1 класса `>50K` | Macro-F1 |
| --- | ---: | ---: | ---: |
| Validation | 6 508 | 0.713487 | 0.816365 |
| Reference | 8 126 | 0.714367 | 0.817670 |
| Current | 8 127 | 0.703191 | 0.810260 |

Сохранённая модель независимо загружена через `load_model`: вероятности и бинарные предсказания совпали с сохранёнными CSV, F1 проверен через `sklearn.metrics.f1_score`. `sex`, `row_id` и `income` отсутствуют в `model.feature_names_`. Контрольные суммы всех подготовленных CSV и модели проверены. `pip check` не обнаружил нарушенных зависимостей.

Порог будущего алерта равен **0.663487**. Лучшая итерация модели — 320 (нумерация с нуля). Повторный полный запуск дал те же метрики и побайтно совпадающие прогнозные CSV; SHA самого файла CBM между обучениями менялся, поэтому побайтная идентичность модели не гарантируется. В манифесте фиксируется SHA текущего файла. Транзитивные зависимости пока не зафиксированы lock-файлом.

Сеть из обычной сессии агента была недоступна: установка в `.venv` и первая загрузка UCI выполнены через разрешённый сетевой запуск. Последующие запуски подготовки используют локальный архив. Результат блока 2 проверяет обучение и артефакты; результаты Evidently и Prometheus описаны ниже в блоках 4 и 5.

## Проверки

Запуск 25 тестов подготовки данных, обучения, сценариев, Evidently и экспортера. Проверяются повторяемость seed, восстановление, сохранение меток и признаков, синтетический дефект, неверный checksum, неопределённые fairness-метрики, свежесть, согласованность метрик и конкурирующие HTTP-запросы. Тесты автономны: используют маленькие явно синтетические данные и временные файлы, не требуют загрузки Adult и не изменяют рабочие артефакты:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Полный план проекта и следующих блоков: [`PROJECT_PLAN.md`](PROJECT_PLAN.md).

## Фактический результат блока 3

На модели блока 2 сгенерированы семь сценариев по 8 127 строк. Повторный запуск дал те же SHA всех семи CSV. Baseline и recovery совпадают побайтно; модель и все CSV блока 2 сохранили свои контрольные суммы. Оркестратор независимо проверил происхождение каждой строки, сохранение меток, повторный inference CatBoost, преобразование вероятностей в дефекте и F1 через sklearn.

| Сценарий | Изменение | F1 `>50K` |
| --- | --- | ---: |
| baseline | Исходный current | 0.703191 |
| feature_drift | Возраст +5 лет | 0.709071 |
| feature_drift_strong | Возраст +15 лет | 0.686700 |
| target_drift | 4 876 положительных меток, около 60% | 0.771259 |
| fairness_shift | У Female 82 положительные метки из 2 721, около 3%; Male неизменён | 0.698373 |
| fairness_defect | Синтетическое подавление вероятности Female | 0.644947 |
| recovery | Возврат к baseline | 0.703191 |

В target_drift имеется 3 093 повторных выборки сверх числа уникальных row_id, в fairness_shift — 1 051. Повторы не являются новыми независимыми наблюдениями. Сдвиг возраста и баланса классов сам по себе не обязан уменьшать F1. Синтетический дефект с F1 ниже заранее сохранённого порога 0.663487 используется в блоке 5 для проверки алерта ModelF1Low.

## Фактический результат блока 4

Все семь сценариев обработаны Evidently 0.7.23. Независимая сверка `F1 >50K` с `sklearn.metrics.f1_score` совпала с результатами monitoring JSON; sample count — 8 127 в каждом current. Macro-F1, групповые selection rates/TPR, DPD, DI и TPR gap пересчитаны из CSV; custom metrics и их tests присутствуют в Evidently snapshot и HTML. HTML создан локально; открыть его можно, например, `reports/evidently/feature_drift/report.html` и `reports/evidently/fairness_defect/report.html`.

| Сценарий | F1 `>50K` | Feature drift share | Age drift test |
| --- | ---: | ---: | --- |
| baseline | 0.703191 | 0/13 | pass |
| feature_drift (+5 лет) | 0.709071 | 1/13 | fail |
| feature_drift_strong (+15 лет) | 0.686700 | 1/13 | fail |
| target_drift | 0.771259 | 7/13 | fail |
| fairness_shift | 0.698373 | 0/13 | pass |
| recovery | 0.703191 | 0/13 | pass |
| fairness_defect | 0.644947 | 0/13 | pass |

Для сдвига возраста общий feature drift share равен 1/13 (0.077), поэтому 50% aggregate test проходит; отдельный age test корректно проваливается. `fairness_shift` имеет целевой drift Female subgroup, хотя общий target drift test проходит. Для baseline DPD около -0.174804, DI 0.322102 и TPR gap -0.065659: абсолютные учебные проверки DPD/DI остаются fail, а recovery не меняет это исходное состояние. В `fairness_defect` F1 ниже сохранённого порога 0.663487.

Запущено `python -m unittest discover -s tests -v`: 17 тестов пройдены. Автономные проверки мониторинга используют небольшие synthetic fixtures, проверяют формулы, Evidently snapshot/tests/HTML, пустые группы, нулевые знаменатели, настройки минимального объёма групп, включённые границы порогов и сохранение доступного reference при недоступном current. Неверный checksum отклоняется без Adult/network. Полный real-data `monitor --scenario all` завершился успешно; для каждого сценария созданы HTML и JSON отчёты с source hashes и UTC freshness. Проверка HTML автоматизирована по содержимому; визуальный просмотр страниц пользователь выполняет самостоятельно. Управление браузером и приложениями на ПК без прямого запроса пользователя не используется.

Типы колонок заданы явно через Evidently DataDefinition из схемы обучения: это исключает автоматическое превращение возраста в категориальный столбец на маленьком пакете. Оркестратор независимо сверил все семь результатов с sklearn (F1), прямыми групповыми формулами и SciPy (Wasserstein и Jensen–Shannon для общего target и Female target). Подбор порогов по текущим сценариям не проводился.

`success` обозначает завершённый расчёт, а здоровье модели определяется отдельными pass/fail проверками. Экспортер проверяет freshness и подавляет business metrics при невалидном либо старом отчёте. Запросы сценариев выполняются последовательно; повторный конкурентный POST получает 409.
# Блок 5. Prometheus и алерты

Экспортер запускает настоящий `calculate_scenario(config, alias)` из Evidently. Он берёт только `reports/evidently/<scenario>/monitoring.json`, если связанный `status.json` и JSON отчёта имеют статус `success`, совпадающий сценарий и один UTC timestamp. После ошибки прошлый timestamp сохраняется, но бизнес-метрики подавляются до следующего успешного расчёта. Неопределённые fairness/target значения не превращаются в ноль. Переключение сценария атомарно; параллельный POST получает HTTP 409.

## Запуск

С имеющимися моделями/сценариями запустите стек из корня проекта:

```powershell
docker compose up --build -d
docker compose ps
```

Первый старт экспортера рассчитывает baseline; это может занять несколько минут. Если окружение чистое и артефактов ещё нет, сначала подготовьте их отдельным init-профилем:

```powershell
docker compose --profile init run --rm init
docker compose --profile init run --rm init python -m adult_monitoring scenarios --config configs/example.yaml
docker compose up --build -d
```

Exporter доступен на `http://127.0.0.1:8000/metrics`, Prometheus на `http://127.0.0.1:9090`. `/health` проверяет только готовность HTTP сервера. Переключение сценария выполняет loopback POST:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/scenario `
  -ContentType 'application/json' -Body '{"scenario":"fairness_defect"}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/scenario `
  -ContentType 'application/json' -Body '{"scenario":"recovery"}'
```

Поддерживаются только `baseline`, `feature_drift`, `feature_drift_strong`, `target_drift`, `fairness_shift`, `fairness_defect`, `recovery`. Расчёт сохраняет отчёт в соответствующем каталоге `reports/evidently/`. Prometheus опрашивает exporter каждые 5 секунд, правила оцениваются каждые 5 секунд, TTL отчёта — 180 секунд. Хранилище Prometheus остаётся в именованном volume.

## PromQL для графиков

Вставьте запросы в Prometheus Graph. Метрика F1 не имеет scenario label, поэтому baseline → defect → recovery рисуется одной временной серией.

```promql
ml_f1_score
ml_f1_threshold
ml_feature_drift_score
ml_feature_drift_detected
ml_target_drift_detected
ml_group_selection_rate
ml_group_tpr
ml_disparate_impact
ml_monitoring_available
time() - ml_monitoring_last_success_timestamp_seconds
```

`ModelF1Low` использует threshold из отчёта обучения (`0.663487`) и должен перейти Pending → Firing на `fairness_defect` (фактический F1 около `0.644947`), затем сняться на recovery (около `0.703191`). В baseline DI около `0.322` уже ниже абсолютного порога `0.8`; `FairnessThresholdViolated` поэтому постоянен и не снимается от простого возврата к baseline. Для ухудшения относительно baseline есть `FairnessDeteriorated`: фиксированный reference DI около `0.32648`, алерт срабатывает, когда текущий DI становится менее 80% reference. Правила также включают индивидуальные feature drift, target и female-target drift, unavailable/stale и scrape-down.

## Локальная проверка

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
docker compose config
docker compose exec prometheus promtool check config /etc/prometheus/prometheus.yml
docker compose exec prometheus promtool test rules /etc/prometheus/rules.test.yml
```

Или проверьте тесты правил до запуска Prometheus образом `prom/prometheus:v3.10.0`: `promtool test rules` ожидает файл `prometheus/rules.test.yml` и `prometheus/rules.yml` рядом.

## Проверка живого алерта без браузера

При запущенном стеке выполнить в PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts/verify_stack.py
.\.venv\Scripts\python.exe scripts/plot_prometheus_evidence.py
```

То же через уже работающий контейнер, без локального Python-окружения:

```powershell
docker compose exec -T exporter python scripts/verify_stack.py --prometheus http://prometheus:9090
docker compose exec -T exporter python scripts/plot_prometheus_evidence.py
```

Скрипт проверяет реальный scrape target и последовательно выбирает baseline, fairness_defect и recovery. Из API правил он ожидает Inactive → Pending → Firing → Inactive, сверяет F1 с порогом и активным сценарием, запрашивает `ALERTS` и измеренные временные ряды через `query_range`. При неуспехе возвращает ненулевой код и пытается восстановить recovery. Это HTTP-проверка работающей системы, без управления браузером.

Результаты пишутся в `reports/prometheus_demo/evidence.json` и `reports/prometheus_demo/f1.png` (не входят в Git). PNG построен по реальным API-данным и наблюдениям состояния алерта; это не скриншот интерфейса Prometheus. Чтобы показать именно интерфейс, самостоятельно откройте `http://127.0.0.1:9090/query`, выберите график и введите `ml_f1_score`; состояние правил доступно в `http://127.0.0.1:9090/alerts`.

Пока обновление выполняется, экспортер продолжает отдавать предыдущий ещё свежий успешный снимок, чтобы не сбрасывать Pending/Firing. В случае ошибки или истечения 180 секунд F1 и другие бизнес-метрики подавляются, доступность становится 0; время последнего успеха не продлевается. Ошибки записываются в журнал контейнера, а фоновый пересчёт выбирает актуальный сценарий под той же блокировкой, что и POST.

## Фактический результат блока 5

Окончательный Docker-образ собран и запущен вместе с Prometheus v3.10.0. Экспортер healthy, scrape target `adult-monitoring` имеет состояние UP. Контрольная сумма исходника экспортера в работающем контейнере совпала с файлом проекта. Все 25 Python-тестов, Compose validation, promtool check config (9 правил) и promtool test rules прошли.

Живая проверка через API окончательного контейнера завершилась успешно:

| Сценарий | Измеренный F1 | Состояние ModelF1Low |
| --- | ---: | --- |
| baseline | 0.7031911889296809 | Inactive |
| fairness_defect | 0.644947209653092 | Pending → Firing |
| recovery | 0.7031911889296809 | Inactive |

Порог во всех случаях — 0.663487. Получена реальная серия `ALERTS{alertname="ModelF1Low",alertstate="firing"}`. В evidence.json сохранены 14 samples F1, 14 samples порога и 4 samples firing ALERTS. График `reports/prometheus_demo/f1.png` проверен по сохранённым данным. Отдельно на работающем стенде зафиксирован FeatureDriftDetected в Firing с `feature="age"`; ответ API сохранён в `reports/prometheus_demo/drift_alert.json`.

Стенд после проверки оставлен в recovery: низкий F1 устранён. Исходный абсолютный fairness-сигнал сохраняется ожидаемо. Интерфейс браузера не открывался автоматически; ручной показ графика и Alerts нужен для защиты. История Prometheus сохраняется в volume; `docker compose down` остановит этот стенд, сохранив историю. Для повторного показа можно снова запустить verify_stack.py.
