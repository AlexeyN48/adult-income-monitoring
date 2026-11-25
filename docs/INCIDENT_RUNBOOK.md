# Runbook инцидентов мониторинга

Команды выполняются из корня проекта. Проверка расчёта (`success`) отделена от pass/fail качества: успешный расчёт с нарушением fairness — нормальный наблюдаемый результат, а не ошибка сервиса.

## Быстрая диагностика

```powershell
docker compose ps
docker compose logs --tail 60 exporter
docker compose logs --tail 40 prometheus
docker compose exec -T prometheus promtool check config /etc/prometheus/prometheus.yml
```

В Prometheus проверить up{job="adult-monitoring"}, ml_monitoring_available, ml_monitoring_errors_total, time() - ml_monitoring_last_success_timestamp_seconds и ml_active_scenario_info.

| Симптом | Проверка и действие |
| --- | --- |
| Exporter не стартует | Проверить журнал, наличие модели и scenario manifest. Для чистого окружения выполнить init по DEMO_RUNBOOK; не переобучать рабочую модель вслепую. |
| Docker npipe permission denied | Проверить docker info в своём терминале и доступ к запущенному Docker Engine. В среде агента требовался разрешённый CLI-вызов; UI и настройки ОС не изменялись. |
| Bind port already in use | Проверить занятые порты. Изменять опубликованные порты именно этого Compose-проекта и ссылки в командах; не останавливать чужие сервисы. |
| Prometheus target DOWN | Проверить exporter /health, журнал, имя exporter:8000 в prometheus.yml и оба контейнера одной Compose-сети. |
| Target UP, но available=0 | HTTP работает, но данных нет/они устарели/расчёт завершился ошибкой. Проверить errors_total, last-success age, status.json и журнал. |
| Метрики F1 отсутствуют | При ошибке или TTL >180 секунд экспортер намеренно подавляет бизнес-серии. Устранить причину и пересчитать сценарий; не подставлять F1=0. |
| POST /scenario вернул 409 | Уже идёт пересчёт. Подождать 5 секунд и повторить тот же запрос; прежний свежий снимок остаётся доступным. |
| POST timeout | Проверить active_scenario_info и журнал перед повтором: расчёт мог завершиться после разрыва HTTP-соединения. |
| Правило не обновилось после правки файла | Выполнить promtool check config, затем docker compose restart prometheus. Перезапуск сохраняет volume с историей. |
| Age drift есть, общий drift зелёный | Проверить per-feature: один столбец из 13 не достигает агрегатного порога 50%. FeatureDriftDetected использует индивидуальный сигнал. |
| Recovery не снимает fairness alert | FairnessThresholdViolated уже нарушен у baseline. Проверять отдельно ModelF1Low/FairnessDeteriorated и абсолютные пороги. |
| Нулевой знаменатель или маленькая группа | FairnessMetricsInvalid либо unavailable сигнализирует непригодность; null не равен отсутствию перекоса. Собрать подходящую разметку/объём, затем пересчитать. |

## Несогласованные SHA, модель или данные

Причины: вручную изменён CSV, обучение/prepare выполнены частично, модель заменена без новых сценариев. Проверки нельзя обходить редактированием manifest, порога или времени последнего успеха: это скроет реальное происхождение результата.

Если решено восстановить артефакты из исходного Adult, сначала остановить только exporter и сохранить нужную прежнюю модель/отчёты. Затем выполнить согласованную последовательность:

```powershell
docker compose stop exporter
docker compose --profile init run --rm --no-deps init
docker compose --profile init run --rm --no-deps init python -m adult_monitoring scenarios --config configs/example.yaml
docker compose --profile init run --rm --no-deps init python -m adult_monitoring monitor --config configs/example.yaml --scenario all
docker compose up -d exporter
```

Это переобучение создаёт новые артефакты и SHA. Не выполнять одновременно со старым экспортером: прямой monitor и HTTP-пересчёт одного сценария используют одни файлы. При восстановлении перепроверить параметры, порог из validation, качество и список входов.

## Критерии восстановления

1. Exporter healthy, target UP, ml_monitoring_available=1.
2. Last-success timestamp обновлён реальным завершённым Evidently-расчётом; status.json и monitoring.json имеют success и одинаковый timestamp.
3. Нужный active_scenario_info=1, F1 и drift/fairness соответствуют этому сценарию.
4. ModelF1Low снят на recovery; исходные абсолютные fairness FAIL не замаскированы.
5. Для проверки цикла можно повторить scripts/verify_stack.py и сохранить новое evidence.json.

Остановка стенда: docker compose down. История в именованном volume сохраняется. Команды удаления volume и данных в инструкции намеренно не требуются.
