--------------------------------------------
Локальный запуск для разработки

source venv/bin/activate

flask --app app run --host=0.0.0.0

--------------------------------------------
Создание докер файла для запуска приложения

-- Собрать "docker_образ" по инструкциям записанным файле Dockerfile
--- Один раз !!!! запустить команду
docker image build -t flask_docker app

-- Один раз !!! запустить новый контейнер из собранного "docker_образа"
   выполнив команду - команда "растянута" на несколько строк

---- начало команды - не копироать
docker run -p 5001:5000 --name flask_docker \
  --mount type=bind,source="${PWD}/model/yolo_new.h5",target=/model/yolo_new.h5 \
  --mount type=bind,source="${PWD}/resource",target=/resource \
  --mount type=bind,source="${PWD}/templates",target=/templates \
  --mount type=bind,source="${PWD}/app/app.py",target=/app.py \
  --mount type=bind,source="${PWD}/app/predict.py",target=/predict.py \
  -d flask_docker
---- завершение команды - не копироать

-- проверить, что контейнер запущен
   1) выполнить команду
   docker container ps

   для работающего контейнера вывод на консоль выглядит так
CONTAINER ID   IMAGE          COMMAND                  CREATED          STATUS          PORTS                                       NAMES
               flask_docker   "flask --app app run…"   36 seconds ago   Up 35 seconds   0.0.0.0:5001->5000/tcp, :::5001->5000/tcp   flask_docker

    2) открыть в web браузере ссылку http://localhost:5001

-- Использование новой модели / или если есть изменения в алгоритме распознавания
   
   - новую модель копировать в каталог model - после требуется остановить и запустить контейнер
   - изменения в алгоритм распознавания вносить в файле app/predict.py - после требуется остановить и запустить контейнер
   
   - для остановки контейнера выполнить команду 
       docker stop flask_docker
   - для остановки контейнера выполнить команду 
       docker start flask_docker 

