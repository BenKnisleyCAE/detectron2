CAE Detectron Docker Setup
=

## Build Docker Image

`cd` into the repo then into the `docker` directory

```sh
docker build --build-arg USER_ID=$UID -t detectron2:v0 .
```

## Run the Docker Container
```sh
docker run -p 7008:7008 \
-v /mnt/c/Users/knisleybe/Desktop/mount/input/:/mnt/input \
-v /mnt/c/Users/knisleybe/Desktop/mount/output/:/mnt/output \ detectron2:v0
```