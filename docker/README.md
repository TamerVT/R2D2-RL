# Docker

Build the image from the repository root:

```sh
docker build -f docker/Dockerfile -t rcs-dev .
```

Run the development container with Docker Compose:

```sh
docker-compose -f docker/compose/dev.yml run --rm rcs
```

Notes:

- The compose setup bind-mounts the repository into `/workspace/robot-control-stack`.
- Python source changes are picked up from the mounted repo.
- If you change C++ code in `rcs` or `rcs_fr3`, rebuild the image.
- For non-GPU hosts, comment out the GPU-related lines in `docker/compose/dev.yml`.
