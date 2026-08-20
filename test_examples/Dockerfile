# Base: debian:bookworm-slim (Debian 12, ~30MB) — lighter than ubuntu:22.04.
# Using the debian-11 build of rDock (glibc 2.31); Debian bookworm (glibc 2.36)
# is fully backward compatible.
#
# Obtain the tarball from:
# https://github.com/CBDD/rDock/releases/download/v24.04.204-legacy/rdock-legacy-24.04.204_debian-11_g++_x86_64.tar.gz
FROM debian:bookworm-slim
LABEL description="rxdock/rdock molecular docking examples" \
      maintainer="rxdock-deepdive-examples"
RUN apt-get update \
    && apt-get install -y --no-install-recommends make g++ libpopt-dev libpopt0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
COPY rdock-legacy-24.04.204_debian-11_g++_x86_64.tar.gz /app/
WORKDIR /app/
RUN tar -zxf rdock-legacy-24.04.204_debian-11_g++_x86_64.tar.gz \
    && rm rdock-legacy-24.04.204_debian-11_g++_x86_64.tar.gz
ENV RBT_ROOT=/app \
    PATH=/app/bin:${PATH} \
    LD_LIBRARY_PATH=/app/lib
