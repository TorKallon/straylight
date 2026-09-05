FROM postgres:17.10-alpine3.24@sha256:742f40ea20b9ff2ff31db5458d127452988a2164df9e17441e191f3b72252193 AS pgvector-build

ARG PGVECTOR_COMMIT=159b79aaad5983fb7459c1e3df2897fbb2d11788
ARG PGVECTOR_RELEASE=v0.8.5
RUN apk add --no-cache build-base git
WORKDIR /src
RUN git clone --filter=blob:none https://github.com/pgvector/pgvector.git . \
    && git checkout --detach "${PGVECTOR_COMMIT}" \
    && test "$(git rev-parse HEAD)" = "${PGVECTOR_COMMIT}" \
    && test "$(git describe --tags --exact-match)" = "${PGVECTOR_RELEASE}" \
    && make OPTFLAGS="" with_llvm=no \
    && make install with_llvm=no

FROM golang:1.26.6-alpine3.23@sha256:e57c41c1d5864341031181b0db34b9a537bb5773eb6428e4e5bdaea0f9135406 AS gosu-build

ARG GOSU_COMMIT=6456aaa0f3c854d199d0f037f068eb97515b7513
ARG GOSU_RELEASE=1.19
RUN apk add --no-cache git
WORKDIR /src
RUN git clone --filter=blob:none https://github.com/tianon/gosu.git . \
    && git checkout --detach "${GOSU_COMMIT}" \
    && test "$(git rev-parse HEAD)" = "${GOSU_COMMIT}" \
    && test "$(git describe --tags --exact-match)" = "${GOSU_RELEASE}" \
    && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/gosu .

FROM postgres:17.10-alpine3.24@sha256:742f40ea20b9ff2ff31db5458d127452988a2164df9e17441e191f3b72252193

ARG BRUNN_BUILD_REVISION=unknown
LABEL org.opencontainers.image.title="Brunn PostgreSQL with pgvector" \
      org.opencontainers.image.revision="${BRUNN_BUILD_REVISION}" \
      org.opencontainers.image.version="17.10-pgvector0.8.5"
RUN apk upgrade --no-cache libcrypto3 libssl3 libuuid
COPY --from=pgvector-build /usr/local/lib/postgresql/vector.so /usr/local/lib/postgresql/vector.so
COPY --from=pgvector-build /usr/local/share/postgresql/extension/vector* /usr/local/share/postgresql/extension/
COPY --from=gosu-build /out/gosu /usr/local/bin/gosu
COPY infra/postgres/healthcheck.sh /usr/local/bin/brunn-postgres-healthcheck
COPY infra/postgres/init/ /docker-entrypoint-initdb.d/

USER postgres:postgres

# The production lexical heap plus GIN/HNSW working set is about 1.95 GiB.
# Reserve enough cache for it without consuming the 8 GiB container's OS headroom.
CMD ["postgres", "-c", "shared_buffers=3GB", "-c", "effective_cache_size=6GB", "-c", "shared_preload_libraries=pg_prewarm"]
