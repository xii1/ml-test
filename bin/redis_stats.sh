#!/usr/bin/env bash

docker exec -it redis redis-cli info stats | grep -E '^keyspace|expired_keys|evicted_keys'