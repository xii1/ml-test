import os


class Config(object):
    ENV = os.getenv('FLASK_ENV')
    if ENV == 'development':
        CACHE_TYPE = 'RedisCache'
        CACHE_REDIS_HOST = 'localhost'
        CACHE_REDIS_PORT = 6379
        CACHE_REDIS_DB = 0
        CACHE_REDIS_URL = 'redis://localhost:6379/0'
        CACHE_DEFAULT_TIMEOUT = 3600
    else:
        CACHE_TYPE = 'RedisCache'
        CACHE_REDIS_HOST = 'redis'
        CACHE_REDIS_PORT = 6379
        CACHE_REDIS_DB = 0
        CACHE_REDIS_URL = 'redis://redis:6379/0'
        CACHE_DEFAULT_TIMEOUT = 3600
