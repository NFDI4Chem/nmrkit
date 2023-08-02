from databases import DatabaseURL
from starlette.config import Config
from starlette.datastructures import Secret
import lwreg

config = Config(".env")

PROJECT_NAME = "NMR Kit"
VERSION = "1.0.0"
API_PREFIX = "/api"

SECRET_KEY = config("SECRET_KEY", cast=Secret, default="CHANGEME")

POSTGRES_USER = config("POSTGRES_USER", cast=str)
POSTGRES_PASSWORD = config("POSTGRES_PASSWORD", cast=Secret)
POSTGRES_SERVER = config("POSTGRES_SERVER", cast=str, default="db")
POSTGRES_PORT = config("POSTGRES_PORT", cast=str, default="5432")
POSTGRES_DB = config("POSTGRES_DB", cast=str)

DATABASE_URL = config(
    "DATABASE_URL",
    cast=DatabaseURL,
    default=f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}",
)

cfg = lwreg.utils.defaultConfig()
cfg["dbname"] = POSTGRES_DB
cfg["host"] = POSTGRES_SERVER
cfg["user"] = POSTGRES_USER
cfg["password"] = POSTGRES_PASSWORD
cfg["dbtype"] = "postgresql"
cfg["standardization"] = config("STANDARDIZATION", cast=str, default="tautomer")

LWREG_CONFIG = cfg
