import os

import pytest

from onecortex import Onecortex

HOST = os.environ.get("ONECORTEX_URL", "http://localhost")
ACCESS_TOKEN = os.environ.get("ONECORTEX_ACCESS_TOKEN", "")
EMAIL = os.environ.get("ONECORTEX_EMAIL", "")
PASSWORD = os.environ.get("ONECORTEX_PASSWORD", "")


@pytest.fixture(scope="session")
def oc_client():
    client = Onecortex(url=HOST)
    if ACCESS_TOKEN:
        client.auth.set_session(access_token=ACCESS_TOKEN)
    elif EMAIL and PASSWORD:
        client.auth.login(email=EMAIL, password=PASSWORD)
    return client
