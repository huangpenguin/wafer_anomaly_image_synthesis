import pytest


@pytest.fixture()
def message():
  return 'hello pytest!'

def test_hello(message):
  assert message == 'hello pytest!'

