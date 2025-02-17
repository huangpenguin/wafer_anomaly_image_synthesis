from time import sleep
import pytest
from bevel_ml.utils.performance import Timer


def test_timer():
    timer = Timer()

    with timer:
        sleep(1.0)
    assert timer.elapsed_time == pytest.approx(1.0, abs=1e-3)

    sleep(1.0)
    assert timer.elapsed_time == pytest.approx(1.0, abs=1e-3)

    with timer:
        sleep(0.5)
    assert timer.elapsed_time == pytest.approx(1.5, abs=1e-3)


def test_timer_error():
    timer = Timer()
    timer.start()

    with pytest.raises(RuntimeError) as e:
        timer.elapsed_time
    assert str(e.value) == "timer must be stopped."

    with pytest.raises(RuntimeError) as e:
        timer.start()
    assert str(e.value) == "timer is already started."
    
    timer.stop()

    with pytest.raises(RuntimeError) as e:
        timer.stop()
    assert str(e.value) == "timer is already stopped."
