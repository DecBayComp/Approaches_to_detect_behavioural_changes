import pytest
import os
import numpy as np

def create_fake_data():
    from numpy.random import default_rng
    rng = default_rng()

    time_column = np.arange(0,90,0.1).reshape(-1,1)
    nrows = len(time_column)
    behaviors = -np.ones((nrows, 6), dtype=int)
    for row, col in enumerate(rng.choice(6, size=nrows)):
        behaviors[row, col] = 1
    length = np.ones((nrows,1))
    xy = 0.05*rng.normal(size=(nrows,10))
    for row in range(nrows):
        xy[row,::2] += np.linspace(0,1,5)

    return np.hstack([time_column, behaviors, length, xy])    

@pytest.fixture(scope="session")
def raw_data_dir(tmp_path_factory):
    raw_data_dir = tmp_path_factory.mktemp("raw_data_dir")
    for tracker in ['t5', 't15']:
        os.mkdir(raw_data_dir / tracker)
        for line in [f'LINE_{i}' for i in range(2)]:
            os.mkdir(raw_data_dir / tracker / line)
            for protocol in ['p_8_45s1x30s0s#p_8_105s10x2s10s#n#n@100', 'p_3gradient1_45s1x30s0s#p_3gradient1_105s10x2s10s#n#n@100']:
                os.mkdir(raw_data_dir / tracker / line / protocol)
                for date_time in [f'datetime_{i}' for i in range(2)]:
                    os.mkdir(raw_data_dir / tracker / line / protocol / date_time)
                    for file_idx in range(5):
                        fn = raw_data_dir / tracker / line / protocol / date_time
                        fn /= f"Point_dynamics_{tracker}_{line}_{protocol}_larva_id_{date_time}_larva_number_{file_idx}.txt"
                        np.savetxt(str(fn), create_fake_data(), delimiter='\t', fmt=['%.3f']+6*['%d']+['%.4f']+10*['%.3f'])
    
    return raw_data_dir

@pytest.fixture(scope="session")
def workspace(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("pytest_workspace")
    return workspace

@pytest.fixture()
def mock_pool():
    class FakePool:
        def __init__(self, *args, **kwargs):
            pass
        def starmap(self, func, iter_args, *args, **kwargs):
            result = [func(*args) for args in iter_args]
            return result
        def join(self):
            pass
        def close(self):
            pass
        def terminate(self):
            pass

    return FakePool

